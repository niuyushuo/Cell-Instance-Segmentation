def SingleTrainer(config):
    import os
    import json
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from datasets.data_bccd import BCCDMaskDataset
    from datasets.create_split import get_common_stems, split_train_val
    from misc.metric_tool import ConfuseMatrixMeter
    from models.losses import segmentation_loss, masked_smooth_l1_loss
    from models.nets import get_scheduler
    from models.uni_segmentor import UNISegmentor, create_uni_encoder

    seed = int(config.get("seed", 8888))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    split_file = config.get("split_file")
    val_ratio = float(config.get("val_ratio", 0.2))
    train_stems = None
    val_stems = None

    if split_file is not None and os.path.exists(split_file):
        with open(split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        train_stems = split_data["train_stems"]
        val_stems = split_data["val_stems"]
        print(f"Loaded train/val split from {split_file}")
    else:
        common_stems = get_common_stems(config["train_original"], config["train_mask"])
        train_stems, val_stems = split_train_val(common_stems, val_ratio=val_ratio, seed=seed)
        print(f"Created random split from train set: ratio={val_ratio}, seed={seed}")
        if split_file is not None:
            split_dir = os.path.dirname(split_file)
            if split_dir:
                os.makedirs(split_dir, exist_ok=True)
            split_data = {
                "train_original": config["train_original"],
                "train_mask": config["train_mask"],
                "seed": seed,
                "val_ratio": val_ratio,
                "num_total": len(common_stems),
                "num_train": len(train_stems),
                "num_val": len(val_stems),
                "train_stems": train_stems,
                "val_stems": val_stems,
            }
            with open(split_file, "w", encoding="utf-8") as f:
                json.dump(split_data, f, indent=2)
            print(f"Saved split to {split_file}")

    train_dataset = BCCDMaskDataset(
        image_dir=config["train_original"],
        mask_dir=config["train_mask"],
        edt_dir=config.get("train_edt"),
        stems=train_stems,
        img_size=config.get("img_size", 224),
        is_train=True,
    )

    val_dataset = BCCDMaskDataset(
        image_dir=config["train_original"],
        mask_dir=config["train_mask"],
        edt_dir=config.get("train_edt"),
        stems=val_stems,
        img_size=config.get("img_size", 224),
        is_train=False,
    )

    test_dataset = None
    if bool(config.get("test_original")) and bool(config.get("test_mask")):
        test_dataset = BCCDMaskDataset(
            image_dir=config["test_original"],
            mask_dir=config["test_mask"],
            edt_dir=config.get("test_edt"),
            img_size=config.get("img_size", 224),
            is_train=False,
        )

    trainloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
    )
    testloader = None
    if test_dataset is not None:
        testloader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            pin_memory=torch.cuda.is_available(),
        )

    encoder = config.get("encoder")
    if encoder is None:
        encoder = create_uni_encoder(
            enc_name=config.get("enc_name", "uni2-h"),
            checkpoint=config.get("encoder_ckpt"),
            use_hf_pretrained=config.get("use_hf_pretrained", True),
        )

    net_G = UNISegmentor(
        encoder=encoder,
        embed_dim=config.get("embed_dim", 1536),
        num_classes=2,
        reg_tokens=config.get("reg_tokens", 8),
        freeze_encoder=config.get("freeze_encoder", True),
        predict_distance=config.get("dist_weight", 0.0) > 0,
    )

    trainable_params = [p for p in net_G.parameters() if p.requires_grad]
    optimizer_G = optim.AdamW(
        trainable_params,
        lr=config["lr"],
        betas=(0.9, 0.999),
        weight_decay=config["wd"],
    )

    epoch_num = int(config.get("max_num_epochs", 100))
    exp_lr_scheduler_G = get_scheduler(optimizer_G, epoch_num)
    running_metric = ConfuseMatrixMeter(n_class=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.device_count() > 1:
        net_G = nn.DataParallel(net_G)
    net_G.to(device)

    print("Training from scratch...")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    if test_dataset is not None:
        print(f"Test samples: {len(test_dataset)}")
    print(f"Device: {device}")

    best_metrics = {"loss": float("inf"), "f1": 0.0, "f1_0": 0.0, "f1_1": 0.0}
    best_test_metrics = None
    output_dir = config.get("output_dir", "checkpoints_single")
    os.makedirs(output_dir, exist_ok=True)

    dist_weight = float(config.get("dist_weight", 0.0))
    ce_weight = float(config.get("ce_weight", 1.0))
    dice_weight = float(config.get("dice_weight", 1.0))

    def run_eval(loader):
        net_G.eval()
        running_metric.clear()
        sum_total = 0.0
        sum_seg = 0.0
        sum_dist = 0.0

        for batch in loader:
            with torch.no_grad():
                img = batch["I"].to(device)
                gt_mask = batch["M"].to(device)
                gt_dist = batch["D"].to(device)
                has_dist = batch["has_dist"].to(device)

                out = net_G(img)
                seg_logits = out["seg_logits"]
                seg_loss, _ = segmentation_loss(
                    seg_logits, gt_mask, ce_weight=ce_weight, dice_weight=dice_weight
                )

                dist_loss = torch.tensor(0.0, device=device)
                if "dist_pred" in out and dist_weight > 0:
                    dist_loss = masked_smooth_l1_loss(out["dist_pred"], gt_dist, valid_mask=has_dist)

                total_loss = seg_loss + dist_weight * dist_loss
                pred = torch.argmax(seg_logits.detach(), dim=1)
                running_metric.update_cm(pr=pred.cpu().numpy(), gt=gt_mask.cpu().numpy())

                sum_total += total_loss.item()
                sum_seg += seg_loss.item()
                sum_dist += dist_loss.item()

        n = len(loader)
        scores = running_metric.get_scores()
        return {
            "loss": sum_total / n,
            "seg_loss": sum_seg / n,
            "dist_loss": sum_dist / n,
            "scores": scores,
        }

    for epoch_id in range(epoch_num):
        net_G.train()
        running_metric.clear()
        train_loss = 0.0
        train_seg_loss = 0.0
        train_dist_loss = 0.0

        for batch in trainloader:
            img = batch["I"].to(device)
            gt_mask = batch["M"].to(device)
            gt_dist = batch["D"].to(device)
            has_dist = batch["has_dist"].to(device)

            out = net_G(img)
            seg_logits = out["seg_logits"]
            seg_loss, _ = segmentation_loss(
                seg_logits, gt_mask, ce_weight=ce_weight, dice_weight=dice_weight
            )

            dist_loss = torch.tensor(0.0, device=device)
            if "dist_pred" in out and dist_weight > 0:
                dist_loss = masked_smooth_l1_loss(out["dist_pred"], gt_dist, valid_mask=has_dist)

            total_loss = seg_loss + dist_weight * dist_loss

            optimizer_G.zero_grad()
            total_loss.backward()
            optimizer_G.step()

            pred = torch.argmax(seg_logits.detach(), dim=1)
            running_metric.update_cm(pr=pred.cpu().numpy(), gt=gt_mask.cpu().numpy())

            train_loss += total_loss.item()
            train_seg_loss += seg_loss.item()
            train_dist_loss += dist_loss.item()

        exp_lr_scheduler_G.step()
        train_scores = running_metric.get_scores()
        print(
            f"[Epoch {epoch_id}] Train Loss: {train_loss / len(trainloader):.4f} | "
            f"Seg: {train_seg_loss / len(trainloader):.4f} | "
            f"Dist: {train_dist_loss / len(trainloader):.4f} | "
            f"mf1: {train_scores['mf1']:.4f}"
        )

        val_result = run_eval(valloader)
        val_scores = val_result["scores"]
        f1_macro = val_scores["mf1"]
        f1_0 = val_scores.get("F1_0", -1)
        f1_1 = val_scores.get("F1_1", -1)

        print(
            f"[Epoch {epoch_id}] Val Loss: {val_result['loss']:.4f} | "
            f"Seg: {val_result['seg_loss']:.4f} | "
            f"Dist: {val_result['dist_loss']:.4f} | "
            f"F1_0: {f1_0:.4f} | F1_1: {f1_1:.4f} | mf1: {f1_macro:.4f}"
        )

        if f1_1 > best_metrics["f1_1"]:
            test_result = None
            if testloader is not None:
                test_result = run_eval(testloader)
                test_scores = test_result["scores"]
                best_test_metrics = {
                    "loss": test_result["loss"],
                    "f1": test_scores["mf1"],
                    "f1_0": test_scores.get("F1_0", -1),
                    "f1_1": test_scores.get("F1_1", -1),
                }
                print(
                    f"[Epoch {epoch_id}] Test@BestVal Loss: {test_result['loss']:.4f} | "
                    f"F1_0: {best_test_metrics['f1_0']:.4f} | "
                    f"F1_1: {best_test_metrics['f1_1']:.4f} | "
                    f"mf1: {best_test_metrics['f1']:.4f}"
                )

            best_metrics = {
                "loss": val_result["loss"],
                "f1": f1_macro,
                "f1_0": f1_0,
                "f1_1": f1_1,
            }
            ckpt_path = os.path.join(output_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch_id,
                    "model_state_dict": net_G.state_dict(),
                    "optimizer_state_dict": optimizer_G.state_dict(),
                    "scheduler_state_dict": exp_lr_scheduler_G.state_dict(),
                    "val_metrics": best_metrics,
                    "test_metrics": best_test_metrics,
                    "config": config,
                },
                ckpt_path,
            )
            print(f"Saved new best model at epoch {epoch_id} -> {ckpt_path}")

    return {
        "best_val": best_metrics,
        "best_test": best_test_metrics,
    }
