import argparse
import json
import os
from pathlib import Path
import sys
import shutil

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

import torch
from torch.utils.data import DataLoader

# Ensure sibling packages (datasets/models) are importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.data_bccd import BCCDMaskDataset
from models.uni_segmentor import UNISegmentor, create_uni_encoder


try:
    from skimage.segmentation import watershed as sk_watershed
except Exception:
    sk_watershed = None


def remove_small_instances(inst_map, area_min):
    if area_min <= 0:
        return inst_map
    out = np.zeros_like(inst_map, dtype=np.int32)
    current = 1
    for k in np.unique(inst_map):
        if k == 0:
            continue
        m = inst_map == k
        if int(m.sum()) >= area_min:
            out[m] = current
            current += 1
    return out


def _build_markers(distance_map, fg_mask, min_distance=11, threshold_abs=0.25):
    d = distance_map.copy()
    d[~fg_mask] = 0.0
    size = int(max(1, 2 * min_distance + 1))
    local_max = d == ndi.maximum_filter(d, size=size, mode="constant")
    peak_mask = local_max & fg_mask & (d >= threshold_abs)
    markers, _ = ndi.label(peak_mask)
    return markers


def _fallback_marker_assignment(markers, fg_mask):
    if markers.max() == 0:
        cc, _ = ndi.label(fg_mask)
        return cc.astype(np.int32)
    nearest_inds = ndi.distance_transform_edt(
        markers == 0, return_distances=False, return_indices=True
    )
    nearest_marker = markers[tuple(nearest_inds)]
    return (nearest_marker * fg_mask).astype(np.int32)


def instance_from_fg_and_distance(
    p_fg,
    d_pred,
    t_fg=0.5,
    d_sigma=1.0,
    min_distance=11,
    threshold_abs=0.25,
    area_min=20,
):
    fg_mask = p_fg > t_fg
    d_smooth = ndi.gaussian_filter(d_pred.astype(np.float32), sigma=d_sigma)
    markers = _build_markers(
        d_smooth, fg_mask, min_distance=min_distance, threshold_abs=threshold_abs
    )

    if sk_watershed is not None and markers.max() > 0:
        inst = sk_watershed(-d_smooth, markers, mask=fg_mask).astype(np.int32)
    else:
        inst = _fallback_marker_assignment(markers, fg_mask)

    inst = remove_small_instances(inst, area_min=area_min)
    return inst


def connected_components_from_mask(binary_mask, area_min=1):
    labeled, _ = ndi.label(binary_mask.astype(np.uint8) > 0)
    return remove_small_instances(labeled.astype(np.int32), area_min=area_min)


def compute_semantic_metrics(pred_fg, gt_fg):
    pred_fg = pred_fg.astype(bool)
    gt_fg = gt_fg.astype(bool)

    tp = int(np.logical_and(pred_fg, gt_fg).sum())
    tn = int(np.logical_and(~pred_fg, ~gt_fg).sum())
    fp = int(np.logical_and(pred_fg, ~gt_fg).sum())
    fn = int(np.logical_and(~pred_fg, gt_fg).sum())

    eps = 1e-8
    f1_fg = (2 * tp) / (2 * tp + fp + fn + eps)
    iou_fg = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "f1_fg": float(f1_fg),
        "dice_fg": float(f1_fg),
        "iou_fg": float(iou_fg),
        "acc": float(acc),
    }


def _instance_areas(inst_map):
    ids, counts = np.unique(inst_map, return_counts=True)
    area = {}
    for i, c in zip(ids, counts):
        if i == 0:
            continue
        area[int(i)] = int(c)
    return area


def _pair_intersections(gt_inst, pred_inst):
    gt_flat = gt_inst.reshape(-1)
    pred_flat = pred_inst.reshape(-1)
    valid = (gt_flat > 0) & (pred_flat > 0)
    gt_v = gt_flat[valid].astype(np.int64)
    pred_v = pred_flat[valid].astype(np.int64)

    if gt_v.size == 0:
        return {}

    pair_codes = gt_v * (pred_v.max() + 1) + pred_v
    unique_codes, counts = np.unique(pair_codes, return_counts=True)
    intersections = {}
    mul = int(pred_v.max() + 1)
    for code, cnt in zip(unique_codes, counts):
        gid = int(code // mul)
        pid = int(code % mul)
        intersections[(gid, pid)] = int(cnt)
    return intersections


def match_instances(gt_inst, pred_inst, iou_thr=0.5):
    gt_area = _instance_areas(gt_inst)
    pred_area = _instance_areas(pred_inst)
    intersections = _pair_intersections(gt_inst, pred_inst)

    pair_scores = []
    for (gid, pid), inter in intersections.items():
        union = gt_area[gid] + pred_area[pid] - inter
        iou = inter / (union + 1e-8)
        pair_scores.append((iou, gid, pid, inter, union))
    pair_scores.sort(key=lambda x: x[0], reverse=True)

    used_g = set()
    used_p = set()
    matched = []
    for iou, gid, pid, inter, union in pair_scores:
        if iou < iou_thr:
            continue
        if gid in used_g or pid in used_p:
            continue
        used_g.add(gid)
        used_p.add(pid)
        matched.append(
            {"gid": gid, "pid": pid, "iou": float(iou), "inter": int(inter), "union": int(union)}
        )

    n_gt = len(gt_area)
    n_pred = len(pred_area)
    n_tp = len(matched)
    n_fp = n_pred - n_tp
    n_fn = n_gt - n_tp

    precision = n_tp / (n_tp + n_fp + 1e-8)
    recall = n_tp / (n_tp + n_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    mean_matched_iou = (
        float(np.mean([m["iou"] for m in matched])) if len(matched) > 0 else 0.0
    )

    matched_gids = {m["gid"] for m in matched}
    matched_pids = {m["pid"] for m in matched}
    inter_sum = sum(m["inter"] for m in matched)
    union_sum = sum(m["union"] for m in matched)
    unpaired_gt = sum(v for k, v in gt_area.items() if k not in matched_gids)
    unpaired_pred = sum(v for k, v in pred_area.items() if k not in matched_pids)
    aji = inter_sum / (union_sum + unpaired_gt + unpaired_pred + 1e-8)

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "tp": n_tp,
        "fp": n_fp,
        "fn": n_fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_matched_iou": mean_matched_iou,
        "aji": float(aji),
    }


def _pred_instance_scores(pred_inst, score_map):
    scores = {}
    ids = np.unique(pred_inst)
    for pid in ids:
        if pid == 0:
            continue
        m = pred_inst == pid
        scores[int(pid)] = float(score_map[m].mean()) if m.any() else 0.0
    return scores


def _detection_tp_fp_for_image(gt_inst, pred_inst, pred_scores, iou_thr):
    gt_area = _instance_areas(gt_inst)
    pred_area = _instance_areas(pred_inst)
    intersections = _pair_intersections(gt_inst, pred_inst)

    pid_to_pairs = {}
    for (gid, pid), inter in intersections.items():
        union = gt_area[gid] + pred_area[pid] - inter
        iou = inter / (union + 1e-8)
        pid_to_pairs.setdefault(pid, []).append((iou, gid))
    for pid in pid_to_pairs:
        pid_to_pairs[pid].sort(reverse=True, key=lambda x: x[0])

    pred_ids = sorted(pred_area.keys(), key=lambda pid: pred_scores.get(pid, 0.0), reverse=True)
    matched_gt = set()
    scores = []
    tps = []
    fps = []
    for pid in pred_ids:
        best_iou = 0.0
        best_gid = None
        for iou, gid in pid_to_pairs.get(pid, []):
            if gid in matched_gt:
                continue
            if iou > best_iou:
                best_iou = iou
                best_gid = gid
        is_tp = int(best_iou >= iou_thr and best_gid is not None)
        if is_tp:
            matched_gt.add(best_gid)
        scores.append(float(pred_scores.get(pid, 0.0)))
        tps.append(is_tp)
        fps.append(1 - is_tp)
    return scores, tps, fps, len(gt_area)


def _average_precision(scores, tps, fps, num_gt):
    if num_gt <= 0 or len(scores) == 0:
        return 0.0

    scores = np.asarray(scores, dtype=np.float32)
    tps = np.asarray(tps, dtype=np.float32)
    fps = np.asarray(fps, dtype=np.float32)

    order = np.argsort(-scores)
    tps = tps[order]
    fps = fps[order]

    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)
    recall = tp_cum / (num_gt + 1e-8)
    precision = tp_cum / (tp_cum + fp_cum + 1e-8)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.shape[0] - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def _instance_boundaries(inst_map):
    b = np.zeros_like(inst_map, dtype=bool)
    ids = np.unique(inst_map)
    for idx in ids:
        if idx == 0:
            continue
        m = inst_map == idx
        eroded = ndi.binary_erosion(m)
        b |= np.logical_and(m, ~eroded)
    return b


def _make_overlay(image_rgb, gt_sem, pred_sem, gt_inst, pred_inst, alpha=0.25):
    canvas = image_rgb.astype(np.float32).copy()
    gt_sem = gt_sem.astype(bool)
    pred_sem = pred_sem.astype(bool)

    # Green tint for GT foreground
    canvas[..., 1] = np.where(gt_sem, canvas[..., 1] * (1 - alpha) + 255 * alpha, canvas[..., 1])
    # Red tint for predicted foreground
    canvas[..., 0] = np.where(pred_sem, canvas[..., 0] * (1 - alpha) + 255 * alpha, canvas[..., 0])

    gt_b = _instance_boundaries(gt_inst)
    pred_b = _instance_boundaries(pred_inst)
    canvas[gt_b] = np.array([0, 255, 0], dtype=np.float32)
    canvas[pred_b] = np.array([255, 0, 0], dtype=np.float32)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _instance_to_color(inst_map, seed=42):
    h, w = inst_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    ids = np.unique(inst_map)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return color

    rng = np.random.default_rng(seed)
    palette = rng.integers(40, 255, size=(int(ids.max()) + 1, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    color = palette[inst_map]
    return color


def _mask_to_rgb(mask_u8):
    m = (mask_u8 > 0).astype(np.uint8) * 255
    return np.stack([m, m, m], axis=-1)


def _make_quad_panel(input_rgb, label_rgb, overlay_rgb, inst_color_rgb):
    h, w = input_rgb.shape[:2]
    label_rgb = label_rgb[:h, :w]
    overlay_rgb = overlay_rgb[:h, :w]
    inst_color_rgb = inst_color_rgb[:h, :w]
    top = np.concatenate([input_rgb, label_rgb], axis=1)
    bot = np.concatenate([overlay_rgb, inst_color_rgb], axis=1)
    return np.concatenate([top, bot], axis=0)


def _mean_dict(list_of_dicts, keys):
    out = {}
    for k in keys:
        vals = [d[k] for d in list_of_dicts]
        out[k] = float(np.mean(vals)) if len(vals) > 0 else 0.0
    return out


def _resolve_split(cfg, split, split_file):
    if split == "test":
        return {
            "image_dir": cfg["test_original"],
            "mask_dir": cfg["test_mask"],
            "edt_dir": cfg.get("test_edt"),
            "stems": None,
        }

    if split_file is None or not os.path.exists(split_file):
        raise RuntimeError(
            f"Split file not found for split='{split}': {split_file}. "
            "Please provide a valid --split_file."
        )

    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    if split == "val":
        stems = split_data["val_stems"]
    elif split == "train":
        stems = split_data["train_stems"]
    else:
        raise ValueError(f"Unsupported split: {split}")

    return {
        "image_dir": cfg["train_original"],
        "mask_dir": cfg["train_mask"],
        "edt_dir": cfg.get("train_edt"),
        "stems": stems,
    }


def _strip_module_prefix(state_dict):
    keys = list(state_dict.keys())
    if len(keys) > 0 and keys[0].startswith("module."):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    except Exception as e:
        # PyTorch >=2.6 defaults to weights_only=True and may fail on full checkpoints.
        if "Weights only load failed" in str(e):
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        else:
            raise
    cfg = ckpt.get("config", {})
    if not cfg:
        raise RuntimeError("Checkpoint does not contain config. Please pass --config_json.")

    split_file = args.split_file or cfg.get("split_file")
    split_info = _resolve_split(cfg, split=args.split, split_file=split_file)

    dataset = BCCDMaskDataset(
        image_dir=split_info["image_dir"],
        mask_dir=split_info["mask_dir"],
        edt_dir=split_info["edt_dir"],
        stems=split_info["stems"],
        img_size=int(cfg.get("img_size", 224)),
        is_train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    encoder = create_uni_encoder(
        enc_name=cfg.get("enc_name", "vit_base_patch16_224"),
        checkpoint=cfg.get("encoder_ckpt"),
        use_hf_pretrained=False,
    )
    model = UNISegmentor(
        encoder=encoder,
        embed_dim=cfg.get("embed_dim"),
        num_classes=2,
        reg_tokens=cfg.get("reg_tokens", 1),
        freeze_encoder=False,
        predict_distance=True,
    )

    state_dict = _strip_module_prefix(ckpt["model_state_dict"])
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    save_dir = Path(args.save_dir)
    sem_dir = save_dir / "pred_semantic"
    inst_dir = save_dir / "pred_instance"
    inst_color_dir = save_dir / "pred_instance_color"
    overlay_dir = save_dir / "overlays"
    panel_dir = save_dir / "panels_input_label_overlay_instance"
    qual_success_dir = save_dir / "qualitative" / "success"
    qual_failure_dir = save_dir / "qualitative" / "failure"
    sem_dir.mkdir(parents=True, exist_ok=True)
    inst_dir.mkdir(parents=True, exist_ok=True)
    inst_color_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    qual_success_dir.mkdir(parents=True, exist_ok=True)
    qual_failure_dir.mkdir(parents=True, exist_ok=True)

    semantic_logs = []
    instance_logs = []
    per_image_logs = []
    ap_thresholds = np.arange(0.5, 1.0, 0.05)
    ap_buffers = {
        float(thr): {"scores": [], "tp": [], "fp": [], "num_gt": 0}
        for thr in ap_thresholds
    }

    with torch.no_grad():
        for batch in loader:
            img = batch["I"].to(device)
            gt_mask = batch["M"].cpu().numpy().astype(np.uint8)
            stems = batch["stem"]
            image_paths = batch["image_path"]

            out = model(img)
            p_fg = torch.softmax(out["seg_logits"], dim=1)[:, 1].cpu().numpy()
            if "dist_pred" in out and not args.disable_dist:
                d_pred = out["dist_pred"][:, 0].cpu().numpy()
            else:
                d_pred = p_fg

            for i in range(p_fg.shape[0]):
                stem = stems[i]
                pred_sem = (p_fg[i] > args.t_fg).astype(np.uint8)
                pred_inst = instance_from_fg_and_distance(
                    p_fg=p_fg[i],
                    d_pred=d_pred[i],
                    t_fg=args.t_fg,
                    d_sigma=args.d_sigma,
                    min_distance=args.min_distance,
                    threshold_abs=args.peak_threshold,
                    area_min=args.area_min,
                )

                gt_sem = gt_mask[i]
                gt_inst = connected_components_from_mask(gt_sem, area_min=args.gt_area_min)

                sem_m = compute_semantic_metrics(pred_sem, gt_sem)
                ins_m = match_instances(gt_inst, pred_inst, iou_thr=args.iou_thr)
                pred_scores = _pred_instance_scores(pred_inst, p_fg[i])

                ap50_scores, ap50_tp, ap50_fp, ap50_n_gt = _detection_tp_fp_for_image(
                    gt_inst, pred_inst, pred_scores, iou_thr=0.50
                )
                ap50_img = _average_precision(ap50_scores, ap50_tp, ap50_fp, ap50_n_gt)

                semantic_logs.append(sem_m)
                instance_logs.append(ins_m)
                per_image_logs.append(
                    {
                        "stem": stem,
                        "dice_fg": sem_m["dice_fg"],
                        "iou_fg": sem_m["iou_fg"],
                        "ap50": ap50_img,
                        "inst_f1": ins_m["f1"],
                        "aji": ins_m["aji"],
                    }
                )

                for thr in ap_thresholds:
                    scores_t, tp_t, fp_t, n_gt_t = _detection_tp_fp_for_image(
                        gt_inst, pred_inst, pred_scores, iou_thr=float(thr)
                    )
                    bucket = ap_buffers[float(thr)]
                    bucket["scores"].extend(scores_t)
                    bucket["tp"].extend(tp_t)
                    bucket["fp"].extend(fp_t)
                    bucket["num_gt"] += int(n_gt_t)

                Image.fromarray((pred_sem * 255).astype(np.uint8), mode="L").save(
                    sem_dir / f"{stem}.png"
                )
                inst_u16 = np.clip(pred_inst, 0, 65535).astype(np.uint16)
                Image.fromarray(inst_u16, mode="I;16").save(inst_dir / f"{stem}.png")
                inst_color = _instance_to_color(inst_u16, seed=42)
                Image.fromarray(inst_color, mode="RGB").save(inst_color_dir / f"{stem}.png")

                raw_img = Image.open(image_paths[i]).convert("RGB")
                raw_img = raw_img.resize((pred_sem.shape[1], pred_sem.shape[0]), resample=Image.BILINEAR)
                raw_np = np.array(raw_img, dtype=np.uint8)
                overlay_np = _make_overlay(raw_np, gt_sem, pred_sem, gt_inst, pred_inst)
                Image.fromarray(overlay_np, mode="RGB").save(overlay_dir / f"{stem}.png")
                label_rgb = _mask_to_rgb(gt_sem)
                panel = _make_quad_panel(raw_np, label_rgb, overlay_np, inst_color)
                Image.fromarray(panel, mode="RGB").save(panel_dir / f"{stem}.png")

    ap_by_thr = {}
    for thr in ap_thresholds:
        bucket = ap_buffers[float(thr)]
        ap = _average_precision(
            scores=bucket["scores"],
            tps=bucket["tp"],
            fps=bucket["fp"],
            num_gt=bucket["num_gt"],
        )
        ap_by_thr[f"AP@{thr:.2f}"] = ap

    map_50_95 = float(np.mean(list(ap_by_thr.values()))) if len(ap_by_thr) > 0 else 0.0
    ap50 = ap_by_thr.get("AP@0.50", 0.0)
    ap75 = ap_by_thr.get("AP@0.75", 0.0)

    sem_keys = ["dice_fg", "iou_fg", "acc"]
    ins_keys = [
        "precision",
        "recall",
        "f1",
        "mean_matched_iou",
        "aji",
        "n_gt",
        "n_pred",
    ]
    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "num_images": len(semantic_logs),
        "semantic": _mean_dict(semantic_logs, sem_keys),
        "instance": _mean_dict(instance_logs, ins_keys),
        "detection": {
            "mAP_50_95": map_50_95,
            "AP@0.50": ap50,
            "AP@0.75": ap75,
            "AP_by_threshold": ap_by_thr,
        },
        "postprocess": {
            "t_fg": args.t_fg,
            "d_sigma": args.d_sigma,
            "min_distance": args.min_distance,
            "peak_threshold": args.peak_threshold,
            "area_min": args.area_min,
            "iou_thr": args.iou_thr,
            "disable_dist": args.disable_dist,
        },
        "save_dir": str(save_dir),
    }

    per_image_sorted_best = sorted(per_image_logs, key=lambda x: x["dice_fg"], reverse=True)
    per_image_sorted_worst = sorted(per_image_logs, key=lambda x: x["dice_fg"])
    best_cases = per_image_sorted_best[: args.qual_topk]
    worst_cases = per_image_sorted_worst[: args.qual_topk]

    for rank, item in enumerate(best_cases, 1):
        stem = item["stem"]
        shutil.copy2(overlay_dir / f"{stem}.png", qual_success_dir / f"{rank:02d}_{stem}.png")
    for rank, item in enumerate(worst_cases, 1):
        stem = item["stem"]
        shutil.copy2(overlay_dir / f"{stem}.png", qual_failure_dir / f"{rank:02d}_{stem}.png")

    (save_dir / "per_image_metrics.json").write_text(json.dumps(per_image_logs, indent=2))
    (save_dir / "qualitative_cases.json").write_text(
        json.dumps({"success": best_cases, "failure": worst_cases}, indent=2)
    )

    out_json = save_dir / "metrics_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("=== Instance Segmentation Evaluation ===")
    print(f"split={summary['split']} | num_images={summary['num_images']}")
    print(
        "semantic: "
        f"Dice_fg={summary['semantic']['dice_fg']:.4f}, "
        f"IoU_fg={summary['semantic']['iou_fg']:.4f}, "
        f"Acc={summary['semantic']['acc']:.4f}"
    )
    print(
        "instance: "
        f"F1@IoU{args.iou_thr:.2f}={summary['instance']['f1']:.4f}, "
        f"AJI={summary['instance']['aji']:.4f}, "
        f"mIoU(matched)={summary['instance']['mean_matched_iou']:.4f}, "
        f"P={summary['instance']['precision']:.4f}, "
        f"R={summary['instance']['recall']:.4f}"
    )
    print(
        "detection: "
        f"mAP(0.50:0.95)={summary['detection']['mAP_50_95']:.4f}, "
        f"AP50={summary['detection']['AP@0.50']:.4f}, "
        f"AP75={summary['detection']['AP@0.75']:.4f}"
    )
    print(
        "qualitative: "
        f"success={qual_success_dir}, failure={qual_failure_dir}"
    )
    print(f"Saved summary -> {out_json}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run instance segmentation + evaluation from trained checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="github_uni/checkpoints_single/best_model.pt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which data split to evaluate.",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Optional split JSON path (required for train/val if not in checkpoint config).",
    )
    parser.add_argument("--save_dir", type=str, default="github_uni/checkpoints_single/instance_eval_test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")

    # postprocess parameters
    parser.add_argument("--t_fg", type=float, default=0.5, help="Foreground threshold.")
    parser.add_argument("--d_sigma", type=float, default=1.0, help="Gaussian sigma on distance map.")
    parser.add_argument("--min_distance", type=int, default=11, help="Minimum peak distance.")
    parser.add_argument("--peak_threshold", type=float, default=0.25, help="Minimum peak value.")
    parser.add_argument("--area_min", type=int, default=20, help="Min predicted instance size.")
    parser.add_argument("--gt_area_min", type=int, default=1, help="Min GT instance size after CC labeling.")
    parser.add_argument("--iou_thr", type=float, default=0.5, help="IoU threshold for instance F1.")
    parser.add_argument("--qual_topk", type=int, default=3, help="Number of best/worst qualitative examples.")
    parser.add_argument("--disable_dist", action="store_true", help="Ignore predicted distance head.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
