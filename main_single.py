from models.train_single import SingleTrainer
import os


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


if __name__ == "__main__":
    config = {
        "seed": 8888,
        "lr": 2e-4,
        "wd": 1e-4,
        "batch_size": 4,
        "img_size": 224,
        "max_num_epochs": 50,
        "num_workers": 4,
        "output_dir": "github_uni/checkpoints_single",
        # split setup (train folder -> train/val split)
        "split_file": "github_uni/datasets/splits/bccd_train_val_split.json",
        "val_ratio": 0.2,
        # encoder settings
        "enc_name": "vit_base_patch16_224",  # ungated fallback; switch to "uni2-h" when access is approved
        "encoder_ckpt": None,          # e.g. "assets/ckpts/uni2-h/pytorch_model.bin"
        "use_hf_pretrained": True,     # set False if using local checkpoint
        "freeze_encoder": True,        # start frozen, then unfreeze later if needed
        "embed_dim": None,             # auto-infer from encoder if possible
        "reg_tokens": 1,               # used as fallback for tokenized encoders
        # multi-task loss weight
        "dist_weight": 0.5,
        # dataset paths
        "train_original": "data/BCCD Dataset with mask/train/original",
        "train_mask": "data/BCCD Dataset with mask/train/mask",
        "train_edt": "data/BCCD Dataset with mask/train/edt",  # optional
        "test_original": "data/BCCD Dataset with mask/test/original",
        "test_mask": "data/BCCD Dataset with mask/test/mask",
        "test_edt": "data/BCCD Dataset with mask/test/edt",    # optional
    }

    result = SingleTrainer(config)
    print("\n=== Final Best Metrics ===")
    print(result)
