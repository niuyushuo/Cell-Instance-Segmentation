# Blood Cell Instance Segmentation (BCCD) with UNI2‑h Pathology Foundation Backbone

This project adapts a pathology foundation-model style encoder for cell segmentation on BCCD, then uses distance-guided watershed post-processing to obtain **instance masks**.

## 1) Model adaptation and design choices

- **Backbone**:
  - Current reported results are based on **UNI2‑h (pathology foundation backbone; access‑restricted on HF)**.
  - Earlier baseline runs used the open timm backbone `vit_base_patch16_224`; architecture is the same but pretrained weights differ.
  - The downstream training/evaluation pipeline is kept the same, so swapping weights is clean and comparable.
- **Token-to-feature adaptation**:
  - for ViT encoders, token outputs are reshaped to a 2D feature map
  - prefix/register tokens are removed before spatial decoding
- **Decoder design** (implemented in `models/uni_segmentor.py`):
  - `1x1` projection from encoder embedding space to decoder channels
  - three-stage upsampling decoder (`F.interpolate` bilinear upsample + Conv-BN-ReLU)
  - final resize to original image resolution
  - normalization in decoder uses **BatchNorm**
  - we intentionally avoid deconvolution/transposed-conv blocks in this baseline
- **Why interpolate + Conv-BN-ReLU instead of deconv**:
  - deconvolution can produce checkerboard artifacts, which are undesirable near thin cell boundaries
  - bilinear interpolation is smoother and more stable for boundary-sensitive pathology masks
  - lighter decoder reduces parameter count and improves training/inference efficiency (important for iterative experiments)
- **Dual-head output**:
  - both heads take the **shared decoded feature map** as input
  - semantic foreground logits (`seg_logits`): optimize pixel-level classification (cell vs background)
  - normalized distance map (`dist_pred`, EDT target): provide shape/separation cues for touching-cell splitting
- **Instance segmentation**:
  - threshold semantic foreground
  - smooth distance map
  - local-peak markers
  - watershed to split touching cells

Why this is pathology-relevant:
- Cell boundaries are often weak and cells overlap; semantic masks alone can merge nearby nuclei/cells.
- EDT-guided post-processing improves separation of adjacent objects, which is critical for instance-level morphology analysis.
- A conservative decoder with strong pretrained features is often preferable for cell segmentation, where annotation noise and morphology variability can destabilize large task-specific heads.
- Lightweight design keeps training cost lower while still preserving useful morphology cues.

## 2) Data setup and preprocessing

Dataset: BCCD with masks.

Image preprocessing + augmentation strategy:
- Keep provided `test/` as hold-out test set.
- Split only `train/` into train/val via `create_split.py`.
- Resize to model patch-compatible size (`224x224`) for ViT patch embedding.
- Normalize image channels with ImageNet-style mean/std to match pretrained encoder statistics.
- Generate EDT labels from mask foreground using `generate_edt.py`.
- Training-time augmentations (intentionally conservative):
  - horizontal/vertical flip
  - 90/180/270 rotation
  - no heavy color jitter, no elastic deformation, no aggressive random crop scale

Why conservative augmentation is used here (pathology cell segmentation context):
- Cell morphology (shape/size/texture) is the signal; strong geometric or photometric distortions can corrupt nuclei/cell appearance and harm boundary learning.
- Histopathology/cytology stain variation is real, but excessive synthetic color shifts can produce unrealistic distributions and reduce generalization.
- Rotation/flip are label-preserving and biologically plausible for patch-level cell images, giving robust gains with low risk.
- With a pretrained foundation backbone and limited dataset size, conservative augmentation usually gives better stability than aggressive augmentation.

Why we did **not** use explicit stain normalization in this baseline:
- We prioritized a stable, reproducible baseline with minimal moving parts first.
- BCCD images are already relatively consistent compared with multi-center histopathology cohorts.
- Heavy stain normalization can sometimes shift subtle cytology textures and introduce preprocessing artifacts.
- Given limited experiment budget, we deferred stain-normalization ablations to future work (Reinhard/Macenko are listed as next steps).

Commands:

```bash
python3 github_uni/datasets/create_split.py \
  --train_original "data/BCCD Dataset with mask/train/original" \
  --train_mask "data/BCCD Dataset with mask/train/mask" \
  --val_ratio 0.2 --seed 8888 \
  --out_json "github_uni/datasets/splits/bccd_train_val_split.json"

python3 github_uni/datasets/generate_edt.py \
  --dataset_root "data/BCCD Dataset with mask" \
  --d_max 15
```

## 3) Training procedure and hyperparameters

Train command:

```bash
python3 github_uni/main_single.py
```

### Local UNI2‑h checkpoint (offline / fixed-path option)

If you want a fixed local checkpoint (no online download at training time):

```bash
export HF_TOKEN=your_hf_token_here
python3 - <<'PY'
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download(
    "MahmoodLab/UNI2-h",
    filename="pytorch_model.bin",
    local_dir="github_uni/assets/ckpts/uni2-h",
    local_dir_use_symlinks=False,
)
print("saved:", ckpt)
PY
```

Then update `github_uni/main_single.py`:

```python
"enc_name": "uni2-h",
"encoder_ckpt": "github_uni/assets/ckpts/uni2-h/pytorch_model.bin",
"use_hf_pretrained": False,
```

Baseline hyperparameters:
- `seed=8888`
- `lr=2e-4`
- `weight_decay=1e-4`
- `batch_size=4`
- `img_size=224`
- `max_num_epochs=50`
- `dist_weight=0.5`
- train/val split file: `github_uni/datasets/splits/bccd_train_val_split.json`
- optimizer: `AdamW`
- LR scheduler: linear decay (`LambdaLR`)

Losses:
- **Segmentation loss**: `CrossEntropy + Dice`
- **Distance loss**: `SmoothL1`
- **Total**: `L = L_seg + dist_weight * L_dist`

Hyperparameter tuning notes:
- We fix a strong baseline (`lr=2e-4`, `wd=1e-4`, `dist_weight=0.5`) and keep it for UNI2-h.
- Earlier vit_base ablations with lower LR/WD did not improve results; to keep the report concise we only retain the best vit_base baseline.
- Tuning objective is explicit: maintain strong semantic Dice/IoU while improving AP/AJI (instance quality).
- In practice, tuning must balance the two heads:
  - overly large `dist_weight` can over-emphasize splitting and increase false positives/over-segmentation
  - overly small `dist_weight` can weaken touching-cell separation
  - we therefore monitor semantic (`Dice/IoU`) and instance (`AP/AJI/F1@0.5`) metrics jointly.

## 4) Quantitative results

### 4.1 Best training metrics (vit_base baseline)

From the best open timm baseline (`vit_base_patch16_224`):

- best validation:
  - loss: `0.2977`
  - macro F1: `0.9238`
  - foreground Dice/F1 (`F1_1`): `0.8945`
- best test-at-best-val:
  - loss: `0.3001`
  - macro F1: `0.9240`
  - foreground Dice/F1 (`F1_1`): `0.8958`

Derived foreground IoU from Dice (`IoU = Dice / (2 - Dice)`):
- test IoU ≈ `0.8114`

Note: UNI2-h evaluation results are reported in Section 4.2 (instance metrics + semantic Dice/IoU on test).

### 4.2 Instance metrics (IoU, Dice, mAP)

Run (UNI2-h, Scheme A default):

```bash
python3 github_uni/eval/evaluate_instance.py \
  --checkpoint github_uni/checkpoints_uni1/best_model.pt \
  --split test \
  --save_dir github_uni/checkpoints_uni1/instance_eval_test_A \
  --t_fg 0.55 --d_sigma 1.2 --min_distance 13 --peak_threshold 0.30 --area_min 30
```

Scheme B (trade-off; higher AP but lower AJI/F1):

```bash
python3 github_uni/eval/evaluate_instance.py \
  --checkpoint github_uni/checkpoints_uni1/best_model.pt \
  --split test \
  --save_dir github_uni/checkpoints_uni1/instance_eval_test_B \
  --t_fg 0.55 --d_sigma 1.0 --min_distance 12 --peak_threshold 0.32 --area_min 25
```

Best open timm baseline (vit_base) for comparison:

```bash
python3 github_uni/eval/evaluate_instance.py \
  --checkpoint github_uni/checkpoints_single/best_model.pt \
  --split test \
  --save_dir github_uni/checkpoints_single/instance_eval_test \
  --t_fg 0.5 --d_sigma 1.0 --min_distance 11 --peak_threshold 0.25 --area_min 20
```

This writes:
- `<save_dir>/metrics_summary.json`
- `pred_semantic/*.png`
- `pred_instance/*.png`
- `pred_instance_color/*.png`
- `overlays/*.png`
- `panels_input_label_overlay_instance/*.png` (2x2 panel for each image: input, label, overlay, instance-color)
- qualitative selections in:
  - `qualitative/success`
  - `qualitative/failure`

Merged test results (all in one table):

| Run | Encoder | Postprocess | Dice | IoU | AP50 | AP75 | mAP(0.50:0.95) | Inst F1@0.5 | AJI |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| UNI2-h (Scheme A, best) | `uni2-h` | `t_fg=0.55`, `d_sigma=1.2`, `min_distance=13`, `peak_threshold=0.30`, `area_min=30` | **0.9357** | **0.8797** | **0.6382** | **0.5338** | **0.4772** | **0.7148** | **0.4746** |
| UNI2-h (Scheme B, trade-off) | `uni2-h` | `t_fg=0.55`, `d_sigma=1.0`, `min_distance=12`, `peak_threshold=0.32`, `area_min=25` | 0.9357 | 0.8797 | 0.6438 | 0.5391 | 0.4809 | 0.7087 | 0.4647 |
| Best open timm baseline | `vit_base_patch16_224` | `t_fg=0.5`, `d_sigma=1.0`, `min_distance=11`, `peak_threshold=0.25`, `area_min=20` | 0.8968 | 0.8144 | 0.5713 | 0.4196 | 0.3549 | 0.6752 | 0.3987 |

Trade-off summary:
- **Scheme A (best/balanced)**: higher stability and better AJI/F1, with strong AP.
- **Scheme B (trade-off)**: slightly higher AP50/AP75/mAP, but lower AJI/F1 and more fragmented instances.

Metrics included in `metrics_summary.json`:
- Semantic: `Dice`, `IoU`, `Acc`
- Detection/instance: `mAP(0.50:0.95)`, `AP50`, `AP75`
- Watershed instance quality: `F1@IoU0.5`, `AJI`, `precision`, `recall`

## 5) Qualitative visualization and error analysis

At least 3-5 visual examples are exported automatically:
- top-performing cases by Dice: `qualitative/success`
- failure cases by Dice: `qualitative/failure`

Overlay color coding:
- green tint/boundary: ground-truth objects
- red tint/boundary: predicted objects

Typical trade-offs to discuss:
- **High Dice/IoU but lower mAP**: foreground region captured, but touching cells still merged.
- **Higher mAP but moderate Dice**: object splitting improved, but some boundary pixels noisy.
- **Failure modes**: clumped cells, weak contrast, tiny objects, heavy stain/background variation.

Metric trade-offs in the context of cell morphology:
- Cell overlap and adhesion make boundary assignment ambiguous; a model can achieve high Dice while still underperforming on instance mAP/AJI due to merge errors.
- Small round cells are sensitive to 1-2 pixel boundary shifts; this may only mildly affect semantic IoU but can flip instance matching outcomes at stricter IoU thresholds.
- Over-splitting improves recall but often reduces precision and AJI; under-splitting does the opposite. Practical deployment should pick a balance based on downstream use (counting vs morphology profiling).
- In this project, semantic metrics are strong while instance metrics are moderate, consistent with morphology-driven touching-cell complexity rather than foreground-detection failure.

## 5.1 Challenge: cell overlap / touching cells

Primary challenge in this assignment is separating overlapped/touching cells.

Current behavior from results:
- Semantic quality is strong (UNI2-h Scheme A Dice `0.9357`, IoU `0.8797`), meaning foreground detection is reliable.
- Instance metrics are improved but still moderate (Scheme A AP50 `0.6382`, mAP `0.4772`, AJI `0.4746`), indicating overlap splitting is better but not perfect.
- Precision and recall are close (`0.7166` vs `0.7360`), and predicted instance count is near GT (`n_pred≈48` vs `n_gt≈46.8`).
- Scheme B pushes AP slightly higher but increases predicted count (`n_pred≈52`) and lowers AJI/F1, consistent with mild over-splitting.

Interpretation:
- The dual-head + watershed strategy helps compared with semantic-only masks, but crowded/low-contrast regions still cause merge/split ambiguity.
- This is exactly where stronger instance-aware supervision and post-processing tuning are most impactful.

## 5.2 Report-ready results table (merged)

Use this compact table in your final submission:

| Run | Encoder | Postprocess | Dice | IoU | AP50 | AP75 | mAP(0.50:0.95) | Inst F1@0.5 | AJI |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| UNI2-h (Scheme A, best) | `uni2-h` | `t_fg=0.55`, `d_sigma=1.2`, `min_distance=13`, `peak_threshold=0.30`, `area_min=30` | **0.9357** | **0.8797** | **0.6382** | **0.5338** | **0.4772** | **0.7148** | **0.4746** |
| UNI2-h (Scheme B, trade-off) | `uni2-h` | `t_fg=0.55`, `d_sigma=1.0`, `min_distance=12`, `peak_threshold=0.32`, `area_min=25` | 0.9357 | 0.8797 | 0.6438 | 0.5391 | 0.4809 | 0.7087 | 0.4647 |
| Best open timm baseline | `vit_base_patch16_224` | `t_fg=0.5`, `d_sigma=1.0`, `min_distance=11`, `peak_threshold=0.25`, `area_min=20` | 0.8968 | 0.8144 | 0.5713 | 0.4196 | 0.3549 | 0.6752 | 0.3987 |

## 6) Reproducible notebook

Notebook deliverable:
- `github_uni/assignment_submission.ipynb`

It contains:
- reproducible commands for split/EDT/train/eval
- metrics loading and reporting cells
- qualitative visualization cells
- interpretation/reflection sections

## 6.1 Large artifacts (Google Drive)

Because full checkpoints and generated artifacts are large, the repository keeps code + reports, and stores large bundles externally.

- Model checkpoints + instance-eval result bundles (compressed):
  - [Google Drive link](https://drive.google.com/file/d/1wYRdsronZV3HBRFiOseTMxTAoewrdp0W/view?usp=drive_link)
- Original images + generated EDT files (compressed):
  - [Google Drive link](https://drive.google.com/file/d/1BL8yWM2eX913N4RWSmOLjmUSXS52JlrX/view?usp=drive_link)

Notes:
- The GitHub repo is the authoritative source for code, scripts, and report text.
- The Drive bundles provide reproducibility for heavyweight model/data artifacts.

## 7) Potential improvements

- Optionally unfreeze top UNI2-h encoder blocks if budget allows.
- Add stain normalization (Reinhard/Macenko) and compare against baseline normalization.
- Try a decoder normalization ablation (`BatchNorm` vs `GroupNorm`), which may improve small-batch stability; not tested due to time.
- Evaluate stronger semantic loss for slight class/decision imbalance in metrics (e.g., Focal loss), despite roughly balanced labels.
- Try more aggressive augmentations in controlled ablations (e.g., random resized crop, mild color jitter, blur) to test robustness gains.
- Tune watershed hyperparameters (`min_distance`, `peak_threshold`, `area_min`).
- Add stronger instance-aware supervision (boundary/center objectives) for higher mAP.
  - Example: add an auxiliary **center heatmap head** (Gaussian peaks at instance centroids) or **boundary loss head** to discourage merges in touching regions.
- Run larger hyperparameter search with Ray Tune once compute budget allows (especially LR, dist_weight, and scheduler), while keeping frozen-encoder lightweight training.
