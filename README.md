# Cell Instance Segmentation with UNI-style Backbone (BCCD)

This project adapts a pathology foundation-model style encoder for cell segmentation on BCCD, then uses distance-guided watershed post-processing to obtain **instance masks**.

## 1) Model adaptation and design choices

- **Backbone**:
  - Current reported results are based on `vit_base_patch16_224` (open timm backbone).
  - We have not received UNI access approval yet, so we use this closest open alternative for the current experiments.
  - The downstream training/evaluation pipeline is kept the same, but pretrained weights differ (general-purpose ImageNet-family pretraining vs UNI pathology pretraining).
  - We keep the same downstream segmentation/instance framework, so later UNI2-h swap is clean.
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
- We first established a strong baseline (`lr=2e-4`, `wd=1e-4`, `dist_weight=0.5`).
- Then we ran a controlled ablation (`lr=1e-4`, `wd=5e-5`) keeping all else fixed.
- Result: lower LR/WD did not improve this split, so baseline is currently preferred.
- Third planned run (`dist_weight=0.3`) targets better instance precision by reducing over-segmentation pressure from distance supervision.
- Tuning objective is explicit: maintain strong semantic Dice/IoU while improving AP/AJI (instance quality).
- In practice, tuning must balance the two heads:
  - overly large `dist_weight` can over-emphasize splitting and increase false positives/over-segmentation
  - overly small `dist_weight` can weaken touching-cell separation
  - we therefore monitor semantic (`Dice/IoU`) and instance (`AP/AJI/F1@0.5`) metrics jointly.

## 4) Quantitative results

### 4.1 Baseline semantic metrics from training checkpoint

From your run:

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

### 4.1.1 Additional experiment (lower LR + lower WD)

Second run (`lr=1e-4`, `wd=5e-5`, checkpoint in `github_uni/checkpoints_single2`):

- best validation:
  - loss: `0.3086`
  - macro F1: `0.9208`
  - foreground Dice/F1 (`F1_1`): `0.8906`
- best test-at-best-val:
  - loss: `0.3117`
  - macro F1: `0.9209`
  - foreground Dice/F1 (`F1_1`): `0.8918`

Comparison vs baseline:
- `F1_1(test)`: `0.8958 -> 0.8918` (baseline better by `+0.0040`)
- interpretation: lower LR/WD made optimization slightly more conservative, but did not improve final segmentation quality on this split.

### 4.1.2 Third experiment (re-balance for instance quality)

Third run (`lr=1e-4`, `wd=1e-4`, `dist_weight=0.3`, checkpoint in `github_uni/checkpoints_single3`):

- best validation:
  - loss: `0.2980`
  - macro F1: `0.9229`
  - foreground Dice/F1 (`F1_1`): `0.8933`
- best test-at-best-val:
  - loss: `0.3008`
  - macro F1: `0.9231`
  - foreground Dice/F1 (`F1_1`): `0.8945`

Comparison:
- better than Exp-2 (`0.8945` vs `0.8918` on `F1_1(test)`)
- slightly below baseline (`0.8945` vs `0.8958`)

### 4.2 Instance metrics (IoU, Dice, mAP)

Run:

```bash
python3 github_uni/eval/evaluate_instance.py \
  --checkpoint github_uni/checkpoints_single/best_model.pt \
  --split test \
  --save_dir github_uni/checkpoints_single/instance_eval_test
```

This writes:
- `github_uni/checkpoints_single/instance_eval_test/metrics_summary.json`
- `pred_semantic/*.png`
- `pred_instance/*.png`
- `pred_instance_color/*.png`
- `overlays/*.png`
- `panels_input_label_overlay_instance/*.png` (2x2 panel for each image: input, label, overlay, instance-color)
- qualitative selections in:
  - `qualitative/success`
  - `qualitative/failure`

If you want to evaluate the second checkpoint too:

```bash
python3 github_uni/eval/evaluate_instance.py \
  --checkpoint github_uni/checkpoints_single2/best_model.pt \
  --split test \
  --save_dir github_uni/checkpoints_single2/instance_eval_test
```

Observed second-run instance results (`checkpoints_single2/instance_eval_test/metrics_summary.json`):
- Dice: `0.8926`
- IoU: `0.8075`
- AP50: `0.5596`
- mAP(0.50:0.95): `0.3378`
- Instance F1@IoU0.5: `0.6752`
- AJI: `0.3935`

Observed third-run instance results (`checkpoints_single3/instance_eval_test/metrics_summary.json`):
- Dice: `0.8953`
- IoU: `0.8119`
- AP50: `0.5643`
- mAP(0.50:0.95): `0.3464`
- Instance F1@IoU0.5: `0.6753`
- AJI: `0.3965`

Compared with baseline (`checkpoints_single`), baseline remains better on semantic quality and detection mAP:
- Dice: `0.8968` vs `0.8926`
- IoU: `0.8144` vs `0.8075`
- AP50: `0.5713` vs `0.5596`
- mAP(0.50:0.95): `0.3549` vs `0.3378`
- AJI: `0.3987` vs `0.3935`

Compared with Exp-2, Exp-3 is consistently stronger:
- Dice: `0.8953` vs `0.8926`
- IoU: `0.8119` vs `0.8075`
- AP50: `0.5643` vs `0.5596`
- mAP(0.50:0.95): `0.3464` vs `0.3378`
- AJI: `0.3965` vs `0.3935`

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

## 5.2 Challenge: cell overlap / touching cells

Primary challenge in this assignment is separating overlapped/touching cells.

Current behavior from results:
- Semantic quality is strong (baseline Dice `0.8968`, IoU `0.8144`), meaning foreground detection is reliable.
- Instance metrics are moderate (baseline AP50 `0.5713`, mAP `0.3549`, AJI `0.3987`), indicating overlap splitting is only partially solved.
- Precision is lower than recall (`0.6366` vs `0.7436`) with slightly high predicted instance count, consistent with occasional over-splitting.

Interpretation:
- The dual-head + watershed strategy helps compared with semantic-only masks, but crowded/low-contrast regions still cause merge/split ambiguity.
- This is exactly where stronger instance-aware supervision and post-processing tuning are most impactful.

## 5.1 Report-ready experiment table

Use this compact table in your final submission:

| Run | lr | wd | dist_weight | Dice/F1_1 (test) | Macro F1 (test) | Notes |
|---|---:|---:|---:|---:|---:|---|
| Baseline (`checkpoints_single`) | 2e-4 | 1e-4 | 0.5 | **0.8958** | **0.9240** | Best semantic baseline |
| Exp-2 (`checkpoints_single2`) | 1e-4 | 5e-5 | 0.5 | 0.8918 | 0.9209 | Slight degradation; AP50=0.5596, mAP=0.3378, AJI=0.3935 |
| Exp-3 (`checkpoints_single3`) | 1e-4 | 1e-4 | 0.3 | 0.8945 | 0.9231 | Better than Exp-2; AP50=0.5643, mAP=0.3464, AJI=0.3965 |

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

- Model checkpoints + instance-eval result bundles (`checkpoints_single`, `checkpoints_single2`, `checkpoints_single3`, compressed):
  - [Google Drive link](https://drive.google.com/file/d/1k2pnDV9Q2iYVAAjH9nfB1HPdU1ZRUFCZ/view?usp=drive_link)
- Original images + generated EDT files (compressed):
  - [Google Drive link](https://drive.google.com/file/d/1BL8yWM2eX913N4RWSmOLjmUSXS52JlrX/view?usp=drive_link)

Notes:
- The GitHub repo is the authoritative source for code, scripts, and report text.
- The Drive bundles provide reproducibility for heavyweight model/data artifacts.

## 7) Potential improvements

- Switch backbone to `uni2-h` once gated access is approved.
- Add stain normalization (Reinhard/Macenko) and compare against baseline normalization.
- Try a decoder normalization ablation (`BatchNorm` vs `GroupNorm`), which may improve small-batch stability; not tested due to time.
- Evaluate stronger semantic loss for slight class/decision imbalance in metrics (e.g., Focal loss), despite roughly balanced labels.
- Try more aggressive augmentations in controlled ablations (e.g., random resized crop, mild color jitter, blur) to test robustness gains.
- Tune watershed hyperparameters (`min_distance`, `peak_threshold`, `area_min`).
- Add stronger instance-aware supervision (boundary/center objectives) for higher mAP.
  - Example: add an auxiliary **center heatmap head** (Gaussian peaks at instance centroids) or **boundary loss head** to discourage merges in touching regions.
- Run larger hyperparameter search with Ray Tune once compute budget allows (especially LR, dist_weight, and scheduler), while keeping frozen-encoder lightweight training.
