# Sub-Saharan wildlife object detection

A TensorFlow Object Detection API experiment comparing SSD ResNet50 FPN and Faster R-CNN ResNet101 on a three-species wildlife dataset. The repository contains notebooks and saved evaluation logs.

## Recorded evaluation

| COCO metric | SSD | Faster R-CNN |
|---|---:|---:|
| AP at IoU 0.50–0.95 | 0.839 | 0.784 |
| AP at IoU 0.50 | 0.910 | 0.912 |
| AP at IoU 0.75 | 0.868 | 0.849 |

These are historical values captured in [Evaluation.ipynb](Evaluation.ipynb). SSD led on the aggregate IoU range and Faster R-CNN led narrowly at IoU 0.50. The original camera-trap images and model weights are absent from this repository. The historical run uses machine-specific paths and cannot be rerun from the checkout alone. Generalisation to unseen camera sites remains unmeasured.

## Data and evaluation workflow

- [Pre-processing](Pre-Processing.ipynb) covers annotation counts, augmentation and TFRecord preparation.
- [Training](Training.ipynb) records detector configuration and training iterations.
- [Evaluation](Evaluation.ipynb) preserves COCO-style metrics and original logs.
- [Inference](inference.ipynb) records exploratory predictions on separate images.

For a fresh experiment, assign acquisition group IDs before creating augmented copies, keep camera sites or collection sessions together, and generate an image manifest with `path,split,group`. Paths are relative to the manifest. The audit exits with code 1 for repeated file contents or an acquisition group found across splits.

```bash
python audit_split.py image_manifest.csv
```

The audit checks byte-identical images and the supplied group IDs. Near-duplicate images require a separate similarity check.

The [companion serving demo](https://github.com/amirosseinz/object-detection-comparison) uses a separate label map. The pre-processing notebook lists eland as the third species and the serving demo maps class 3 to lion. Treat the historical training AP and serving timings as separate experiments.
