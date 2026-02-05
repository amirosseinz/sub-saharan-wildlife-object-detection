# Sub-Saharan Wildlife Object Detection

> Deep learning-based wildlife species detection using SSD and Faster R-CNN architectures for camera trap imagery analysis.

**Keywords**: Computer Vision, Object Detection, Transfer Learning, TensorFlow, Camera Trap, Wildlife Conservation, SSD, Faster R-CNN

---

## Problem Statement

Automated wildlife monitoring through camera trap imagery is critical for conservation efforts but requires robust detection systems that handle challenging conditions: variable lighting, occluded subjects, and multiple animals per frame. This project implements and compares two state-of-the-art object detection architectures to identify three Sub-Saharan species from camera trap images.

---

## Approach

### Models Implemented
| Architecture | Backbone | Input Size | Pre-training |
|-------------|----------|------------|--------------|
| SSD ResNet50 V1 FPN | ResNet50 + FPN | 640×640 | COCO 2017 |
| Faster R-CNN | ResNet101 | 640×640 | COCO 2017 |

### Key Techniques
- **Transfer learning** from COCO pre-trained weights → faster convergence on small wildlife dataset
- **Data augmentation**: Horizontal flip, rotation, brightness adjustment, grayscale conversion
- **Multi-scale detection** via Feature Pyramid Network (FPN) in SSD architecture
- **COCO-style evaluation**: mAP at multiple IoU thresholds (0.50, 0.75, 0.50:0.95)

### Dataset
- **Species**: *Struthio camelus* (Ostrich), *Phacochoerus africanus* (Warthog), *Tragelaphus oryx* (Eland)
- **Annotation format**: Pascal VOC XML (bounding boxes)
- **Split**: 90% training / 10% test

---

## Results

| Metric | SSD ResNet50 FPN | Faster R-CNN ResNet101 |
|--------|------------------|------------------------|
| **AP @ IoU=0.50:0.95** | **0.839** | 0.784 |
| **AP @ IoU=0.50** | 0.910 | 0.912 |
| **AP @ IoU=0.75** | **0.868** | 0.849 |
| **AR (maxDets=10)** | **0.860** | 0.820 |
| Final Training Loss | 0.252 | 0.076 |
| Training Time | ~6 hours | ~14 hours |

**Key finding**: SSD achieves higher mAP across IoU thresholds with 2.3× faster training, making it the preferred choice for deployment scenarios requiring faster inference.

---

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **640×640 input resolution** | Balanced GPU memory usage against detection accuracy; pre-trained weights available. Trade-off: higher resolution (1024×1024) would improve small object detection. |
| **Batch size = 4 for Faster R-CNN** | Memory constraint with ResNet101 backbone (~16-20GB VRAM required). SSD trained at batch_size=16. |
| **Class balance by annotation count** | Object detection requires counting bounding boxes, not image files—a single image may contain multiple animals. |
| **No pre-emptive resizing** | Input dimensions handled by model pipeline.config, preserving flexibility for hyperparameter tuning. |

---

## What This Demonstrates

> End-to-end object detection pipeline: data exploration with correct class balance analysis, preprocessing with augmentation, comparative model training (single-stage vs. two-stage detectors), hyperparameter tuning across multiple training phases, and COCO-style evaluation with IoU-based metrics.

**Technical competencies**:
- TensorFlow Object Detection API workflow
- Transfer learning for domain-specific detection
- GPU memory optimization (batch size tuning)
- Quantitative model comparison using mAP/IoU metrics

---

## Reproducibility

### Requirements
```bash
pip install -r requirements.txt
```

### Project Structure
```
├── Pre-Processing.ipynb    # EDA, augmentation, TFRecord generation
├── Training.ipynb          # Model training with hyperparameter tuning
├── Evaluation.ipynb        # COCO-style metrics and comparison
├── inference.ipynb         # Detection on new images
├── tensorboard & freeze.ipynb  # Model export and visualization
├── models/                 # Exported SavedModel files (included)
│   ├── ssd_resnet50_v1_fpn_640x640/
│   └── faster_rcnn_resnet101_v1_640x640/
└── requirements.txt
```

### What's Included
- ✅ **Jupyter notebooks**: Complete pipeline from EDA to evaluation
- ✅ **Trained models**: Exported SavedModel files ready for inference
- ❌ **Dataset**: Not included (proprietary camera trap imagery)

### Running Inference
With the provided saved models, you can run inference on your own images:
```python
import tensorflow as tf
model = tf.saved_model.load("models/ssd_resnet50_v1_fpn_640x640/saved_model")
```

---

## 🐳 Production Deployment

A containerized deployment of these trained models is available as a separate MLOps project:

### [Dockerized Wildlife Detection API](https://github.com/amirosseinz/object-detection-comparison)

**Features**:
- REST API for real-time inference
- Docker containerization for reproducible deployment
- Support for both SSD and Faster R-CNN models
- Scalable architecture for production workloads

```bash
# Quick start
docker pull ahakrami98/object-detection-comparison:latest
docker run -p 80:80 object-detection-comparison:latest
```

> *Link to deployment repository: [https://github.com/amirosseinz/object-detection-comparison]*

---

## Limitations & Future Work

1. **Resolution trade-off**: 640×640 may undersample camera trap images (~1920×1080 native). Higher resolution training recommended for production.
2. **Class imbalance**: Annotation distribution skewed; weighted loss or focal loss could improve minority class detection.
3. **Dataset availability**: Training data not publicly available due to proprietary restrictions.

---

## Related Projects

| Project | Description |
|---------|-------------|
| **This Repository** | Model development, training, and evaluation notebooks |
| **[Wildlife Detection MLOps](link-to-your-mlops-repo)** | Dockerized REST API deployment with CI/CD pipeline |

---

## License

Academic project - LJMU MSc Computer Vision coursework.
