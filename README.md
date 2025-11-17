# Image-based Cattle Disease Detection System

## 📋 Overview

This repository implements a **production-ready demonstration pipeline** for detecting cattle diseases from images using deep learning and computer vision. The system combines a **Keras/TensorFlow classification model** with a **Flask REST API** and a **modern web interface** to enable real-time disease detection and analysis.

**Supported disease classes:**
- Lumpy Skin Disease (LSD)
- Foot-and-Mouth Disease (FMD)
- Mastitis
- Healthy (no disease)

**Note:** This is a research/prototype system intended for research and educational purposes, not clinical diagnosis.

---

## 🎯 Key Features

- ✅ **Transfer Learning**: Pre-trained backbone (MobileNetV2/EfficientNetB0) fine-tuned for cattle disease classification
- ✅ **Smart Image Vetting**: ImageNet-based heuristics to reject non-cow images and reduce false positives
- ✅ **REST API**: Flask server with `/predict` endpoint for image classification
- ✅ **Web UI**: Drag-and-drop interface with real-time preview and results visualization
- ✅ **Dark Mode**: Toggle between light and dark themes (persisted in localStorage)
- ✅ **Confidence Scores**: Per-class probability distributions for interpretability
- ✅ **Deployment Ready**: Guidelines for production deployment with WSGI + NGINX

---

## 📁 Project Structure

```
Cattle_Disease_Project_modified/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── class_indices.json                 # Class label mappings (0->healthy, etc.)
├── cattle_disease_model_simple.h5     # Trained Keras model weights
│
├── app_modified.py                    # Flask server (main entry point)
├── app_robust.py                      # Alternate server implementation
│
├── templates/
│   └── index.html                     # Web UI template
├── static/
│   └── css/
│       └── style.css                  # UI styling + dark mode
│
└── dataset/                           # Training images (organized by class)
    ├── healthy/
    ├── lumpy_skin/
    ├── mastitis/
    └── foot_mouth/
```

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- **Python**: 3.8, 3.9, or 3.10
- **pip**: Latest version
- **Virtual Environment**: Recommended (venv or Conda)

### Installation

**1. Clone or download the repository:**
```powershell
cd C:\Users\hp\Desktop\Cattle_Disease_Project_modified
```

**2. Create and activate a virtual environment:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**3. Install dependencies:**
```powershell
pip install -r requirements.txt
```

**4. Run the Flask server:**
```powershell
python app_modified.py
```

You should see output like:
```
* Running on http://127.0.0.1:5000
* WARNING: This is a development server. Do not use it in production.
```

**5. Open the web UI:**
Navigate to: **http://127.0.0.1:5000** in your web browser

---

## 📊 Model Architecture & Training

### Model Backbone

The system uses **transfer learning** on a pre-trained ImageNet backbone for efficient training and good accuracy:

| Component | Details |
|-----------|---------|
| **Base Model** | MobileNetV2 or EfficientNetB0 (ImageNet pretrained) |
| **Input Size** | 224 × 224 × 3 (RGB images) |
| **Preprocessing** | Backend-specific (e.g., `mobilenet_v2.preprocess_input`: [-1, 1] range) |
| **Pooling** | Global Average Pooling |
| **Classification Head** | Dropout(0.3) → Dense(4, softmax) |
| **Classes** | 4 (healthy, lumpy_skin, mastitis, foot_mouth) |

### Training Pipeline

**Loss Function:** Categorical Cross-Entropy  
**Optimizer:** Adam (learning rate: 1e-4)  
**Batch Size:** 32  
**Epochs:** 30 (with early stopping)  

**Callbacks Used:**
- `ModelCheckpoint`: Save best weights (monitor: val_loss)
- `ReduceLROnPlateau`: Reduce LR if validation loss plateaus
- `EarlyStopping`: Stop if no improvement for 5-8 epochs

**Data Augmentation (training only):**
- Random horizontal flip (50% probability)
- Random rotation (±20°)
- Random brightness/contrast jitter
- Random zoom and crop
- Small translations

**Train/Val/Test Split:** 80% / 10% / 10%

### Training Command Example

```powershell
python train_simple_robust.py `
  --data_dir dataset `
  --backbone mobilenetv2 `
  --img_size 224 `
  --batch_size 32 `
  --epochs 30 `
  --lr 1e-4
```

### Evaluation Metrics

Track these during training and validation:
- **Accuracy**: Overall classification accuracy
- **Per-class Precision/Recall/F1**: Identifies class-specific performance
- **Confusion Matrix**: Reveals misclassification patterns
- **ROC/AUC**: Especially useful for imbalanced datasets

---

## 🔬 Preprocessing & Inference

### Preprocessing Steps (Training & Inference)

1. **Load Image**: PIL or OpenCV
2. **Resize**: Bilinear interpolation to 224×224
3. **Normalize**: Convert to float32 and apply backbone-specific preprocessing
4. **Batch**: Expand dims to (1, 224, 224, 3)

### Python Inference Example

```python
from PIL import Image
import numpy as np
import tensorflow as tf
import json

# Load model once (not per prediction in production)
model = tf.keras.models.load_model('cattle_disease_model_simple.h5')

# Load class mapping
with open('class_indices.json') as f:
    class_indices = json.load(f)
    idx_to_label = {v: k for k, v in class_indices.items()}

# Preprocess image
img = Image.open('path/to/cow_image.jpg').convert('RGB')
img = img.resize((224, 224))
img_array = np.array(img, dtype='float32')

# Apply backbone-specific preprocessing
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
img_batch = np.expand_dims(img_array, 0)

# Predict
predictions = model.predict(img_batch)[0]  # Shape: (4,)
pred_idx = int(np.argmax(predictions))
pred_label = idx_to_label[pred_idx]
confidence = float(predictions[pred_idx])

print(f"Prediction: {pred_label} (confidence: {confidence:.2%})")
```

---

## 🛡️ Inference Pipeline & Vetting Heuristics

The server implements a **layered vetting strategy** to reduce false positives on non-cow images (documents, screenshots, etc.):

### Decision Logic (as implemented in `app_modified.py`)

```
1. Run disease model → get softmax probabilities
                ↓
2. Run ImageNet MobileNetV2 (lazy-loaded) on same image
                ↓
3. Decode top-5 semantic labels from ImageNet
                ↓
4. Apply decision rules:
   
   IF ImageNet labels indicate non-natural/document content
      AND prob > NON_NATURAL_PROB_THRESHOLD (default: 0.15)
      THEN: Reject image (HTTP 400)
   
   ELSE IF ImageNet shows bovine-related keywords
      (ox, cow, bull, bison) AND prob > BOVINE_PROB_THRESHOLD (default: 0.05)
      THEN: Accept image
   
   ELSE IF disease model top probability > DISEASE_CONF_THRESHOLD (default: 0.35)
      THEN: Accept image
   
   ELSE:
      Reject or flag image
```

### Configuration Parameters (in `app_modified.py`)

```python
DISEASE_CONF_THRESHOLD = 0.35          # Disease model confidence cutoff
BOVINE_PROB_THRESHOLD = 0.05           # ImageNet bovine label threshold
NON_NATURAL_PROB_THRESHOLD = 0.15      # ImageNet non-natural label threshold
```

**Tuning Guide:**
- Too many false positives on non-cows? → **Increase** `NON_NATURAL_PROB_THRESHOLD`
- Legitimate close-ups rejected? → **Lower** `NON_NATURAL_PROB_THRESHOLD` or **raise** `BOVINE_PROB_THRESHOLD`
- Need stricter vetting? → **Lower** `DISEASE_CONF_THRESHOLD`

---

## 📥 Dataset Organization

Organize your training images in a class-based directory structure:

```
dataset/
├── healthy/
│   ├── cow_001.jpg
│   ├── cow_002.jpg
│   └── ...
├── lumpy_skin/
│   ├── disease_001.jpg
│   └── ...
├── mastitis/
│   ├── udder_001.jpg
│   └── ...
└── foot_mouth/
    ├── mouth_001.jpg
    └── ...
```

**Best Practices:**
- **Consistency**: Use descriptive filenames
- **Diversity**: Include varied lighting, angles, and camera types
- **Balance**: Aim for ~equal samples per class (or use class weights if imbalanced)
- **Privacy**: Keep sensitive images out of public repositories
- **Metadata**: Store source information in a separate CSV file

---

## 🔧 REST API Reference

### Endpoints

#### `GET /` — Web UI
Serves the HTML interface at `http://127.0.0.1:5000/`

#### `POST /predict` — Image Classification
**URL:** `http://127.0.0.1:5000/predict`  
**Content-Type:** `multipart/form-data`  
**Parameter:** `file` (image file)

### Request Examples

**Using curl (PowerShell):**
```powershell
curl -X POST -F "file=@C:\path\to\image.jpg" http://127.0.0.1:5000/predict
```

**Using Python (requests library):**
```python
import requests

files = {'file': open('image.jpg', 'rb')}
response = requests.post('http://127.0.0.1:5000/predict', files=files)
result = response.json()
print(result)
```

### Response Format

**Success (HTTP 200):**
```json
{
  "status": "success",
  "predictions": [
    {"label": "mastitis", "probability": 0.67},
    {"label": "healthy", "probability": 0.20},
    {"label": "lumpy_skin", "probability": 0.10},
    {"label": "foot_mouth", "probability": 0.03}
  ],
  "accepted": true
}
```

**Rejected (HTTP 400):**
```json
{
  "status": "rejected",
  "message": "Uploaded image appears to be a document (envelope). Please upload a clear cow image.",
  "accepted": false
}
```

---

## 🌐 Web UI Features

### Upload & Preview
- **Drag & drop** images or click to browse
- **Real-time preview** before analysis
- Supported formats: JPG, PNG, JPEG

### Image Upload Guidelines
The UI displays recommended capture techniques:
- **Lumpy Skin**: Full-body side view showing skin texture and any nodules
- **Mastitis**: Close-up of udder/teats from side or below, well-lit
- **FMD**: Close-up of mouth interior, tongue, or foot lesions
- **Healthy**: Clear reference images of normal cattle skin/udder

### Results Display
- Confidence scores with progress bars
- Per-class probability percentages
- Clear indication of top prediction
- Confidence-based color coding (green=high, yellow=medium, red=low)

### Dark Mode
- Toggle available in header
- Preference persisted in browser localStorage
- Automatic light/dark theme switching

---

## 💻 Deployment (Production)

### Development vs. Production

**Development (Current):**
- Flask development server
- Single-threaded
- Auto-reloads on code changes
- Suitable for testing and prototyping

**Production:**
- WSGI server (Gunicorn, Waitress)
- Reverse proxy (NGINX)
- Load balancing
- SSL/TLS encryption

### Recommended Production Setup

```
┌─────────────┐         ┌──────────────┐        ┌──────────────┐
│   Client    │◄───────►│  NGINX       │◄──────►│  Gunicorn    │
│  (Browser)  │         │  (Reverse    │        │  + Flask +   │
└─────────────┘         │   Proxy)     │        │  Keras Model │
                        └──────────────┘        └──────────────┘
                             ▲
                    TLS/SSL Termination
                    Static Asset Caching
                    Rate Limiting
```

### Deployment Steps

**1. Install production dependencies:**
```powershell
pip install gunicorn  # or waitress on Windows
```

**2. Run with Gunicorn:**
```powershell
gunicorn --workers 4 --bind 0.0.0.0:5000 app_modified:app
```

**3. Configure NGINX:**
```nginx
upstream flask_app {
    server localhost:5000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://flask_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /static/ {
        alias /path/to/static/;
        expires 1d;
        add_header Cache-Control "public, immutable";
    }
}
```

### Model Serving Options

For high-throughput production scenarios, consider:
- **TensorFlow Serving**: Containerized model server with gRPC API
- **TorchServe**: Similar for PyTorch models
- **NVIDIA Triton Inference Server**: Multi-GPU, multi-model batching

---

## 🔍 Troubleshooting & Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **Model file not found** | Missing `cattle_disease_model_simple.h5` | Ensure model file exists in repo root or update path in `app_modified.py` |
| **TensorFlow import error** | Version incompatibility | Verify Python 3.8-3.10 and reinstall TensorFlow: `pip install --upgrade tensorflow` |
| **Non-cow images classified as disease** | Low `NON_NATURAL_PROB_THRESHOLD` | Increase threshold or expand non-natural keywords list in `app_modified.py` |
| **False rejections of close-ups** | Too-high `NON_NATURAL_PROB_THRESHOLD` | Lower threshold and re-test with problem images |
| **Slow inference** | Large model or insufficient memory | Consider lighter backbone (MobileNetV2 vs EfficientNetB0) |
| **Port 5000 already in use** | Another process using port | Change port: `python app_modified.py --port 5001` |

---

## 🔐 Security & Privacy Considerations

- **Image Handling**: Uploaded images are processed in-memory and **not persisted** by default
- **To enable persistence** (for audit/debugging): Modify `app_modified.py` to save images to an `uploads/` folder and add it to `.gitignore`
- **Production**: Implement authentication, rate limiting, and audit logging
- **Data Retention**: Define and enforce data deletion policies
- **User Consent**: Ensure users consent to image processing before upload

---

## 📚 Reproducibility & Environment

### Python Version
```powershell
python --version
# Should output: Python 3.8.x, 3.9.x, or 3.10.x
```

### Virtual Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Freeze Dependencies
After installing and testing:
```powershell
pip freeze > requirements_exact.txt
```

### Reproducible Training
```powershell
# Set random seeds for reproducibility
python train_simple_robust.py --seed 42 --data_dir dataset
```

---

## 🎓 Model Tuning & Hyperparameter Guide

### When Accuracy is Low

1. **Increase augmentation intensity**: Rotation, zoom, brightness
2. **Gather more training data**: Especially for underperforming classes
3. **Try stronger backbone**: EfficientNetB4/B5 instead of MobileNetV2
4. **Increase training epochs**: Use early stopping to avoid overfitting
5. **Class weight balancing**: Handle imbalanced datasets

### When Overfitting (train accuracy >> val accuracy)

1. **Increase dropout rate**: 0.3 → 0.5
2. **Add L2 regularization**: kernel_regularizer=l2(0.001)
3. **Increase augmentation**: More aggressive transformations
4. **Reduce model complexity**: Use lighter backbone
5. **Data augmentation**: Effective regularization strategy

### When Inference is Slow

1. **Use MobileNetV2**: Faster than EfficientNet variants
2. **Quantize model**: `tf.lite.TFLiteConverter` for ~4x speedup
3. **Batch predictions**: Process multiple images at once
4. **GPU acceleration**: Enable CUDA if available

---

## 📄 License & Contributing

### License
Add a `LICENSE` file when you decide on a license. **MIT** is recommended for permissive reuse:
```
Permission is hereby granted, free of charge, to any person obtaining a copy...
```

### Contributing Guidelines
- **Report Bugs**: Open GitHub issues with reproducible steps
- **Propose Features**: Discuss enhancements before implementing
- **Submit PRs**: Keep datasets out of version control; use external hosting
- **Documentation**: Update README for any new features

### Authors & Acknowledgments
- Primary Repository: https://github.com/SahilSinha007/Image-based-cow-disease-detection
- Built with: TensorFlow, Flask, Bootstrap

---

## 🖼️ Screenshots

Add sample screenshots to demonstrate the web UI:

**Upload Interface:**
![Upload](screenshots/screenshot-upload.png)

**Prediction Results:**
![Results](screenshots/screenshot-result.png)

*To add screenshots: create a `screenshots/` folder, add images, and update the paths above.*

---

## 📞 Support & Contact

For questions, issues, or feedback:
- **GitHub Issues**: Open an issue on the repository
- **Email**: Contact the repository maintainer
- **Discussion**: Use GitHub Discussions for feature ideas

---

## 🎯 Next Steps & Roadmap

**Short-term improvements:**
- [ ] Add rejection reason details to JSON response
- [ ] Implement "Proceed anyway" UI override
- [ ] Create training data collection guidelines

**Medium-term:**
- [ ] Train dedicated binary cow/not-cow classifier
- [ ] Add model quantization for edge deployment
- [ ] Implement automated testing and CI/CD

**Long-term:**
- [ ] Multi-model ensemble for improved accuracy
- [ ] Localization/segmentation of disease regions
- [ ] Mobile app (iOS/Android)
- [ ] Real-time video stream processing

---

**Last Updated:** November 17, 2025  
**Status:** Production-ready demo — use for research and prototyping
