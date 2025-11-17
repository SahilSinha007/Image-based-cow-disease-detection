# AI-Powered Cattle Disease Detection
git add .
git commit -m "Initial project commit"
gh repo create <repo-name> --public --source=. --remote=origin --push
This repository contains an easy-to-run Flask web application and supporting files used to detect cattle diseases from images using a pre-trained Keras model. It includes:

- A web UI: `templates/index.html` + `static/` (CSS, icons, JS) for image upload and results.
- Server code: `app_modified.py` (server used during development) and `app_robust.py` (untouched/alternate).
- Model mapping: `class_indices.json` (maps labels) and a recommended model file `cattle_disease_model_simple.h5` (excluded from the repo by default).

This README documents how to run the app locally, the API, the cow-detection safeguards, configuration and deployment tips, and troubleshooting.

Contents
- **Prerequisites**
- **Quick start (local)**
- **Project structure**
- **API endpoints & examples**
- **Cow-detection and false-positive mitigation**
- **Deployment & GitHub**
- **Development notes, training, and future improvements**
- **Troubleshooting**
- **License & Contributing**

Prerequisites
-------------
- Python 3.8+ (3.9/3.10 recommended)
- pip
- (Optional) Git and GitHub CLI (`gh`) if you want to create a remote repository programmatically
- Internet connection for the first run if using MobileNetV2 fallback (it will download ImageNet weights)

Quick start (local)
-------------------
PowerShell commands (run inside `C:\Users\hp\Desktop\Cattle_Disease_Project_modified`):

1) Create & activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

3) Run the app (development):

```powershell
python .\app_modified.py
```

4) Open the UI in your browser: http://127.0.0.1:5000

Project structure
-----------------
Key files and directories:

- `app_modified.py` — main Flask server with added cow-detection safeguards (this file is the one used for local testing).
- `app_robust.py` — alternate app file (left unchanged per project history).
- `templates/index.html` — HTML UI (image upload, progress, results, guidelines, dark mode toggle).
- `static/css/style.css` — CSS and dark mode styles.
- `cattle_disease_model_simple.h5` — expected trained model (not committed by default due to size).
- `class_indices.json` — mapping between class names and model indices.
- `uploads/` — runtime folder where uploaded images are stored (ignored by git).
- `.gitignore` — ignores virtualenvs, model files, uploads, logs, etc.

API endpoints & examples
------------------------
The Flask app exposes the following useful endpoints:

- `GET /` — returns the web UI (browser)
- `POST /predict` — accepts multipart file upload, returns JSON prediction and metadata
- `GET /health` — health/status JSON for quick checks
- `GET /model-stats` — detailed model metrics JSON

Example: Predict (PowerShell curl)

```powershell
$file = 'C:\path\to\your\image.jpg'
curl -X POST -F "file=@$file" http://127.0.0.1:5000/predict
```

Expected JSON (success):

```json
{
  "success": true,
  "prediction": "Mastitis",
  "confidence": 66.9,
  "class": "mastitis",
  "description": "Mastitis detected - inflammation of the mammary gland.",
  "all_predictions": {"mastitis": 66.9, "lumpy_skin": 10.2, "healthy": 20.0, "foot_mouth": 2.9},
  "image_info": {"original_size": [1280, 960], "processed_shape": [1,224,224,3]},
  "timestamp": "2025-11-17T..."
}
```

If the server rejects the image (e.g., non-cow document or a screenshot), the response will be a 400 JSON with an explanatory message, for example:

```json
{ "error": "Uploaded image does not appear to contain a cow or relevant cow body part. Please upload a clear image of a cow (full body or close-up of mouth, hoof, udder).", "success": false }
```

Cow-detection and false-positive mitigation (detailed)
-----------------------------------------------------
This project includes defensive logic to reduce incorrect disease predictions for images that are not cows (documents, screenshots, game images, etc.). The checks are implemented in `app_modified.py` in function `is_likely_cow(...)` and follow this strategy:

1. ImageNet (MobileNetV2) fallback (lazy-loaded): when available the app runs the uploaded image through a MobileNetV2 pretrained on ImageNet and decodes the top labels.
   - If ImageNet suggests bovine-related labels (e.g., `cow`, `ox`, `bull`, `bison`) with modest probability (default 5%), the image is accepted as containing a cow.
   - If ImageNet suggests document/screen/game/statue/man-made labels with higher probability (default 15%) or the top label is a non-natural object with probability >= 10%, the image is rejected.

2. Disease-model fallback: if ImageNet is unavailable or inconclusive, the disease model's confidence is used as a fallback. A disease prediction (mastitis, lumpy_skin, foot_mouth) equal to or above a configured threshold (default 35%) will be accepted for close-up images.

3. Tunable thresholds and behavior:
   - `DISEASE_CONF_THRESHOLD` (default 0.35) — disease model confidence cutoff used as a fallback.
   - `bovine_prob_threshold` (default 0.05) — ImageNet probability to accept bovine label.
   - `non_natural_prob_threshold` (default 0.15) — ImageNet probability to reject non-natural labels.

Why this exists: the disease model is trained to detect visual patterns on cow skin, udder, hooves and mouth. However, on completely unrelated images (papers, screenshots, game scenes), the model can produce confident but spurious outputs. The ImageNet-based vetting reduces such false positives by rejecting obvious non-animal images.

Notes and limitations
---------------------
- This mechanism is a heuristic: ImageNet labels can be noisy and may not always correctly indicate 'cow' vs 'not cow'.
- Offline or air-gapped environments without ImageNet weights will fall back to disease-model confidence behavior; that is less reliable for non-cow images.
- The long-term robust solution is to train a small binary classifier (cow vs not-cow) using transfer learning — see "Future improvements" below.

Configuration and environment variables
--------------------------------------
The code does not currently require environment variables, but you may prefer to expose some config via env vars in production, for example:

- `APP_MODEL_FILE` — path to the model file (default `cattle_disease_model_simple.h5`)
- `DISABLE_IMAGENET_CHECK` — set to `1` to disable ImageNet vetting (useful in offline contexts)

You can modify `app_modified.py` to read these from `os.environ` before starting the model.

Development notes (how the UI works)
-----------------------------------
- The UI in `templates/index.html` supports drag/drop and file selection.
- After upload, the UI shows a preview and an "Analyze Image" button. Results are displayed with confidence bars and descriptive text.
- I added an "Image upload guidelines" section that instructs how to take photos for each disease:
  - Lumpy Skin Disease: full-body side view to show skin lumps
  - Foot and Mouth Disease: close-up of mouth and lower legs/hooves
  - Mastitis: close-up of the udder from the side/below
- A dark-mode toggle stores the preference in `localStorage`.

Security and privacy
--------------------
- Uploaded images are stored in `uploads/` folder by default. That folder is ignored in git. Consider cleaning it periodically.
- Do not commit images, model files, or any private data to the repository. Use cloud storage or GitHub Releases for large models.

How to publish to GitHub
------------------------
1) Create a new repository on GitHub (via website or `gh` CLI).

2) From your project folder:

```powershell
git init
git add .
git commit -m "Initial commit - cattle disease detection"
git remote add origin https://github.com/<your-username>/<repo>.git
git branch -M main
git push -u origin main
```

If you have the GitHub CLI installed you can create and push in one step:

```powershell
gh repo create <repo-name> --public --source=. --remote=origin --push
```

CI/CD (optional)
-----------------
If you want a continuous integration workflow, add a GitHub Actions YAML file to `.github/workflows/` to run linting/tests. A simple workflow might run `pip install -r requirements.txt` and run a lint or unit test script.

Future improvements and advanced ideas
-------------------------------------
- Train a binary "cow vs non-cow" classifier with transfer learning (MobileNetV2/EfficientNet) to remove reliance on ImageNet heuristics.
- Add an option in the UI to "Proceed anyway" when an image is rejected by the vetting logic (useful for advanced users).
- Move model files to cloud storage and download them on first-run, or use Git LFS for versioning.
- Add end-to-end tests, and a GitHub Action to run them on PRs.

Troubleshooting
---------------
- Q: The server fails to load MobileNetV2 weights (download error).
  - A: Make sure the host has internet access and sufficient disk space. To skip the ImageNet fallback, set `DISABLE_IMAGENET_CHECK=1` and rely on disease-model confidence.

- Q: The app predicts a disease for a non-cow image.
  - A: This project includes heuristics to reduce that, but the most robust fix is to train a dedicated cow/not-cow detector. As a quick fix you can increase `non_natural_prob_threshold` in `app_modified.py` or enable the conservative top-label veto.

- Q: The page is blank or fonts/styles not applied.
  - A: Check browser console for errors and ensure `/static/css/style.css` is loading (server logs will show 200/304 for static assets).

Contact and support
-------------------
If you want, I can:

- Add a textual explanation of which check allowed the upload in the `/predict` JSON response (e.g., `accepted_by: "imagenet_ox"` or `rejection_reason: "non_natural_envelope"`).
- Add a client-side "Proceed anyway" override.
- Scaffold a transfer-learning training script for a robust cow/not-cow classifier.

License & Contributing
----------------------
Add a `LICENSE` file when you choose a license (MIT, Apache-2.0, etc.). If you want I can add an `MIT` license file and a short `CONTRIBUTING.md` with basic guidelines.

----

If you'd like, I can now:

- Add the acceptance/rejection reason to `/predict` JSON responses and update the UI to show it.
- Add a `Proceed anyway` UI toggle to override a rejection.
- Scaffold a small transfer-learning training script for a cow/not-cow classifier.

Tell me which of those you'd like next and I will implement it.
# AI-Powered Cattle Disease Detection System

An advanced machine learning application that detects cattle diseases from images using deep learning and computer vision techniques.

## 🎯 Features

- **AI-Powered Detection**: Uses EfficientNetB0 pre-trained model with transfer learning
- **Disease Classification**: Detects 4 conditions - Healthy, Lumpy Skin Disease, Foot and Mouth Disease, Mastitis  
- **Dataset Imbalance Handling**: Implements data augmentation and class weights for optimal performance
- **Modern Web Interface**: Drag-and-drop image upload with real-time predictions
- **Detailed Analysis**: Provides confidence scores, disease descriptions, and recommendations
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 📁 Project Structure

```
Cattle_Disease_Project/
├── app.py                      # Flask web application
├── train_model.py             # Model training script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── dataset/                   # Training dataset (add your images here)
│   ├── healthy/              # Healthy cattle images
│   ├── lumpy_skin/           # Lumpy skin disease images
│   ├── foot_mouth/           # Foot and mouth disease images
│   └── mastitis/             # Mastitis images
├── templates/
│   └── index.html            # Web interface template
├── static/
│   └── css/
│       └── style.css         # Styling for web interface
└── uploads/                  # Temporary file storage (created automatically)
```

## 🚀 Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
- Add your cattle images to the respective folders in `dataset/`
- Organize images by disease category:
  - `dataset/healthy/` - Healthy cattle images
  - `dataset/lumpy_skin/` - Lumpy skin disease images  
  - `dataset/foot_mouth/` - Foot and mouth disease images
  - `dataset/mastitis/` - Mastitis images

### 3. Train the Model
```bash
python train_model.py
```
This will:
- Load and preprocess your dataset
- Apply data augmentation to handle class imbalance
- Train the EfficientNetB0 model with transfer learning
- Generate training plots and confusion matrix
- Save the trained model as `cattle_disease_model.h5`
- Create `class_indices.json` for class mapping

### 4. Run the Web Application
```bash
python app.py
```

### 5. Access the Application
Open your browser and go to: `http://127.0.0.1:5000`

## 🎯 Key Technical Features

### Dataset Imbalance Handling
- **Data Augmentation**: Rotation, zoom, flip, shift transformations
- **Class Weights**: Penalizes model for misclassifying minority classes
- **Target Accuracy**: >80% across all disease categories

### Model Architecture
- **Base Model**: EfficientNetB0 pre-trained on ImageNet
- **Transfer Learning**: Frozen base layers with custom classification head
- **Input Size**: 224x224 pixels
- **Output**: 4-class softmax classification

### Web Interface
- **Drag & Drop**: Easy image upload functionality
- **Real-time Preview**: Image preview before analysis
- **Detailed Results**: Confidence scores for all classes
- **Disease Information**: Descriptions, severity levels, recommendations
- **Error Handling**: Comprehensive error management
- **Responsive Design**: Mobile-friendly interface

## 📊 Expected Performance

The system is designed to achieve >80% accuracy through:
- Advanced data augmentation techniques
- Class weight balancing for imbalanced datasets
- Transfer learning from EfficientNetB0
- Proper train/validation splitting (80/20)

## 🔧 Troubleshooting

### Common Issues:

1. **Model Loading Error**: Ensure `train_model.py` has been run successfully
2. **Dataset Error**: Check that images are properly organized in dataset folders
3. **Memory Issues**: Reduce batch size in `train_model.py` if needed
4. **Port Issues**: Change port in `app.py` if 5000 is occupied

### System Requirements:
- Python 3.7+
- TensorFlow 2.13.0
- 8GB+ RAM recommended for training
- GPU optional but recommended for faster training

## 📝 Usage Instructions

1. **Training Phase**: Run `train_model.py` with your dataset
2. **Deployment Phase**: Run `app.py` to start the web server
3. **Prediction Phase**: Upload cattle images through the web interface
4. **Analysis**: Review detailed predictions and recommendations

## 🚨 Important Notes

- Always train the model before running the web application
- Ensure proper dataset organization for optimal results
- The system requires internet connection for initial model download
- Generated model files (`cattle_disease_model.h5`, `class_indices.json`) are essential for the web app

## 📄 License

This project is designed for educational and research purposes in veterinary AI applications.

---

**Developed with ❤️ for cattle health monitoring using AI**
