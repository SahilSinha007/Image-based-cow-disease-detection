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
***Cattle Disease Detection — Detailed Project Overview***

**Project Summary**

- **Purpose:** Provide an end-to-end demo that classifies cattle images into disease categories (lumpy skin, mastitis, foot-and-mouth disease, and healthy) using a Keras/TensorFlow model served by a Flask backend and a lightweight web UI.
- **Key Components:**
  - **Flask server:** `app_modified.py` (serves model inference API, image vetting heuristics, and static pages).
  - **Frontend:** `templates/index.html` and `static/css/style.css` (upload UI, image preview, dark-mode toggle, guidance cards).
  - **Model weights:** `cattle_disease_model_simple.h5` (Keras saved model file) and `class_indices.json` (mapping from model output index to class labels).
  - **Dataset (local):** `dataset/` (training images grouped by class). NOTE: large dataset files may increase repo size — see "Repository size" below.

**Why this project exists**

- To demonstrate a practical image-classification pipeline for an applied veterinary use-case. It shows how to:
  - Collect and structure image data for transfer learning.
  - Train a Keras model for multi-class classification of disease vs healthy.
  - Deploy the model behind a Flask API with added heuristics to reduce false positives from non-cow images.
  - Provide a simple browser UI to upload images and obtain predictions.

**Repository structure (important files)**

- `app_modified.py`: Flask application with `/` (UI), `/predict` (POST image inference), and other helper endpoints. Contains ImageNet-based vetting heuristics to block non-cow inputs.
- `templates/index.html`: Main web page — upload controls, preview, image guidance cards, dark-mode toggle.
- `static/css/style.css`: CSS styles including light/dark mode and guideline card styles.
- `cattle_disease_model_simple.h5`: Trained Keras model file used for prediction.
- `class_indices.json`: JSON mapping of model class indices to human-readable labels.
- `dataset/`: Example training images organized in subfolders per class (e.g., `lumpy_skin/`, `mastitis/`, `foot_mouth/`, `healthy/`).
- `requirements.txt`: Python dependencies used for local environment.

**Getting started**

**Prerequisites**

- Python 3.8+ (3.9 or 3.10 recommended for TensorFlow compatibility)
- A virtual environment (recommended) and pip

**Quick install**

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Flask app (development mode):

```powershell
python app_modified.py
```

4. Open the UI at `http://127.0.0.1:5000/` in your browser.

**How to use the web UI**

- Click or drag an image onto the upload area.
- The UI will show a preview and you can press **Analyze** to send the image to the backend.
- The UI includes an **Image upload guidelines** panel describing recommended capture angles and close-ups for each disease:
  - **Lumpy Skin:** full-body side view showing raised nodules on skin.
  - **Foot-and-Mouth Disease (FMD):** close-up of mouth, tongue, or foot lesions; include interior mouth shots if possible.
  - **Mastitis:** clear close-up of the udder/teats; non-blurry and well-lit.
- The UI supports a light background and a persisted dark-mode toggle stored in `localStorage`.

**Screenshots**

Below are placeholder screenshots showing the web UI and example results. Add your screenshots to a `screenshots/` folder at the project root and update filenames as needed.

![UI - Upload area](screenshots/screenshot-upload.png)

![UI - Prediction result](screenshots/screenshot-result.png)

Tip: Commit small, optimized JPG/PNG screenshots (under 500 KB) or host them externally if you want to keep the repository size small.

**API and backend behavior**

**Endpoints**

- `GET /` — Serves the HTML UI (`templates/index.html`).
- `POST /predict` — Accepts a multipart form with `file` and returns JSON prediction or a rejection message.

**Inference workflow (what happens on `/predict`)**

1. Server reads uploaded image and applies preprocessing expected by the disease model (resize, scaling, channel order).
2. Server runs the disease classifier (Keras model) to produce softmax scores for each class.
3. Server optionally runs an ImageNet pretrained MobileNetV2 (lazy-loaded fallback) on a copy of the image to get coarse semantic labels. This is used as a heuristic filter to detect obvious non-cow images (documents, screenshots, artifacts) and to boost confidence if the image contains bovine-related labels.
4. A combined heuristic decides whether to accept the image for final classification:
   - If the disease-model top class probability is above a configurable threshold (default ~0.35), accept and return prediction.
   - If ImageNet strongly indicates the image is a document/screenshot or otherwise non-natural, the backend rejects to avoid false positive disease predictions on non-cow images.
   - If ImageNet shows bovine-related labels (e.g., "ox", "cow", "bull") even at modest probability, it can help accept close-up images that lack broad context.

**Returned JSON**

Success (HTTP 200):

```json
{
  "status": "success",
  "predictions": [
    {"label": "lumpy_skin", "probability": 0.823},
    {"label": "mastitis", "probability": 0.105},
    {"label": "healthy", "probability": 0.072}
  ],
  "accepted": true
}
```

Rejection (HTTP 400):

```json
{
  "status": "rejected",
  "message": "Image appears to be a document (envelope) — please upload a cow image.",
  "accepted": false
}
```

**Heuristics and tuning guide**

The project uses a few conservative heuristics to avoid predicting diseases on non-cow images. These are implemented in `app_modified.py` and controlled by constants near the top of the file. Typical values used during development:

- `DISEASE_CONF_THRESHOLD = 0.35` — Disease model confidence above which predictions are accepted.
- `bovine_prob_threshold ≈ 0.05` — If ImageNet assigns >5% mass to bovine-related labels, this helps accept a close-up.
- `non_natural_prob_threshold ≈ 0.15` — If ImageNet assigns >15% to non-natural/document labels (envelope, binder, monitor, etc.), reject.

**Important practical notes:**

- These heuristics are conservative; they reduce obvious false positives but can still fail on ambiguous images. If you find legitimate cow close-ups being rejected, increase the `bovine_prob_threshold` tolerance or lower the `non_natural_prob_threshold` temporarily while logging the ImageNet labels to determine better keywords.
- Long-term robust fix: train a dedicated binary classifier (cow vs not-cow) using transfer learning and a curated dataset of both cow close-ups and likely confounders (documents, screens, pets, landscapes). This is recommended for production.

**Model training notes**

- The repository includes a training script `train_simple_robust.py` used to prepare the `cattle_disease_model_simple.h5` model. Key steps typically are:
  - Organize images in `dataset/<class_name>/` folders.
  - Use data augmentation (rotation/flip/brightness) to improve generalization for different lighting and angles.
  - Use a pre-trained backbone (e.g., MobileNetV2 / EfficientNet) and fine-tune on the task.
- If you want to retrain:
  1. Ensure your `dataset/` has sufficiently diverse images for each class.
  2. Update hyperparameters in `train_simple_robust.py` as needed.
  3. Train on a GPU for reasonable speed.

**Repository size & Git guidance**

- The `dataset/` folder contains many images and increases repository size. For a clean public repo you may want to:
  - Remove dataset files from git history using `git filter-repo` or the BFG Repo-Cleaner.
  - Add datasets to `.gitignore` and host them externally (OSF, Zenodo, Google Drive) with download scripts.
  - Use Git LFS for large binary files if you need to keep some images tracked.

Example commands to remove the dataset from tracking (keep locally):

```powershell
git rm -r --cached dataset
echo "dataset/" >> .gitignore
git add .gitignore
git commit -m "Remove dataset from repo and add to .gitignore"
git push origin main
```

**Development, tests, and CI**

- This project doesn't include an automated test suite yet. For local testing:
  - Start the server: `python app_modified.py` and test uploads via the UI.
  - Use `curl` or a small script to POST images to `/predict` for bulk checks.

Example curl request

```powershell
curl -X POST -F "file=@path\to\image.jpg" http://127.0.0.1:5000/predict
```

**Troubleshooting**

- TensorFlow errors on import: ensure installed TensorFlow is compatible with your Python version. On Windows, TensorFlow 2.10 or later usually requires Python 3.8–3.10.
- Model not found: confirm `cattle_disease_model_simple.h5` is present in the repo root or update `app_modified.py` to point at the correct path.
- False positives on documents/screens: inspect server logs — `app_modified.py` logs ImageNet decoded labels when the ImageNet fallback is used. Tune thresholds or add keywords to the `non_natural_keywords`/`bovine_keywords` lists in `app_modified.py`.

**Security & Privacy**

- NOTE: the project previously used an `uploads/` folder for temporary storage. That folder has been removed from the repository; by default the current server processes uploaded images in-memory and does not persist them to disk. If you want to persist uploads again (for debugging or audit logs), create an `uploads/` folder at the repo root and update `app_modified.py` to save incoming files, and be sure to add the folder to `.gitignore` to avoid committing user images.

If running in production, add explicit policies for data retention, secure storage, and user consent.

**Contributing**

- If you want to contribute improvements:
  - Open issues for bugs or feature requests (e.g., a dedicated cow/not-cow classifier, CI/CD, frontend UX fixes).
  - Fork the repo, create feature branches, and open pull requests. Describe changes and include reproducible steps.

**License & Acknowledgements**

- This repository currently contains no explicit license file. If you plan to share publicly, add a `LICENSE` file (MIT or Apache 2.0 are common choices).
- Acknowledge any third-party datasets you used for training if they require attribution.

**Contact / Maintainer**

- Repository owner: GitHub `SahilSinha007` (see project remote URL).

**Next steps I can help with**

- Remove `dataset/` from the repo and add Git LFS integration.
- Add a small binary cow/not-cow classifier to reduce heuristic complexity.
- Add a structured `rejection_reason` field to `/predict` JSON and a front-end "Proceed anyway" override that allows users to bypass the heuristic (useful for research/testing).

If you'd like, I can now make any of the above changes (cleanup dataset, add LFS, implement UI override, or include a cow/not-cow classifier scaffold). Tell me which you'd like next.
