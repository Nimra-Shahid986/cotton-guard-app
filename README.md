# 🌿 Cotton Guard — Cotton Leaf Disease Detection System

A deep learning-powered Streamlit application for detecting diseases in cotton leaves, designed for Pakistani cotton farmers.

## Features

- **Dual Model Support**: Swin Transformer (SAR-CLD 2024) & ConvNeXt-T (Cotton Leaf Disease)
- **11 Disease Classes**: 7 classes (SAR-CLD) + 4 classes (Cotton Leaf Disease)
- **Real-time Inference**: Upload a leaf photo → instant diagnosis with confidence scores
- **Disease Info & Treatment**: Symptoms, severity, treatment, and prevention tips
- **Built-in Chatbot**: Ask questions about diseases, treatments, and how the app works

## Project Structure

```
cotton-leaf-app/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .streamlit/
│   └── config.toml               # Streamlit dark theme config
└── models/
    ├── swin_t_best.pt            # Swin-T weights (SAR-CLD 2024, 7 classes)
    └── convnext_t_best.pt        # ConvNeXt-T weights (Cotton Leaf Disease, 4 classes)
```

## Datasets & Models

| Dataset | Classes | Best Model | Accuracy |
|---------|---------|-----------|----------|
| SAR-CLD 2024 | 7 (Bacterial Blight, Curl Virus, Healthy Leaf, Herbicide Growth Damage, Leaf Hopper Jassids, Leaf Redding, Leaf Variegation) | Swin-T | ~98.4% |
| Cotton Leaf Disease | 4 (Bacterial Blight, Curl Virus, Fussarium Wilt, Healthy) | ConvNeXt-T | ~97.7% |

## Step-by-Step Deployment Guide

### Step 1: Create a GitHub Repository

1. Go to [github.com](https://github.com) → click **"New repository"**
2. Name it `cotton-guard-app` (or any name you prefer)
3. Set it to **Public** (required for free Streamlit Cloud deployment)
4. Do NOT initialize with README (we have our own)
5. Click **"Create repository"**

### Step 2: Upload Files to GitHub

**Option A — GitHub Web Interface (Easiest)**

1. In your new repo, click **"uploading an existing file"** link
2. Drag and drop these files one by one:
   - `app.py`
   - `requirements.txt`
   - `README.md`
3. Click **"Commit changes"**
4. Now create the folders:
   - Click **"Add file"** → **"Create new file"**
   - Type `.streamlit/config.toml` as the filename
   - Paste the contents of `config.toml` into the editor
   - Click **"Commit changes"**
5. For models — since they are large files:
   - Click **"Add file"** → **"Create new file"**
   - Type `models/.gitkeep` → commit (this creates the folder)
   - Then upload model files into the `models/` folder

> ⚠️ **Large File Note**: `convnext_t_best.pt` is ~107MB which exceeds GitHub's 100MB limit for web upload. You'll need **Git LFS**:

**Option B — Git Command Line with LFS (Recommended for large models)**

```bash
# Clone your empty repo
git clone https://github.com/YOUR_USERNAME/cotton-guard-app.git
cd cotton-guard-app

# Install and set up Git LFS
git lfs install
git lfs track "*.pt"

# Copy all project files into this folder (app.py, requirements.txt, etc.)
# Copy models/ folder with both .pt files

# Add everything
git add .gitattributes
git add .
git commit -m "Initial commit: Cotton Guard app with models"
git push origin main
```

### Step 3: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your **GitHub** account
3. Click **"New app"**
4. Fill in:
   - **Repository**: `YOUR_USERNAME/cotton-guard-app`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy!"**
6. Wait 3-5 minutes for it to install dependencies and start

### Step 4: Test Your App

1. Once deployed, you'll get a URL like `https://cotton-guard-app.streamlit.app`
2. Upload a cotton leaf image
3. Select dataset → click **Analyze Leaf**
4. Try the chatbot on the right side!

## Running Locally (Optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Training Details

Both models were trained using:
- 80/20 stratified train/validation split
- Focal Loss with class weights (handles class imbalance)
- AdamW optimizer (lr=3e-4, weight_decay=1e-4)
- Cosine Annealing LR scheduler
- Data augmentation: flips, rotations, color jitter
- Early stopping (patience=10)
- ImageNet pre-trained weights as initialization
- WeightedRandomSampler for balanced batch sampling
