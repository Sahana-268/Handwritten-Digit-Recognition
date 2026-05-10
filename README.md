# Handwritten Digit Recognition with CNN

A high-accuracy Convolutional Neural Network (CNN) for recognizing handwritten digits using the MNIST dataset. This project includes model training, prediction capabilities, and an interactive web-based drawing application.

## 🎯 Project Overview

- **Dataset:** MNIST handwritten digit dataset (0-9)
- **Model:** Convolutional Neural Network (CNN)
- **Accuracy:** 99.64% on test set
- **Features:** Training, prediction, and interactive web UI

## 📋 Prerequisites

- Python 3.10 or newer
- pip (Python package manager)

## 🚀 Installation

1. **Create a virtual environment:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. **Install dependencies:**
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

##  Training the Model

Train the CNN with custom parameters:

```powershell
python train.py --epochs 10 --batch-size 128 --lr 0.001
```

**What happens:**
- Downloads MNIST dataset to `data/` directory
- Trains the CNN model
- Evaluates performance on validation set each epoch
- Saves the best model to `artifacts/best_model.pt`

**Expected Results:**
- Fresh runs typically achieve 99%+ test accuracy on standard CPU/GPU

## 🔮 Make Predictions

Predict digits from custom images:

```powershell
python predict.py path\to\digit.png --checkpoint artifacts\best_model.pt
```

**Image Requirements:**
- Black ink on white background
- Automatically inverted and centered to match MNIST format

## 🎨 Interactive Web Application

Run the local browser-based drawing app:

```powershell
.\run_app.bat
```

Then open in your browser:
```
http://127.0.0.1:8000
```

**Features:**
- Draw digits directly on canvas
- Real-time prediction while drawing
- Upload existing digit images
- Adjust text size (S / M / L)
- Change brush thickness (S / M / L)
- Draw in different colors

## ✅ Testing

Run the test suite:

```powershell
.\run_test.ps1
```

Or if PowerShell blocks scripts:

```powershell
.\run_test.bat
```

**What the test does:**
- Loads the trained checkpoint
- Tests prediction on sample digits
- Verifies accuracy on first 1,000 MNIST test images

## 📁 Project Structure

```
.
├── train.py                 # Training script
├── predict.py              # Prediction script
├── app.py                  # Web app backend
├── requirements.txt        # Python dependencies
├── artifacts/
│   ├── best_model.pt       # Trained model checkpoint
│   ├── metrics.json        # Training metrics
│   └── confusion_matrix.csv # Model confusion matrix
├── data/
│   └── MNIST/              # Dataset directory
├── src/
│   └── digit_recognizer/
│       ├── data.py         # Data loading utilities
│       ├── model.py        # CNN model definition
│       ├── preprocess.py   # Data preprocessing
│       └── utils.py        # Helper functions
└── web/                    # Frontend files
    ├── index.html
    ├── app.js
    └── style.css
```

## 📦 Dependencies

See `requirements.txt` for complete list of dependencies.

## 📝 License

This project is open source and available for educational purposes.
