# Handwritten Digit Recognition with CNN

This project trains a high-accuracy Convolutional Neural Network on the
MNIST handwritten digit dataset, which contains labeled grayscale images of
digits from `0` to `9`.

## Setup

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you are using the bundled Codex Python runtime in this workspace, replace
`python` with:

```powershell
& 'C:\Users\SAHANA FATHIMA\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
```

## Train

```powershell
python train.py --epochs 10 --batch-size 128 --lr 0.001
```

The script downloads MNIST into `data/`, trains the CNN, evaluates it on a
validation split each epoch, and saves the best model to:

```text
artifacts/best_model.pt
```

The included trained checkpoint reached `99.65%` test accuracy on MNIST. With
the default architecture and 8-10 epochs, fresh runs should typically reach
about `99%+` test accuracy on a normal CPU/GPU setup.

## Predict a Custom Digit Image

```powershell
python predict.py path\to\digit.png --checkpoint artifacts\best_model.pt
```

For black ink on a white background, the predictor automatically inverts and
centers the digit to better match MNIST.

## Live Drawing App

Run the local browser app:

```powershell
.\run_app.bat
```

Then open:

```text
http://127.0.0.1:8000
```

Draw a digit on the canvas. The app predicts automatically while you draw, and
you can also click `Predict`. Use the `S / M / L` font controls to resize the
interface text, the `S / M / L` brush controls to change drawing thickness, and
the ink swatches to draw in different dark colours.

## Test The Project

From this project folder, run:

```powershell
.\run_test.ps1
```

If PowerShell blocks scripts on your machine, run:

```powershell
.\run_test.bat
```

The test loads the trained checkpoint, predicts a sample digit, and verifies
accuracy on the first 1,000 MNIST test images.

## Project Structure

```text
.
|-- train.py
|-- predict.py
|-- requirements.txt
|-- src/
    |-- digit_recognizer/
        |-- data.py
        |-- model.py
        |-- preprocess.py
        |-- utils.py
```
