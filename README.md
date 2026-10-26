# Computer Vision + Digital Image Processing — MSCS

A single semester-long practical repository for Computer Vision (CV) and Digital Image Processing (DIP).

## Learning workflow
1. Study the core concept and its mathematics.
2. Implement and run an experiment.
3. Save the code and a short observation.
4. Commit and push to GitHub.

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python lessons/01_image_basics.py
```

## Directory structure
- `lessons/` — lesson-wise runnable experiments
- `notebooks/` — Jupyter study notebooks
- `matlab/` — MATLAB equivalents when applicable
- `src/` — reusable functions
- `datasets/` — sample images (avoid uploading large datasets)
- `results/` — generated output (ignored by Git)
- `projects/` — mini-projects and final semester project

## Progress
- [ ] Lesson 01 — Digital image fundamentals and pixel representation
- [ ] Sampling and quantization
- [ ] Enhancement and histograms
- [ ] Convolution and smoothing
- [ ] Derivatives and edge detection
- [ ] Color processing and segmentation
- [ ] Feature extraction and matching
- [ ] Object detection and recognition
- [ ] Final integrated project

## Lecture sources
University CV Lectures 1–2, DIP lecture slides, image sampling interactive slides, and both course outlines (provided separately).
