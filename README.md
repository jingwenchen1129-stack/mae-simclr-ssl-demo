# Self-Supervised Vision Demo

This project is a compact interactive demo for image-transform self-supervised learning.

## What It Includes

- `app.py`: core source file and Streamlit web app.
- `outputs/self_supervised_vision_report.pdf`: experiment report with prompt, screenshots, summary, and Agent/LLM information.
- `make_report.py`: reproducible report generator.

## Tasks

- Masked Autoencoder style image reconstruction:
  - Split a generated image into patches.
  - Randomly mask patches.
  - Train a small NumPy patch decoder to reconstruct masked patch colors.
  - Compare two masking ratios.

- Rotation prediction:
  - Rotate each image by 0, 90, 180, or 270 degrees.
  - Train a NumPy softmax classifier to predict the rotation.
  - Show loss and accuracy curves.

## Run

```bash
streamlit run app.py
```

## Agent / LLM

- Codex GPT-5
