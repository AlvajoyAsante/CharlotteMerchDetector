# Charlotte Merch Detector

![Charlotte Merch Detector demo](./docs/hero.jpg)

A lightweight live detector that identifies University of North Carolina at Charlotte (UNCC) merchandise in webcam video using a custom YOLO model.

This project runs a small Gradio app that streams your webcam, runs inference with a custom Ultralytics YOLO model, and highlights detected UNCC logos on torsos and headgear. It also counts and labels items considered "UNCC merch" when a UNCC logo is found inside a detected torso or headgear region.

## Features

- Real-time webcam detection (CPU-friendly)
- Custom YOLO model (`models/CMM-Yolo11.pt`) trained to detect UNCC-specific classes: UNCC-LOGO, UNCC TORSO, UNCC HEADGEAR
- Visual overlays showing detected boxes, labels, and a merch count

## Repository structure

- `merch_detector.py` — main application (Gradio frontend + YOLO inference)
- `models/` — folder containing model weights
	- `CMM-Yolo11.pt` — custom trained model used by the app
	- `yolov8n.pt` — (optional) base model checkpoint
- `requirements.txt` — Python dependencies
- `presentation/Final Presentation.pdf` — project presentation

## Requirements

This project was developed for CPU usage. Install dependencies from `requirements.txt` into a virtual environment.

Minimum tested packages (see `requirements.txt` for full list):

- Python 3.8+
- ultralytics
- opencv-python
- gradio

## Installation

1. Create and activate a virtual environment (Windows PowerShell):

```
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Verify the `models/CMM-Yolo11.pt` file exists. If you don't have it, place your trained Ultralytics checkpoint at that path or update `merch_detector.py` to point to your model.

## Usage

Run the Gradio app which will open a browser window showing the webcam stream and live detections:

```
python merch_detector.py
```

The app uses your default webcam. It streams frames to the model, draws bounding boxes and labels, and displays a running "UNCC MERCH COUNT".

## How detection works (brief)

- The YOLO model outputs bounding boxes for multiple classes.
- Detected `UNCC-LOGO` boxes are compared to `UNCC TORSO` and `UNCC HEADGEAR` boxes.
- If a logo's center lies inside a torso/headgear box, that host box is considered "UNCC MERCH" and is highlighted and counted.

## Notes & troubleshooting

- CPU vs GPU: The app forces the model onto CPU (`model.to('cpu')`). If you have a CUDA-capable GPU and want GPU inference, remove or change that line in `merch_detector.py` and ensure PyTorch + CUDA are installed.
- Model path: By default the app expects `./models/CMM-Yolo11.pt`. Edit `model_path` in `merch_detector.py` if you store weights elsewhere.
- Webcam access: Gradio's `Image` component uses browser access to your webcam; grant permission when prompted.
- Confidence threshold: The code uses a 0.75 confidence cutoff in `model(..., conf=0.75)`. Lower this if you need more detections, but expect more false positives.

## Development & contributions

If you improve the detector or retrain the model, please:

1. Add the new checkpoint to the `models/` folder or update `model_path`.
2. Consider adding a small README note describing how the model was trained (dataset, augmentations, epochs, backbone).

## License

MIT License; Copyright (c) 2025; Alvajoy Asante, Aiden Valentine, Sofia Mata Avila, Santhosh Balla, Batman Whiteside
