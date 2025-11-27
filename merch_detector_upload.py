# MIT License
#
# Copyright (c) 2025 Alvajoy Asante, Aiden Valentine, Sofia Mata Avila, Santhosh Balla, Batman Whiteside
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import gradio as gr
from ultralytics import YOLO
import cv2
import os
import torch

model_path = r"./models/CMM-Yolo11.pt"
model = None

print(f"Attempting to load model from: {model_path}")

try:
    if os.path.exists(model_path):
        model = YOLO(model_path)
        model.to('cpu')
        print(f"✅ Model loaded on CPU: {model_path}")
    else:
        print(f"❌ Model file not found: {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")


def logo_inside(logo, host):
    lx1, ly1, lx2, ly2 = logo
    bx1, by1, bx2, by2 = host
    cx = (lx1 + lx2) // 2
    cy = (ly1 + ly2) // 2
    return bx1 <= cx <= bx2 and by1 <= cy <= by2


def detect_objects(frame):
    global model

    if frame is None:
        return None

    # No flip for uploaded images
    # frame = cv2.flip(frame, 1) 
    frame = frame.copy()

    if model is None:
        return frame

    target_width = 640
    h, w = frame.shape[:2]
    scale = target_width / w

    if scale < 1:
        inference_frame = cv2.resize(frame, None, fx=scale, fy=scale)
    else:
        inference_frame = frame
        scale = 1.0

    try:
        frame_bgr = cv2.cvtColor(inference_frame, cv2.COLOR_RGB2BGR)
        results = model(frame_bgr, conf=0.2, verbose=False, half=False)

        if len(results) == 0 or results[0].boxes is None:
            return frame

        logos, torsos, headgears, all_boxes = [], [], [], []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if scale != 1.0:
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)

            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = model.names[cls].upper()

            if name == "UNCC-LOGO":
                logos.append((x1, y1, x2, y2))
            elif name == "UNCC TORSO":
                torsos.append((x1, y1, x2, y2))
            elif name == "UNCC HEADGEAR":
                headgears.append((x1, y1, x2, y2))

            all_boxes.append((x1, y1, x2, y2, name, conf))

        for (x1, y1, x2, y2, name, conf) in all_boxes:
            label = f"{name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        merch_boxes = []
        for logo in logos:
            for host in torsos + headgears:
                if logo_inside(logo, host):
                    merch_boxes.append(host)

        # Count merch
        merch_count = len(merch_boxes)

        for (x1, y1, x2, y2) in merch_boxes:
            pad_x = int((x2 - x1) * 0.15)
            pad_y = int((y2 - y1) * 0.15)

            nx1 = max(0, x1 - pad_x)
            ny1 = max(0, y1 - pad_y)
            nx2 = min(frame.shape[1], x2 + pad_x)
            ny2 = min(frame.shape[0], y2 + pad_y)

            cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (255, 0, 255), 3)
            cv2.putText(frame, "UNCC MERCH", (nx1, ny1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        cv2.putText(frame, f"UNCC MERCH COUNT: {merch_count}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 0, 255), 2)

    except Exception as e:
        print(f"Error during inference: {e}")

    return frame


with gr.Blocks(title="YOLO Image Detection") as demo:
    gr.Markdown("# UNCC Merch Detector — Image Upload Mode")
    
    with gr.Row():
        input_img = gr.Image(sources=["upload"], type="numpy", label="Upload Image")
        output_img = gr.Image(type="numpy", label="Detection Result")
        
    detect_btn = gr.Button("Detect Merch")
    
    detect_btn.click(detect_objects, inputs=input_img, outputs=output_img)

if __name__ == "__main__":
    demo.launch()
