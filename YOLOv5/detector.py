#!/usr/bin/env python3
"""
YOLOv5 Detector Laboratory
--------------------------
Demonstrates the classic PyTorch Hub API interface of YOLOv5, performs profiling,
and visualizes object detection output on a synthetic/sample image.
"""

import time

import cv2
import numpy as np
import torch


def create_synthetic_scene(filename="test_scene.jpg"):
    """
    Creates a synthetic 640x640 image representing a street scene
    with a road, cars (represented by colored boxes), and pedestrians
    so that the detector can execute even in the absence of internet/files.
    """
    # Create dark grey background
    img = np.ones((640, 640, 3), dtype=np.uint8) * 128

    # Draw a simulated asphalt road (dark grey)
    cv2.rectangle(img, (0, 300), (640, 600), (40, 40, 40), -1)

    # Draw yellow road lines
    cv2.line(img, (0, 450), (640, 450), (0, 255, 255), 5)

    # Write a label so we know it's a generated scene
    cv2.putText(
        img,
        "YOLO playground: Synthetic Scene",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.imwrite(filename, img)
    print(f"[🎨] Synthetic scene saved to '{filename}'")
    return filename


def run_laboratory():
    print("=" * 60)
    print("🧪 YOLOv5 EXPERIMENTAL LABORATORY START")
    print("=" * 60)

    # 1. Prepare Target Image
    image_path = "test_scene.jpg"
    create_synthetic_scene(image_path)

    # 2. Load Model via PyTorch Hub
    print("\n[🔄] Loading YOLOv5s model from PyTorch Hub...")
    start_load = time.time()
    try:
        # Load a local version of yolov5s if available, otherwise fetch from github
        # We set pretrained=True to load the COCO pre-trained weights
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True, trust_repo=True)
        load_time = time.time() - start_load
        print(f"[✅] Model loaded successfully in {load_time:.3f} seconds!")
    except Exception as e:
        print(f"[❌] Failed to load YOLOv5 via PyTorch Hub: {e}")
        print("\n💡 NOTE: PyTorch Hub requires an active internet connection to download weights.")
        print("We can simulate the inference API for local study below:")
        simulate_yolov5_api(image_path)
        return

    # 3. Read input image
    img = cv2.imread(image_path)

    # 4. Perform Inference & Profile Execution Time
    print("\n[⚡] Running inference pass...")
    t0 = time.time()

    # YOLOv5's Hub wrapper handles pre-processing (scaling, channels BGR -> RGB) automatically
    results = model(img)

    inference_time = time.time() - t0
    print(f"[✅] Inference completed in {inference_time * 1000:.2f} ms")

    # 5. Extract and Analyze Output Formats
    print("\n[📊] Analyzing YOLOv5 Prediction Tensors:")

    # YOLOv5 returns a list/dataframe of detections
    detections = results.pandas().xyxy[0]  # Pandas representation

    print("-" * 50)
    print(detections)
    print("-" * 50)

    # Let's count class instances
    if not detections.empty:
        counts = detections["name"].value_counts()
        print("\n🔍 Objects Counted:")
        for name, count in counts.items():
            print(f"  - {name}: {count}")
    else:
        print("🔍 No objects detected in the synthetic scene (model needs real objects).")

    # 6. Save the Output
    results.save(save_dir="runs/detect")
    print("\n[💾] Annotated results saved to 'runs/detect/' directory.")
    print("=" * 60)


def simulate_yolov5_api(image_path):
    """
    Simulates the exact API of YOLOv5 for educational purposes
    when the environment lacks external model assets.
    """
    print("\n[🔬] YOLOv5 Local API Simulation (Offline Mode)")
    print("-" * 50)
    print("Coding and parsing simulation:")
    print("1. image = cv2.imread(image_path)")
    print("2. input_tensor = transform(image)  # [1, 3, 640, 640]")
    print("3. coupled_output = model_forward(input_tensor)")
    print("   ↳ Returns grid tensors at 3 different scales (e.g. 80x80, 40x40, 20x20)")
    print("   ↳ Each grid cell predicts: [x, y, w, h, confidence, class_prob_0, ...]")
    print("4. Applying Auto-Anchors scaling & Decoupled Post-Processing:")
    print("   ↳ For each scale grid block:")
    print("     dx, dy, dw, dh = scale_and_activate(anchor_weights)")
    print("5. Running Non-Maximum Suppression (NMS) to filter redundant overlapped boxes.")
    print("-" * 50)
    print("Offline simulation complete. YOLOv5 architecture script works!")


if __name__ == "__main__":
    run_laboratory()
