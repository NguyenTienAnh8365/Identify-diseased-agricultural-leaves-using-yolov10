# Identify-diseased-agricultural-leaves-using-yolov10

This project uses the YOLOv10 model to classify and detect agricultural leaf diseases. The goal is to detect and classify diseases on leaves to assist farmers in caring for and managing their crops.

## Table of Contents
- [Introduction](#introduction)
- [System Requirements](#system-requirements)
- [Data Preparation](#data-preparation)
- [Installation](#installation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Prediction](#prediction)

## Introduction

In agriculture, early detection and classification of leaf diseases are crucial to ensure the best growth for crops. This project applies the YOLOv10 model to perform this task with high accuracy and fast processing time.

## System Requirements

- Python 3.x
- Google Colab or a computer with a GPU
- Necessary Python libraries (listed in `requirements.txt`)

## Data Preparation

1. Connect Google Drive.

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. Create a directory to store the data.

    ```python
    import os
    datasets_path = '/content/drive/MyDrive/Project_prediction_yolov10/datasets'
    os.makedirs(datasets_path, exist_ok=True)
    ```

3. Download and unzip the dataset.

    ```bash
    !gdown --output "{datasets_path}/face_dataset.zip" "https://drive.google.com/uc?id=1LBpKKXFcfvUVgyk3tgQH6YxNp1KXX0Va"
    !unzip -q '/content/drive/MyDrive/Project_prediction_yolov10/datasets/face_dataset.zip' -d '/content/drive/MyDrive/Project_prediction_yolov10/datasets'
    ```

## Installation

1. Clone the YOLOv10 repository and install dependencies.

    ```bash
    !git clone https://github.com/THU-MIG/yolov10.git
    %cd yolov10
    !pip install -q -r requirements.txt
    !pip install -e .
    ```

2. Install additional required libraries.

    ```bash
    !pip install -U ultralytics
    ```

## Model Training

1. Download the initial weights of YOLOv10.

    ```bash
    !wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt -O //content/drive/MyDrive/Project_prediction_yolov10/yolov10n.pt
    ```

2. Initialize and configure the YOLOv10 model.

    ```python
    from ultralytics import YOLO

    model_path = "/content/drive/MyDrive/Project_prediction_yolov10/yolov10n.pt"
    model = YOLO(model_path)

    yaml_path = "/content/drive/MyDrive/Project_prediction_yolov10/datasets/data.yaml"
    EPOCHS = 100
    IMG_SIZE = 480
    BATCH_SIZE = 32

    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    import torch
    torch.cuda.empty_cache()
    ```

3. Train the model.

    ```python
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        # model=model_path
    )
    ```

4. Copy and save the best model weights.

    ```python
    import shutil

    source_path = '/content/runs/detect/train/weights/best.pt'
    destination_path = '/content/drive/MyDrive/Project_prediction_yolov10/best.pt'

    shutil.copy(source_path, destination_path)
    print(f"File copied to {destination_path}")
    ```

## Model Evaluation

1. Evaluate the model on the test set.

    ```python
    model = YOLO("/content/drive/MyDrive/Project_prediction_yolov10/best.pt")
    model.val(
        data=yaml_path,
        imgsz=IMG_SIZE,
        split='test'
    )
    ```

2. Display the validation results.

    ```python
    import matplotlib.pyplot as plt

    # Labels
    img = plt.imread("/content/runs/detect/val/val_batch2_labels.jpg")
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Predictions
    img = plt.imread("/content/runs/detect/val/val_batch2_pred.jpg")
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    ```

## Prediction

1. Make predictions on new images.

    ```python
    from PIL import Image
    import matplotlib.pyplot as plt

    input_image_path = "/content/drive/MyDrive/Project_prediction_yolov10/Image/test_image.png"
    image = Image.open(input_image_path)

    results = model.predict(source=input_image_path)

    # Plot the image and draw the bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    # Extract predictions
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1

            rect = plt.Rectangle((x1, y1), width, height, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)

            plt.text(x1, y1, f'{label} {score:.2f}', color='white', fontsize=12, bbox=dict(facecolor='red', edgecolor='red', pad=2.0))

    plt.axis('off')
    plt.show()
    ```

I hope this guide will help you implement the project of classifying and detecting diseases on agricultural leaves easily.
