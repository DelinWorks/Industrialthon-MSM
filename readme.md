---

# Industrialthon PPE Computer Vision Project
Manufacturing Safety and Equipment YOLOv5 trained models

## Models
YOLOv5 is used for object detection and is trained on custom datasets.
see: https://github.com/ultralytics/yolov5 to train your custom datasets.

## Datasets used.
1. Dataset 'construction-v2-dataset' is used to detect people and personal protictive equipment.
2. Dataset 'leaks-v3-dataset' is used to detect oil leaks and fluids on the ground.
3. Dataset 'garbage-v2-dataset' is used to detect garbage and waste in the facility.

`NOTE: the version indicates how many times the dataset has been replaced by a refined and better one.`

## Training Information
   construction-v2-dataset model summary:
   ```
   Training: 1112 images, Testing & Validation: 378 images
   Model summary: 157 layers, 7012822 parameters, 0 gradients, 15.8 GFLOPs
                  Class     Images  Instances    Precision      Recall     mAP50   mAP50-95
                    all        398       1073        0.921       0.798     0.899       0.67
   ```
   leaks-v3-dataset model summary:
   ```
   Training: 8457 images, Testing & Validation: 1213 images
   Model summary: 157 layers, 7012822 parameters, 0 gradients, 15.8 GFLOPs
                  Class     Images  Instances    Precision      Recall     mAP50   mAP50-95
                    all        370        451        0.922       0.845     0.917        0.6
   ```
   garbage-v2-dataset model summary:
   ```
   Training: 1397 images, Testing & Validation: 598 images
   Model summary: 157 layers, 7012822 parameters, 0 gradients, 15.8 GFLOPs
                  Class     Images  Instances    Precision      Recall     mAP50   mAP50-95
                    all        398       1073        0.871       0.877     0.922      0.665
   ```
NOTE: the number of layers and parameters is the same because we are using yolov5s (small) weights for fast training
      since we are time constrained, yolov5m and yolov5l (medium and large) can be used for better object detection and
      accuracy. yolov5-sahi (Slicing Aided Hyper Inference) can be used to help in detecting small objects.
      see: https://github.com/zahidesatmutlu/yolov5-sahi
      
## Usage
1. Install required dependencies 
   ```
   pip install opencv-python
   pip install ultralytics
   pip install torch torchvision torchaudio
   pip install opencv-contrib-python
   ```
2. Train your own custom dataset or use the pretrained ones
   ```
   python yolov5/train.py --batch 16 --epochs 20 --data dataset/data.yaml --weights yolov5s.pt
   ```
3. Use more complex weights for better detection and accuracy (will increase training time)
   ```
   yolov5m.pt, yolov5l.pt, yolov5x.pt (medium, large and extra large weights)
   ```
4. Run `MSD.py` to test object detection through camera using OpenCV.
5. Run `MSD2.py` to test object detection using images in test folder (press n to cycle images).

## Nvidia CUDA (GPU acceleration)
You need to have GPU acceleration enabled for faster peformance, otherwise CPU is used
look up a guide on how to install nvidia cuda drivers on your machine if it has an nvidia gpu.
for cuda ensure you have the graphics drivers installed along with Nvidia CUDA toolkit
https://developer.nvidia.com/cuda-downloads

## Features

- Realtime Object detection and recognition using GPU acceleration.
- Detect multiple types of objects from custom datasets.

## Example screen shots
![image](https://github.com/user-attachments/assets/3fab06fe-5a8e-43d9-8e99-3e8d2776e358)
![image](https://github.com/user-attachments/assets/5666fec6-5b5b-4be2-8b69-b958610f160b)
![image](https://github.com/user-attachments/assets/fca185af-c0ea-415d-8859-9dea0380dab6)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create your feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'feature X'`
4. Push to the branch: `git push origin feature-name`
5. Open a pull request
