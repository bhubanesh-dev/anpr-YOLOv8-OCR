﻿# anpr-YOLOv8-Optical Charecter recognition
This is my Bachelors degree project. It implements YOLOv8 and a CNN based custom OCR model to perform Automatic Number Plate Detection(ANPR) on Indian four wheelers.

The system is optimized to perform OCR on 2-Row Indian HSRNs(High Security Number Plate).

The model detects 1-Row HSRNs(from Bikes) but to doesn't perform OCR correctly.

It doesn't detect Indian Green HSRNs(for EVS) at all.

# Some things to Note
The license detection model is completely trained by me using YOLOv8 in Google Collab. You can check the YOLO_Models folder for the code.
The OCR Model & Implementation I used is from the post Characters Segmentation and Recognition for Vehicle License Plate by Minh Thang Dang(Thanks !!!).
# Important !!! Read it
When I implemented the project, I used Tensorflow for GPU v2.12 and PyTorch on CUDA 11.8, which did not conflict at all. But if you try to install current tensorflow[and-cuda] package , they will conflict. So you have to set up two different environments to run the project , i.e., one for the License Plate Detection and one for OCR if you want to use GPU in project. Normal CPU users don't need to do this.
