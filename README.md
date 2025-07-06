# 🏛️ Landmark Classification & Tagging for Social Media using Convolutional Neural Networks

This project focuses on developing an image classification algorithm capable of recognizing **landmarks** in user-supplied photographs. The task is especially relevant in real-world applications like **photo-sharing platforms** that rely on location data for tagging, search, and photo organization — even when such metadata is missing.

You will build two convolutional neural networks (CNNs):
1. A custom CNN model trained from scratch
2. A transfer learning model using a pre-trained architecture

---

## 📘 Project Overview

In modern photo storage and sharing services, many uploaded images lack GPS metadata. Automatically identifying the **landmark** in an image allows platforms to infer geographic locations and offer advanced tagging or photo organization features.

The goal of this project is to develop a model that:
- Takes a photo as input
- Predicts the **top-k most relevant landmarks**
- Uses a classification pipeline trained on images from **50 landmark classes** worldwide

---

## 🧠 Learning Objectives

By completing this project, you will:
- Learn to implement a custom CNN using PyTorch
- Apply transfer learning with pre-trained models (e.g., ResNet, VGG)
- Fine-tune models for multi-class image classification
- Evaluate top-k predictions
- Deploy a function that takes a new image and returns landmark predictions

---

## 🧪 Project Steps

### Step 0: Setup & Data
- Download the dataset of labeled landmark images
- Set up required libraries and dependencies

### Step 1: Build a CNN from Scratch
- Design a custom convolutional neural network
- Train and validate on the dataset
- Track accuracy and loss over epochs

### Step 2: Transfer Learning
- Load a pre-trained model (e.g., ResNet50)
- Replace the classifier head with a custom output layer
- Fine-tune the model using the dataset
- Achieve improved performance with fewer resources

### Step 3: Inference Algorithm
- Load a trained model
- Accept user-supplied images
- Predict top-k landmark classes with confidence scores

---

## 🧰 Tools & Libraries

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- PIL (Python Imaging Library)
- torchvision (datasets, models, transforms)

---

---

## 📏 Evaluation

- Model accuracy (Top-1 and Top-k)
- Training and validation loss curves
- Qualitative evaluation through sample predictions

---

## 🎯 Deliverables

- Custom CNN and transfer learning models
- A function that predicts landmarks for any input image
- Code written in a Jupyter Notebook following the provided template
- Final model capable of identifying 50 distinct landmark classes

---

## 👤 Contact

**Waqar Ahmed**  
📧 Email: waqar.nu@gmail.com  
🔗 GitHub: [waqar-ahmed91](https://github.com/waqar-ahmed91)

---

## 📜 License

This project is developed for educational purposes as part of a deep learning course.


