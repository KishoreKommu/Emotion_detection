# 🎭 Real-Time Emotion Detection using Deep Learning

## 📌 Project Overview

This project focuses on building a **Real-Time Emotion Detection System** using **Deep Learning and Computer Vision**. The system is capable of detecting human faces from images or live webcam feed and predicting the **facial emotion** displayed.

The model is trained using a Convolutional Neural Network (CNN) on a labeled facial expression dataset and can classify emotions such as:

* Angry 😠
* Disgust 🤢
* Fear 😨
* Happy 😊
* Neutral 😐
* Sad 😢
* Surprise 😲

---

## 🚀 Features

* 📷 Detects faces using OpenCV
* 🧠 Predicts emotions using CNN model
* 🖼️ Works on both **images** and **real-time webcam**
* 🎯 Displays emotion with confidence score
* 🔄 Smooth real-time predictions using frame averaging
* 📊 Visualizes training accuracy and loss

---

## 🧠 Technologies Used

* **Python**
* **TensorFlow / Keras**
* **OpenCV**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**

---

## 📂 Project Structure

```
emotion_detection/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── models/
│   └── emotion_model.h5
│
├── test_images/
│   └── test1.jpg
│
├── train_emotion_model.py
├── test_emotion_model.py
├── realtime_emotion_detection.py
└── README.md
```

---

## 📊 Dataset

The model is trained on a facial emotion dataset (FER-2013 style) containing grayscale face images categorized into 7 emotion classes.

Each class is stored in separate folders:

```
train/
    angry/
    happy/
    sad/
    neutral/
    ...
```

---

## ⚙️ How It Works

### 1️⃣ Data Preprocessing

* Images resized to **48x48**
* Converted to grayscale
* Normalized pixel values (0–1)
* Data augmentation applied (rotation, flip, zoom)

---

### 2️⃣ Model Architecture

A Convolutional Neural Network (CNN) is used:

```
Input (48x48x1)
→ Conv2D + ReLU
→ MaxPooling
→ Conv2D + ReLU
→ MaxPooling
→ Conv2D + ReLU
→ MaxPooling
→ Flatten
→ Dense Layer
→ Dropout
→ Output (Softmax - 7 classes)
```

---

### 3️⃣ Training

* Loss Function: `categorical_crossentropy`
* Optimizer: `Adam`
* Epochs: 10–30
* Batch Size: 32

---

### 4️⃣ Prediction Pipeline

For both image and webcam:

```
Input → Face Detection → Face Crop → Resize → Normalize → Model Prediction → Display Emotion
```

---

## 🖼️ Image Prediction

To test on a single image:

```bash
python test_emotion_model.py
```

The system:

* Detects face
* Predicts emotion
* Displays result with confidence

---

## 🎥 Real-Time Webcam Detection

To run real-time emotion detection:

```bash
python realtime_emotion_detection.py
```

* Opens webcam
* Detects face continuously
* Displays emotion live
* Press **'q'** to exit

---

## 📈 Results

* Successfully detects and classifies facial emotions
* Works in real-time using webcam
* Accuracy improves with more training epochs and data

---

## ⚠️ Limitations

* Model may confuse similar emotions (e.g., neutral vs sad)
* Performance depends on lighting and face visibility
* Requires clear frontal face for best accuracy

---

## 🔧 Future Improvements

* Improve accuracy using Transfer Learning (ResNet, MobileNet)
* Add emotion tracking over time
* Integrate into a **Django web application**
* Deploy as a web app or mobile app
* Combine with Face Mask Detection

---

## 🎯 Applications

* Smart surveillance systems
* Mental health analysis
* Human-computer interaction
* Smart classrooms
* Customer sentiment analysis

---

## 👨‍💻 Author

**Kishore Kommu**
B.Tech IT Student | Aspiring Software Engineer

---

## ⭐ Conclusion

This project demonstrates the practical application of **Deep Learning and Computer Vision** in analyzing human emotions. It provides a strong foundation for building intelligent AI-based systems and enhances real-world problem-solving skills.

---
