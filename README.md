# 🎭 Emotion AI Dashboard

### Real-Time Facial Emotion Recognition using Deep Learning

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python"/>
<img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow"/>
<img src="https://img.shields.io/badge/PyTorch-Latest-red?style=for-the-badge&logo=pytorch"/>
<img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit"/>
<img src="https://img.shields.io/badge/OpenCV-ComputerVision-green?style=for-the-badge&logo=opencv"/>

</p>

---

## 📌 Overview

Emotion AI Dashboard is an end-to-end Deep Learning application for **Facial Emotion Recognition (FER)** that detects human emotions from facial expressions in real time.

The project combines modern Computer Vision techniques with interactive dashboards to deliver an intuitive and professional AI experience.

Supported emotion classes include:

* 😊 Happy
* 😢 Sad
* 😐 Neutral

---

## ✨ Features

✅ Real-Time Emotion Prediction
✅ Deep Learning Inference Pipeline
✅ TensorFlow and PyTorch Model Support
✅ Interactive Analytics Dashboard
✅ Dataset Exploration Tools
✅ Beautiful Glassmorphism UI Design
✅ Performance Visualization using Plotly
✅ Streamlit Web Application

---

# 🏗️ System Architecture

```text
Input Image
     │
     ▼
Image Upload
     │
     ▼
Preprocessing
(Grayscale → Resize → Normalize)
     │
     ▼
CNN Model Inference
     │
     ▼
Prediction Layer
     │
     ▼
Emotion Classification
     │
     ▼
Dashboard Visualization
```

---

# 🔄 Inference Pipeline

1. User uploads facial image.
2. Image is converted to grayscale.
3. Image resized to `48x48`.
4. Pixel values normalized.
5. Processed image sent to CNN model.
6. Model predicts emotion probabilities.
7. Dashboard visualizes confidence scores and analytics.

---

# 🧠 Deep Learning Architecture

Example CNN Architecture:

```text
Input Layer (48x48x1)
        ↓
Conv2D + ReLU
        ↓
MaxPooling
        ↓
Conv2D + ReLU
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense Layer
        ↓
Dropout
        ↓
Softmax Output Layer
```

---

# 📂 Project Structure

```bash
Emotion-AI/
│
├── app.py
├── model/
│   ├── emotion_model.h5
│
├── notebooks/
│   └── FER_training.ipynb
│
├── assets/
│   ├── screenshots/
│
├── requirements.txt
├── README.md
│
└── dataset/
```

---

# 🛠️ Technologies Used

| Category             | Technologies               |
| -------------------- | -------------------------- |
| Programming Language | Python                     |
| Deep Learning        | TensorFlow, Keras, PyTorch |
| Computer Vision      | OpenCV                     |
| Frontend             | Streamlit                  |
| Visualization        | Plotly                     |
| Data Processing      | NumPy, Pandas              |
| Image Handling       | Pillow                     |

---

# 📊 Dashboard Modules

### 🏠 Home

Overview of project statistics and capabilities.

### 🎭 Predict Emotion

Upload image and perform emotion prediction.

### 📈 Model Analytics

Interactive charts for training metrics.

### 🗂 Dataset Explorer

Dataset visualization and class distribution analysis.

### ℹ About Project

Technical details and project information.

---

# 📷 Screenshots

## Home Page

![Home](assets/screenshots/home.png)

## Prediction Page

![Prediction](assets/screenshots/predict.png)

## Analytics Dashboard

![Analytics](assets/screenshots/analytics.png)

---

# 🚀 Live Demo

🔗 **Demo Link:**
[Add Streamlit Deployment URL Here]

---

# ⚙️ Installation

```bash
git clone https://github.com/yourusername/emotion-ai-dashboard.git

cd emotion-ai-dashboard

pip install -r requirements.txt

streamlit run app.py
```

---

# 📈 Future Improvements

* Support all FER2013 classes.
* Webcam real-time detection.
* Face localization using Haar Cascade.
* Model explainability with Grad-CAM.
* Docker deployment.
* Cloud inference API.

---

# 👩‍💻 Author

## Nouran Nasser

Computer Science Student | AI & Machine Learning Enthusiast

* LinkedIn: https://linkedin.com/in/your-linkedin
* GitHub: https://github.com/your-github
* Email: [your@email.com](mailto:your@email.com)

---

# ⭐ Support

If you like this project, don't forget to give it a ⭐ on GitHub.

---

<p align="center">
Made with ❤️ using Deep Learning and Computer Vision
</p>
