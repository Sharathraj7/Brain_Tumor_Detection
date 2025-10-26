# **Brain Tumor Detection using Deep Learning (Deployed on Hugging Face & Render)**

* This project detects and classifies brain tumors from MRI scans using a deep learning model.
* The model is hosted on Hugging Face, and the frontend web application is deployed on Render, providing a full end-to-end machine learning solution.

## **Overview**

The application allows users to upload an MRI brain scan and instantly receive a tumor detection result.
The backend model performs image classification and returns predictions through an API endpoint, while the frontend handles user interaction and visualization.

## **What I Learned**

* How to train and test a deep learning model for medical image classification.

* Hosting machine learning models on Hugging Face Spaces.

* Building and deploying a Flask-based or HTML/JS frontend on Render.

* Connecting frontend and backend using REST APIs.

* Managing dependencies and environment setup using requirements.txt and Procfile.

* Understanding deployment workflow and scalability concepts.

## **Tools & Libraries**

* Python, Flask, NumPy, OpenCV, PyTorch, YOLOv12

* Hugging Face Spaces (Model Hosting)

* Render (Frontend Deployment)

* HTML / CSS / JavaScript

## **Architecture**
```
User → Render Frontend → Hugging Face API → Deep Learning Model → Prediction → Render UI
```
## **How to Run Locally**

1. ### Clone this repository
```
git clone https://github.com/Sharathraj7/Brain_Tumor_Detection.git
cd Brain_Tumor_Detection
```

2. ### Install dependencies
```
pip install -r requirements.txt
```

3. ### Run the Flask app (if using locally)
```
python app.py
```

4. ### Open your browser at
```
http://127.0.0.1:5000
```
## **Demo**
Here’s a short demonstration of the Brain Tumor Detection web app in action:

![Recording 2025-10-26 105552 (1)](https://github.com/user-attachments/assets/e3954ed8-af86-448a-845b-64b11e11221a)

## **Files and Folders**

- **app.py** – Main backend file for API or local testing.

- **templates/** – HTML files for the web UI.

- **static/** – CSS, JS, and image assets.

- **trained_model/** – Pretrained model files (if included).

- **new_Brain_tumor.ipynb** – Jupyter notebook for model training.

- **requirements.txt** – Python dependencies.

- **Procfile** – Deployment configuration for Render.

## **Live Project**

* Frontend (Render): [Add your Render app URL here]

* Backend (Hugging Face): [Add your Hugging Face Space or API URL here]

## **Future Enhancements**

* Add explainability features (Grad-CAM heatmaps).

* Improve UI with real-time prediction visualization.

* Extend model to classify multiple tumor types.

## **Author**

Sharathraj B.

Machine Learning & AI Enthusiast | Computer Science Engineer

GitHub: [Sharathraj7](https://github.com/Sharathraj7)
