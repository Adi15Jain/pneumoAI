✅ README.md

# 🧠 PneumoAI: Pneumonia Detection from Chest X-Rays using CNN (PyTorch)

PneumoAI is a deep learning-based web application that detects pneumonia from chest X-ray images using a Convolutional Neural Network (CNN) built with PyTorch. The application is designed for medical diagnostic support and educational purposes.

## 🚀 Features

- CNN-based binary classification (Pneumonia vs. Normal)
- PyTorch-based model training and evaluation
- Drag-and-drop X-ray upload
- Real-time prediction with confidence
- Loading spinner, alerts, and dark mode
- Fully responsive React frontend
- REST API backend using FastAPI/Flask (you can adapt depending on what you used)

## 🧑‍💻 Tech Stack

- **Frontend**: React, Tailwind CSS
- **Backend**: FastAPI
- **Modeling**: PyTorch, NumPy, Matplotlib
- **Deployment**: Vercel (Frontend), Render/Any other platform (Backend)

---

## ⚙️ Getting Started

### 1. Clone the Repository

git clone https://github.com/Adi15Jain/pneumoAI.git
cd pneumoAI 2. Install Dependencies

pip install -r requirements.txt
Make sure you're using Python 3.8+

3. Download Dataset
   The dataset (Kaggle’s Chest X-Ray Images (Pneumonia)) is not included in this repository due to size constraints.

📥 To use it:
Download the dataset manually from Kaggle.

Extract and place it under the following directory:

PneumoAI/
└── chest_xray/
├── train/
├── val/
└── test/
Update paths in the notebook accordingly.

4. Train the Model
   Open the notebook:

cd model
jupyter notebook pneumonia_cnn.ipynb
Train the CNN model. The model will be saved as model.pth.

5. Add Trained Model to Backend
   After training:

Move the model.pth file to the backend directory.

Ensure the backend code correctly loads the model.

model.load_state_dict(torch.load("model.pth", map_location=device))

6. Run the Backend

cd backend
python app.py

7. Run the Frontend

cd frontend
npm install
npm run dev
Make sure the backend server is running and update API endpoint in your frontend if needed.
