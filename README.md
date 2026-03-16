# Fake Profile Identification using Machine Learning

## 📌 Project Overview

Fake profiles on social media platforms can spread misinformation, perform scams, and manipulate online discussions.
This project aims to detect **fake social media profiles** using **Machine Learning techniques** and provides a **Tkinter-based desktop application** for user interaction.

The system analyzes profile features such as **followers, following, and number of posts** and predicts whether a profile is **Fake or Genuine**.

---

## 🚀 Features

* Detects fake social media profiles using a trained ML model
* Simple **GUI application built with Tkinter**
* Loads trained model using **Joblib**
* Allows users to input profile data and get prediction instantly
* Option to **retrain the model using new dataset**
* Lightweight and easy to run locally

---

## 🛠️ Technologies Used

* **Python**
* **Machine Learning (Scikit-learn)**
* **Tkinter (GUI)**
* **Pandas & NumPy**
* **Joblib (Model Saving/Loading)**

---

## 📂 Project Structure

```
Fake-social-profile-detection/
│
├── ml/
│   ├── model.pkl                # Trained ML model
│
├── dataset/
│   ├── profiles.csv             # Training dataset
│
├── app.py                       # Tkinter GUI application
├── train_model.py               # Script to train ML model
├── requirements.txt             # Required libraries
└── README.md                    # Project documentation
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/fake-profile-identification.git
cd fake-profile-identification
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
python app.py
```

---

## 🧠 Machine Learning Model

The project uses **Random Forest Classifier** to classify profiles based on extracted features.

### Features Used

* Followers count
* Following count
* Number of posts

### Model Training

The model is trained using the dataset and saved as:

```
model.pkl
```

This file is loaded into the GUI application for making predictions.

---

## 🖥️ Application Workflow

1. User enters profile details in the GUI.
2. The application sends input data to the trained ML model.
3. The model predicts whether the profile is **Fake or Genuine**.
4. The result is displayed as a popup message.

---

## 🔄 Model Retraining

The application also allows retraining the model with a new dataset (`profiles.csv`).
This helps improve accuracy when more data becomes available.

---

## 📸 Example Output

Input:

```
Followers: 50
Following: 1000
Posts: 1
```

Output:

```
⚠️ Fake Profile Detected
```

---

## 📌 Future Improvements

* Add **NLP analysis of profile bio**
* Detect fake accounts using **profile images**
* Build **web-based application using Flask or Streamlit**
* Improve model accuracy with larger datasets

---

## 👩‍💻 Author

**Saraswati Sonale**

---

## 📜 License

This project is for **educational and research purposes**.
