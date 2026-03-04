import tkinter as tk
from tkinter import messagebox
import joblib
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

# Prediction function
def predict_profile(features):
    result = model.predict([features])[0]
    return "Fake Profile 🚨" if result == 1 else "Real Profile ✅"

# GUI Window
root = tk.Tk()
root.title("Fake Profile Detector")
root.geometry("400x500")

# Input fields
labels = [
    "Followers", "Following", "Posts", "Account Age (days)",
    "Is Verified (0 = No, 1 = Yes)", "Average Likes",
    "Bio Length", "Has Profile Picture (0 = No, 1 = Yes)"
]

entries = []
for label in labels:
    tk.Label(root, text=label).pack(pady=3)
    entry = tk.Entry(root)
    entry.pack()
    entries.append(entry)

# Button function
def check_profile():
    try:
        # Convert all inputs to numbers
        features = [float(entry.get()) for entry in entries]
        result = predict_profile(features)
        messagebox.showinfo("Result", f"The profile is: {result}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers (0 or 1 for yes/no fields).")

# Button
tk.Button(root, text="Check Profile", command=check_profile).pack(pady=20)

root.mainloop()
