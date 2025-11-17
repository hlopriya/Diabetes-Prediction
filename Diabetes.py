import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

# Load Dataset
dataset = pd.read_csv(r"C:\Users\Admin\Desktop\VS Code\Python_Programming\ML PBL\diabetes_prediction_dataset.csv")

# Encode categorical variables (if any)
for col in dataset.columns:
    if dataset[col].dtype == 'object':
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col])

# Features and Target
X = dataset.drop("diabetes", axis=1)
y = dataset["diabetes"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = GaussianNB()
model.fit(X_train, y_train) 

# GUI
def predict_diabetes():
    try:
        # Collect user input
        user_input = [float(entries[col].get()) for col in X.columns]
        # Scale input
        user_input_scaled = scaler.transform([user_input])
        # Predict
        prediction = model.predict(user_input_scaled)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        messagebox.showinfo("Prediction Result", f"The model predicts: {result}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values for all fields.")

# Create Tkinter window
root = tk.Tk()
root.title("Diabetes Prediction")
root.geometry("600x800")  
root.resizable(True, True)

# Create a scrollable frame
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)

canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

form_frame = tk.Frame(canvas)
canvas.create_window((0,0), window=form_frame, anchor="nw")

# Add input fields dynamically
entries = {}
for idx, col in enumerate(X.columns):
    tk.Label(form_frame, text=col, font=("Arial", 20)).grid(row=idx, column=0, padx=10, pady=5, sticky="w")
    entry = tk.Entry(form_frame, font=("Arial", 20))
    entry.grid(row=idx, column=1, padx=15, pady=10)
    entries[col] = entry

# Predict button
tk.Button(form_frame, text="Predict", font=("Arial", 20, "bold"), bg="#4CAF50", fg="white", command=predict_diabetes)\
    .grid(row=len(X.columns), column=0, columnspan=5, pady=20)

root.mainloop()