import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import pandas as pd
import numpy as np

# Load the trained pipeline model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Load the car data from CSV
car = pd.read_csv('Cleaned_Car_data.csv')

# Dynamically extract values from the car data
companies = sorted(car['company'].unique())
car_models = sorted(car['name'].unique())
years = sorted(car['year'].unique(), reverse=True)
fuel_types = car['fuel_type'].unique().tolist()
# print(fuel_types)
# print(companies)
# Tkinter GUI setup
root = tk.Tk()
root.title("Car Price Predictor")
root.geometry("500x500")
root.configure(bg="#75767B")

# Title
title = tk.Label(root, text="Welcome to Car Price Predictor", font=("Helvetica", 16, "bold"), bg="#75767B", fg="white")
title.pack(pady=20)

# Form Frame
form_frame = tk.Frame(root, bg="#75767B")
form_frame.pack()

# Company
tk.Label(form_frame, text="Select the company:", bg="#75767B", fg="white").grid(row=0, column=0, sticky="w")
company_cb = ttk.Combobox(form_frame, values=companies, state="readonly")
company_cb.grid(row=0, column=1, pady=5)
company_cb.current(0)

# Model
tk.Label(form_frame, text="Select the model:", bg="#75767B", fg="white").grid(row=1, column=0, sticky="w")
model_cb = ttk.Combobox(form_frame, values=[], state="readonly")
model_cb.grid(row=1, column=1, pady=5)

# Updated models based on selected company
def update_models(event=None):
    selected = company_cb.get()
    # Filtered car models for the selected company
    available_models = sorted(car[car['company'] == selected]['name'].unique())
    model_cb['values'] = available_models
    if available_models:
        model_cb.current(0)

company_cb.bind("<<ComboboxSelected>>", update_models)

# Initialized models 
update_models()

# Year
tk.Label(form_frame, text="Select year of purchase:", bg="#75767B", fg="white").grid(row=2, column=0, sticky="w")
year_cb = ttk.Combobox(form_frame, values=years, state="readonly")
year_cb.grid(row=2, column=1, pady=5)
year_cb.current(0)

# Fuel Type
tk.Label(form_frame, text="Select fuel type:", bg="#75767B", fg="white").grid(row=3, column=0, sticky="w")
fuel_cb = ttk.Combobox(form_frame, values=fuel_types, state="readonly")
fuel_cb.grid(row=3, column=1, pady=5)
fuel_cb.current(0)

# Kilometres Driven
tk.Label(form_frame, text="Kilometres Driven:", bg="#75767B", fg="white").grid(row=4, column=0, sticky="w")
km_entry = tk.Entry(form_frame)
km_entry.grid(row=4, column=1, pady=5)

# Prediction Label
prediction_label = tk.Label(root, text="", font=("Helvetica", 14), bg="#75767B", fg="white")
prediction_label.pack(pady=20)

# Predict Button
def on_predict():
    try:
        km = int(km_entry.get())  # Ensure the kilometers driven is a valid number
    except ValueError:
        messagebox.showerror("Invalid Input", "Kilometres driven must be a number.")
        return

    # Prepare input data for prediction
    car_model = model_cb.get()
    company = company_cb.get()
    year = int(year_cb.get())
    fuel_type = fuel_cb.get()

    # Format the input to match model expectations
    input_df = pd.DataFrame(
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
        data=np.array([car_model, company, year, km, fuel_type]).reshape(1, 5)
    )

    # Making prediction 
    try:
        # Check 
        prediction = model.predict(input_df)
        prediction_label.config(text=f"Predicted Price: ₹ {np.round(prediction[0], 2):,}")
    except KeyError as e:
        messagebox.showerror("Error", f"Unknown field selected: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")

predict_btn = tk.Button(root, text="Predict Price", command=on_predict, bg="blue", fg="white", font=("Helvetica", 12, "bold"))
predict_btn.pack(pady=10)

root.mainloop()
