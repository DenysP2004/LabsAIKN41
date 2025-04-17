import tkinter as tk
import pandas as pd

# Expanded data for the DataFrame with more symptoms and diagnoses
data = {
    'symptom': ['fever', 'cough', 'headache', 'runny nose', 'fatigue', 'sore throat', 'chills', 'body ache'],
    'diagnosis': ['Flu', 'Common Cold', 'Migraine', 'Common Cold', 'Flu', 'Common Cold', 'Flu', 'Flu']
}
df = pd.DataFrame(data)

window = tk.Tk()
window.title("Symptom Checker")
window.geometry("400x300")
label = tk.Label(window, text="Enter your symptoms (comma separated):")
label.pack()
entry = tk.Entry(window)
entry.pack()

def process_query(symptoms):
    symptoms_list = [symptom.strip() for symptom in symptoms.split(',')]
    possible_diagnoses = df[df['symptom'].isin(symptoms_list)]['diagnosis'].unique()
    
    if len(possible_diagnoses) > 0:
        diagnosis = ', '.join(possible_diagnoses)
    else:
        diagnosis = "Sorry, I don't have a diagnosis for those symptoms."
    return diagnosis

def your_function():
    symptoms = entry.get()
    result = process_query(symptoms)
    result_label.config(text=f"Possible Diagnoses: {result}")

button = tk.Button(window, text="Get Diagnosis", command=your_function)
button.pack()
result_label = tk.Label(window, text="Diagnosis will be displayed here")
result_label.pack()
window.mainloop()