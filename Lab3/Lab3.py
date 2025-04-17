import tkinter as tk
import pandas as pd

# Sample data for the DataFrame
data = {
    'symptom': ['Гарячка', 'Кашель', 'Головний біль'],
    'diagnosis': ['Грип', 'Застуда', 'Мігрень']
}
df = pd.DataFrame(data)

window = tk.Tk()
window.title("Symptom Checker")
window.geometry("400x300")
label = tk.Label(window, text="Введіть свій симптом:")
label.pack()
entry = tk.Entry(window)
entry.pack()

def process_query(symptom):
    # Check if the symptom exists in the DataFrame
    if symptom in df['symptom'].values:
        diagnosis = df[df['symptom'] == symptom]['diagnosis'].values[0]
    else:
        diagnosis = "Діагноз не знайдено."
    return diagnosis

def your_function():
    symptom = entry.get()
    result = process_query(symptom)
    result_label.config(text=f"Діагноз: {result}")

button = tk.Button(window, text="Отримати діагноз", command=your_function)
button.pack()
result_label = tk.Label(window, text="Тут буде відображено діагноз")
result_label.pack()
window.mainloop()