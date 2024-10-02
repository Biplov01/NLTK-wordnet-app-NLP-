import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load the trained Multinomial Naive Bayes model and the CountVectorizer
model_path = r'D:\data science projects\naive_bayes_model.pkl'
vectorizer_path = r'D:\data science projects\count_vectorizer.pkl'  # Update this path if you saved the vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Function to predict sentiment
def predict_sentiment():
    # Get the input from the text field
    input_text = text_input.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showwarning("Input Error", "Please enter some text.")
        return
    
    # Transform the input text using the loaded vectorizer
    input_vector = vectorizer.transform([input_text])
    
    # Make prediction using the loaded model
    prediction = model.predict(input_vector)
    
    # Display the prediction result
    sentiment = "Positive" if prediction[0] == 1 else "Negative"  # Assuming 1 is positive and 0 is negative
    messagebox.showinfo("Prediction Result", f"The sentiment is: {sentiment}")

# Create the main application window
root = tk.Tk()
root.title("Sentiment Analysis App")

# Create and place the input text field
text_input = tk.Text(root, height=10, width=50)
text_input.pack(pady=10)

# Create and place the predict button
predict_button = tk.Button(root, text="Predict Sentiment", command=predict_sentiment)
predict_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
