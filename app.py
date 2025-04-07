import os
import pickle
import numpy as np
from flask import Flask, request, render_template

# Create Flask app (using 'app' variable name that Gunicorn expects)
app = Flask(__name__)

# Load the model with proper error handling and path resolution
try:
    model_path = os.path.join(os.path.dirname(__file__), "joy.pkl")
    model = pickle.load(open(model_path, "rb"))
except Exception as e:
    model = None
    print(f"Error loading model: {str(e)}")

@app.route("/")
def home():
    return render_template("index.html", prediction_text="")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", 
                             prediction_text="Error: Model not loaded properly")
    
    try:
        # Get form data and convert to float
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        
        # Make prediction
        prediction = model.predict(features)
        return render_template("index.html", 
                             prediction_text=f"The recommended crop is: {prediction[0]}")
    
    except Exception as e:
        return render_template("index.html", 
                             prediction_text=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
