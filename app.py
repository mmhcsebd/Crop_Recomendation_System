import os
import pickle

import numpy as np
from flask import Flask, request, render_template




# Create flask app
flask_app = Flask(__name__)

# Load the model with error handling
try:
    model = pickle.load(open("joy.pkl", "rb"))
except (FileNotFoundError, IOError, pickle.UnpicklingError) as e:
    model = None
    print(f"Error loading the model file: {e}")


@flask_app.route("/")
def home():
    return render_template("index.html", prediction_text="")  # Blank prediction initially

@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text="The Predicted Crop is {}".format(prediction[0]))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    flask_app.run(host="0.0.0.0", port=port, debug=True)

