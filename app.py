from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

# Initialize app
app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "pipeline.pkl")

pipeline = pickle.load(open(model_path, "rb"))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        input_features = [x for x in request.form.values()]

        # Match dataset column names
        features_name = [
            'rbc', 'pc', 'bgr', 'bu',
            'pe', 'ane', 'dm', 'cad'
        ]

        # Convert to DataFrame
        df_to_predict = pd.DataFrame([input_features], columns=features_name)

        # Prediction
        output = pipeline.predict(df_to_predict)

        # Result message
        if output[0] == 0:
            prediction_text = "Great! You DON'T have Chronic Kidney Disease."
        else:
            prediction_text = "Oops! You may have Chronic Kidney Disease. Please consult a doctor."

        return render_template('result.html', prediction_text=prediction_text)

    except Exception as e:
        return f"Error: {str(e)}"



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)