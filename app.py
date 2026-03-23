from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

pipeline = pickle.load(open("pipeline.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # No float conversion
        input_features = [x for x in request.form.values()]

        # ✅ MATCH dataset column names
        features_name = [
            'rbc', 'pc', 'bgr', 'bu', 
            'pe', 'ane', 'dm', 'cad'
        ]

        df_to_predict = pd.DataFrame([input_features], columns=features_name)

        output = pipeline.predict(df_to_predict)

        if output[0] == 0:
            prediction_text = "Great! You DON'T have Chronic Kidney Disease."
        else:
            prediction_text = "Oops! You may have Chronic Kidney Disease. Please consult a doctor."

        return render_template('result.html', prediction_text=prediction_text)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)