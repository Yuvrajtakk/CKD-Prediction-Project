
from flask import Flask, render_template, request
import numpy as np
import pandas as pd  # <-- Import pandas
import pickle

app = Flask(__name__)
model = pickle.load(open('../CKD.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = [
        'red_blood_cells', 'pus_cell', 'blood_glucose_random', 'blood_urea', 
        'pedal_edema', 'anemia', 'diabetes_mellitus', 'coronary_artery_disease'
    ]
    
    # Create a DataFrame with the feature names
    df_to_predict = pd.DataFrame(features_value, columns=features_name)
    
    # Make a prediction using the DataFrame
    output = model.predict(df_to_predict)

    if output[0] == 1:
        prediction_text = "Great! You DON'T have Chronic Kidney Disease."
    else:
        prediction_text = "Oops! You may have Chronic Kidney Disease. Please consult a doctor."
        
    return render_template('result.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)