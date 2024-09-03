from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pandas as pd
from src.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

class PredictionPipeline:
    def __init__(self, model_path, preprocessor_path):
        # Load the pre-trained model and preprocessor
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, input_data):
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        # Transform the data using the preprocessor
        processed_data = self.preprocessor.transform(input_df)
        # Make predictions using the model
        predictions = self.model.predict(processed_data)
        return predictions.tolist()
    
# Initialize the prediction pipeline with model and preprocessor paths
prediction_pipeline = PredictionPipeline(
    model_path='artifacts/XGBRegressor.pkl',
    preprocessor_path='artifacts/preprocessor.pkl'
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_prediction():
    input_data = {
        'Car Name': request.form.get('Car Name'),
        'Year': float(request.form.get('Year')),
        'Distance': int(request.form.get('Distance')),
        'Owner': int(request.form.get('Owner')),
        'Fuel': request.form.get('Fuel'),
        'Drive': request.form.get('Drive'),
        'Type': request.form.get('Type')
    }

    # Make prediction
    predictions = prediction_pipeline.predict(input_data)
    
    return render_template('results.html', predictions=predictions)
    
if __name__ == '__main__':
    app.run(debug=True)
