# batch_predict.py
import pandas as pd
import joblib

def predict_from_csv(input_file, output_file):
    model = joblib.load('models/model_pipeline.pkl')
    df = pd.read_csv(input_file)
    
    predictions = model.predict(df)
    df['predicted_charges'] = predictions
    
    df.to_csv(output_file, index=False)
    print(f"✅ Predictions saved to {output_file}")

# Example usage:
predict_from_csv('input_data/new_insurance_data.csv', 'output/predicted_output.csv')
