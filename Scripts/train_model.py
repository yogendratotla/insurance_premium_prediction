# train_model.py
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error, r2_score
from preprocess import load_and_prepare_data, split_data, build_preprocessor


# Load & preprocess
df = load_and_prepare_data('data\Insurance_Prediction.csv')
X_train, X_test, X_live, y_train, y_test, y_live = split_data(df)
preprocessor = build_preprocessor()

# Build model pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train
model_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = model_pipeline.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_score = np.mean(cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='r2'))

print("\n📊 Test Evaluation:")
print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ R² Score: {r2:.4f}")
print(f"✅ CV Score: {cv_score:.4f}")

# os.makedirs('../models', exist_ok=True)

# Save model  
joblib.dump(model_pipeline, 'models/model_pipeline.pkl')

print("\n✅ Model saved to 'models\model_pipeline.pkl'")
