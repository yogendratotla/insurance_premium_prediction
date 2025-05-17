# preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    # Type conversions
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df['children'] = pd.to_numeric(df['children'], errors='coerce')
    df['charges'] = pd.to_numeric(df['charges'], errors='coerce')
    
    return df

def split_data(df):
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_test, X_live, y_test, y_live = train_test_split(X_temp, y_temp, test_size=0.3333, random_state=42)
    
    return X_train, X_test, X_live, y_train, y_test, y_live

def build_preprocessor():
    cat_cols = ['gender', 'smoker', 'region', 'medical_history', 
                'family_medical_history', 'exercise_frequency', 
                'occupation', 'coverage_level']
    num_cols = ['age', 'children', 'bmi']
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, cat_cols),
        ('num', num_pipeline, num_cols)
    ])

    return preprocessor
