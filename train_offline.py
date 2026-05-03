import pandas as pd
import joblib
import json
import os
from engine import load_file, engineer_features, preprocess, train_model

def build_offline_model():
    print("Loading historical dataset...")
    df = pd.read_csv("sample_data/sample_transactions.csv")
    
    mapping = {
        'amount': 'amount',
        'time': 'date',
        'vendor': 'vendor',
        'location': 'location',
        'account_id': 'account_id',
        'label': None
    }
    
    print("Engineering features...")
    feature_df, feature_names, stats, original_df = engineer_features(df, mapping)
    
    # We do NOT split data for training the unsupervised model
    print("Scaling features...")
    (X_train_scaled, _,
     _, _,
     _, _,
     scaler, _) = preprocess(feature_df, df, mapping, allow_split=False)
     
    print("Training Isolation Forest ('auto' contamination)...")
    model, train_time = train_model(X_train_scaled, contamination='auto')
    
    print(f"Model trained in {train_time:.2f} seconds.")
    
    os.makedirs("models", exist_ok=True)
    
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    
    metadata = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "training_samples": len(df)
    }
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Saved model artifacts to 'models/' directory.")
    print("Offline training complete!")

if __name__ == "__main__":
    build_offline_model()
