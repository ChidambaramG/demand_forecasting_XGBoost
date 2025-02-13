import modal
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

app = modal.App("sales-prediction-api")

class predictionInput(BaseModel):
    start_date: str
    end_date: str
    store: str
    item: str

# Create a Modal volume to store the model and related files
volume = modal.Volume.from_name("sales-prediction-volume", create_if_missing=True)

# Create a Modal image with the necessary dependencies
image = (modal.Image.debian_slim().pip_install(
    "pandas",
    "scikit-learn",
    "xgboost",
    "joblib",
    "fastapi[standard]"
)
.add_local_dir("root/data", remote_path="/root/data"))

@app.function(
    image=image,
    volumes={"/root/model": volume}
)
def train_and_save_model():
    # Load the data
    df = pd.read_csv("/root/data/train.csv")

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Feature engineering
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # Label encode store and item
    le_store = LabelEncoder()
    le_item = LabelEncoder()
    df['store'] = le_store.fit_transform(df['store'])
    df['item'] = le_item.fit_transform(df['item'])

    # Prepare features and target
    features = ['store', 'item', 'year', 'month', 'day', 'dayofweek', 'is_weekend']
    X = df[features]
    y = df['sales']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Root Mean Squared Error: {rmse}")

    r2 = r2_score(y_test, y_pred)
    print(f"R-squared Score (Test Accuracy): {r2:.4f}")

    # Create both remote and local directories
    import os
    os.makedirs("/root/model", exist_ok=True)
    os.makedirs("model", exist_ok=True)  # Local directory

    # Save to Modal volume
    joblib.dump(model, "/root/model/xgboost_model.joblib")
    joblib.dump(le_store, "/root/model/le_store.joblib")
    joblib.dump(le_item, "/root/model/le_item.joblib")

    print("Model saved both to Modal volume and local directory")
    return "Training completed and models saved"


@app.function(image=image, volumes={"/root/model": volume})
@modal.web_endpoint(method="POST")
def predict_sales(data: predictionInput):
    # data = request.json
    start_date = data.start_date
    end_date = data.end_date
    store = data.store
    item = data.item

    if not all([start_date, end_date, store, item]):
        return {"error": "Missing required fields"}, 400

    try:
        model = joblib.load("/root/model/xgboost_model.joblib")
        le_store = joblib.load("/root/model/le_store.joblib")
        le_item = joblib.load("/root/model/le_item.joblib")

        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create DataFrame with all dates in range
        df = pd.DataFrame(date_range, columns=['date'])
        
        # Add store and item columns
        df['store'] = store
        df['item'] = item
        
        # Feature engineering
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

        # Encode store and item
        df['store'] = le_store.transform(df['store'])
        df['item'] = le_item.transform(df['item'])

        # Prepare features for prediction
        features = ['store', 'item', 'year', 'month', 'day', 'dayofweek', 'is_weekend']
        X = df[features]

        # Make predictions for all dates
        predictions = model.predict(X)
        total_sales = float(np.sum(predictions))
        daily_sales = predictions.tolist()
        
        return {
            "total_predicted_sales": total_sales,
            "daily_predictions": [
                {"date": date.strftime("%Y-%m-%d"), "sales": sales} 
                for date, sales in zip(date_range, daily_sales)
            ]
        }
        
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    # Train and save the model when the script is run directly
    with modal.App() as app:
        train_and_save_model.call()
