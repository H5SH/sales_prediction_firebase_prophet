from prophet import Prophet
import pandas as pd
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


cred = credentials.Certificate("C:/H5SH/personal/companies/Karbon/pharmacy-fyp/models/sales_prediction/firebase.json") 
firebase_admin.initialize_app(cred)
db = firestore.client()

# model = Prophet()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # List of allowed origins, e.g., ["https://example.com"]
    allow_credentials=True,  # Allow cookies and authentication
    allow_methods=["*"],  # HTTP methods to allow, e.g., ["GET", "POST"]
    allow_headers=["*"],  # HTTP headers to allow, e.g., ["Content-Type"]
)

@app.get("/")
def root():
    return {"message": "Firestore + Prophet Sales Prediction API"}

@app.get("/predict-sales")
async def predict_sales(collection: str = "sales_data", periods: int = 30):
    try:
        data = fetch_sales_data(collection)
        if(data.empty):
            return {"error": "sales data is empty"}
        
        processed_data = preprocess_data(data)
        forecast = forecast_sales(processed_data)

        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)

        return result.to_dict(orient="records")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        


def fetch_sales_data(collection: str):

    """data from Firestore."""

    sales_ref = db.collection(collection)
    docs = sales_ref.stream()

    sales_data = []
    for doc in docs:
        record = doc.to_dict()
        if "date" in record and "quantity_sold" in record:
            sales_data.append({
                "ds": datetime.strptime(record["date"], "%Y-%m-%d"),
                "y": record["quantity_sold"],
                "weather_condition": record.get("weather_condition", None),
                "temperature": record.get("temperature", None),
                "humidity": record.get("humidity", None),
                "is_promotion": 1 if record.get("is_promotion", False) else 0,
                "day_of_week": record.get("day_of_week", None),
            })

    return pd.DataFrame(sales_data)

def preprocess_data(sales_data):

    """data for Prophet."""

    weather_dummies = pd.get_dummies(sales_data["weather_condition"], prefix="weather")
    day_dummies = pd.get_dummies(sales_data["day_of_week"], prefix="day")

    # Combine dummy variables with sales data
    processed_data = pd.concat([sales_data, weather_dummies, day_dummies], axis=1)

    # Drop unused columns
    processed_data = processed_data.drop(columns=["weather_condition", "day_of_week"])

    return processed_data

def forecast_sales(sales_data):

    """Prophet to forecast"""

    model = Prophet()
    model.add_regressor("temperature")
    model.add_regressor("humidity")
    model.add_regressor("is_promotion")

    for column in sales_data.columns:
        if column.startswith("weather_") or column.startswith("day_"):
            model.add_regressor(column)

    # Fit the model
    model.fit(sales_data)

    # Predict for the next 30 days
    future = model.make_future_dataframe(periods=30)
    future["temperature"] = sales_data["temperature"].mean()  
    future["humidity"] = sales_data["humidity"].mean()
    future["is_promotion"] = 0 

    # Add dummy variables for future predictions
    for column in sales_data.columns:
        if column.startswith("weather_") or column.startswith("day_"):
            future[column] = 0 

    forecast = model.predict(future)

    return forecast

def main():

    print("Fetching sales data from Firestore...")
    sales_data = fetch_sales_data()

    if sales_data.empty:
        print("No sales data found in Firestore.")
        return

    print("Fetched sales data:")
    print(sales_data)

    print("Preprocessing sales data...")
    processed_data = preprocess_data(sales_data)

    print("Processed sales data:")
    print(processed_data)

    print("Forecasting sales...")
    forecast = forecast_sales(processed_data)

    print("Forecasted sales:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# if __name__ == "__main__":
#     main()