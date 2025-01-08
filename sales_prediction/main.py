from google.cloud import firestore
from prophet import Prophet
import pandas as pd
from datetime import datetime

# Initialize Firestore client
db = firestore.Client()

def fetch_sales_data():
    """Fetch sales data from Firestore."""
    sales_ref = db.collection("sales")
    docs = sales_ref.stream()

    # Convert Firestore documents to a DataFrame
    sales_data = []
    for doc in docs:
        record = doc.to_dict()
        if "date" in record and "sales" in record:
            sales_data.append({
                "ds": datetime.strptime(record["date"], "%Y-%m-%d"),
                "y": record["sales"]
            })

    return pd.DataFrame(sales_data)

def forecast_sales(sales_data):
    """Use Prophet to forecast future sales."""
    model = Prophet()
    model.fit(sales_data)

    # Predict for the next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    return forecast

def main():
    # Step 1: Fetch sales data from Firestore
    print("Fetching sales data from Firestore...")
    sales_data = fetch_sales_data()

    if sales_data.empty:
        print("No sales data found in Firestore.")
        return

    print("Fetched sales data:")
    print(sales_data)

    # Step 2: Forecast future sales using Prophet
    print("Forecasting sales...")
    forecast = forecast_sales(sales_data)

    # Step 3: Display forecasted results
    print("Forecasted sales:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

if __name__ == "__main__":
    main()
