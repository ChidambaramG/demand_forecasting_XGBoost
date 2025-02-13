# Sales Prediction API with Modal

This project implements a sales prediction API using Modal, a platform for running and deploying containerized applications.  It leverages machine learning with XGBoost to forecast sales based on historical data.

## Overview

The application trains an XGBoost regression model on sales data, saves the trained model and label encoders to a Modal volume, and exposes a web endpoint for making predictions.  The API accepts a start date, end date, store ID, and item ID as input and returns the total predicted sales for the given period, along with daily predicted sales.

## Features

* **Model Training:** Trains an XGBoost model on historical sales data, including feature engineering (year, month, day, day of week, weekend indicator).
* **Model Persistence:** Saves the trained model and label encoders to a Modal volume for easy access during prediction.
* **Prediction Endpoint:** Provides a web endpoint (`/predict_sales`) for making sales predictions.
* **Input Validation:** Validates input parameters to ensure required fields are present.
* **Error Handling:** Implements error handling to catch and return informative error messages.
* **Daily Predictions:** Returns daily sales predictions within the specified date range.
* **Total Sales Prediction:** Calculates and returns the total predicted sales for the entire period.

## Getting Started

### Prerequisites

* Python 3.10+
* Modal account and CLI installed

### Installation

1. Clone the repository:

```bash
git clone https://github.com/ChidambaramG/demand_forecasting_XGBoost.git  # Replace with your repo URL
cd demand_forecasting_XGBoost
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure Modal:

```bash
modal token new
```

4. Build and deploy the application:

```bash
modal deploy serve.py
```

5. Test the API:

```bash
curl --location '<your_modal_url>' \
--header 'Content-Type: application/json' \
--data '{
    "start_date": "2018-01-01",
    "end_date": "2018-01-8", 
    "store": "1", 
    "item": "9"  
}'
```
