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

* Python 3.7+
* Modal account and CLI installed
* Docker (for building the Modal image)

### Installation

1. Clone the repository:

```bash
git clone [https://github.com/](https://github.com/)<your_username>/sales-prediction-api.git  # Replace with your repo URL
cd sales-prediction-api
