import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek

    drop_cols = ['Invoice ID', 'Customer type',
                 'Time', 'Payment',
                 'gross margin percentage']
    df.drop(columns=[c for c in drop_cols
                     if c in df.columns], inplace=True)

    le = LabelEncoder()
    for col in ['Gender', 'Product line']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    return df

def predict_sales(df, product, start_date, end_date):
    product_df = df[df['Product line'] == product].copy()

    product_df['Date'] = pd.to_datetime(product_df['Date'])
    monthly = product_df.groupby(
        product_df['Date'].dt.to_period('M')
    )['Quantity'].sum()

    if len(monthly) >= 3:
        trend = monthly.rolling(3).mean().dropna().iloc[-1]
    else:
        trend = monthly.mean()

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    days = (end - start).days + 1

    daily_avg = float(trend) / 30
    predicted_total = round(daily_avg * days)

    return max(1, int(predicted_total))
