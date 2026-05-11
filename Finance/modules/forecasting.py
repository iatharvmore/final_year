import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def forecast(df):
    if df.empty or 'date' not in df.columns or 'expense' not in df.columns or 'budget' not in df.columns:
        return pd.DataFrame()
        
    df = df.copy().sort_values('date')
    df['day'] = np.arange(len(df))

    X = df[['day']]
    y_exp = df['expense']
    y_bud = df['budget']

    model_exp = LinearRegression()
    model_exp.fit(X, y_exp)
    
    model_bud = LinearRegression()
    model_bud.fit(X, y_bud)

    last_date = df['date'].iloc[-1]
    future_days = np.array([[len(df)+i] for i in range(1, 6)]) # next 5 days
    pred_exp = model_exp.predict(future_days)
    pred_bud = model_bud.predict(future_days)
    
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 6)]
    
    forecast_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
        'Projected Expense': np.round(pred_exp, 2),
        'Projected Budget': np.round(pred_bud, 2)
    })
    
    # Add a practical risk indicator
    forecast_df['Status'] = np.where(
        forecast_df['Projected Expense'] > forecast_df['Projected Budget'],
        '🔴 At Risk (Over Budget)',
        '🟢 Safe'
    )
    
    return forecast_df
