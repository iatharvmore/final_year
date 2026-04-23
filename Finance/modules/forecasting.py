from sklearn.linear_model import LinearRegression
import numpy as np

def forecast(df):
    df = df.sort_values('date')
    df['day'] = np.arange(len(df))

    X = df[['day']]
    y = df['expense']

    model = LinearRegression()
    model.fit(X, y)

    future = np.array([[len(df)+i] for i in range(5)])
    return model.predict(future)
