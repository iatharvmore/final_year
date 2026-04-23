def detect_anomaly(df):
    mean = df['expense'].mean()
    std = df['expense'].std()

    df['anomaly'] = df['expense'].apply(
        lambda x: "Yes" if abs(x - mean) > 2*std else "No"
    )
    return df
