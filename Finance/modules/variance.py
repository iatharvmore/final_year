def variance(df):
    df['variance'] = df['expense'] - df['budget']
    return df
