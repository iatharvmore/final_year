def recommend(df):
    recs = []

    if df['expense'].mean() > df['budget'].mean():
        recs.append("Reduce overall spending.")

    if df['variance'].sum() > 0:
        recs.append("Reallocate budget.")

    if len(df[df['anomaly']=="Yes"]) > 0:
        recs.append("Investigate anomalies.")

    return recs
