def recommend(df):
    recs = []

    if df['expense'].mean() > df['budget'].mean():
        recs.append("Reduce overall spending to align with the average budget.")

    if df['variance'].sum() > 0:
        recs.append("Reallocate budget to cover positive variance areas.")

    if len(df[df['anomaly']=="Yes"]) > 0:
        recs.append("Investigate dates flagged as anomalies to understand irregular spending.")

    if len(recs) == 0:
        recs.append("Finances are stable and tracking well. Maintain current budgeting strategies.")

    return recs
