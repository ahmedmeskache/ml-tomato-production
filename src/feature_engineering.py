def add_features(df):
    """
    Add new features to improve model performance.
    Example: average temperature between morning and evening.
    :param df: pandas DataFrame
    :return: pandas DataFrame with added features
    """
    df['avg_temp'] = (df['temp_morning'] + df['temp_evening']) / 2
    print("Feature 'avg_temp' added.")
    return df

if __name__ == "__main__":
    import pandas as pd
    sample_data = {
        'temp_morning': [22.5, 23.1],
        'temp_evening': [24.1, 25.0],
        'humidity': [70, 65],
        'ph': [6.5, 6.8],
        'water_flow': [30, 28],
        'fertilizer': [100, 110],
        'production_speed': [120, 115],
        'quality': [7.8, 8.2]
    }
    df = pd.DataFrame(sample_data)
    df = add_features(df)
    print(df)
