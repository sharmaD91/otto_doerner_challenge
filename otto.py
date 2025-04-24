import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor

if __name__ == "__main__":
    # check if combined_df.pkl exists
    try:
        df = pd.read_pickle('otto_files/combined_df.pkl')
    except FileNotFoundError:
        df21 = pd.read_csv('otto_files/MMX_Hackathon2025_year2021.CSV', sep=';', encoding='latin1')
        df22 = pd.read_excel('otto_files/MMX_Hackathon2025_year2022.xlsx', sheet_name='MMX_Hackathon_2025')
        df23 = pd.read_excel('otto_files/MMX_Hackathon2025_year2023.xlsx', sheet_name='MMX_Hackathon_2025')
        df24 = pd.read_excel('otto_files/MMX_Hackathon2025_year2024.xlsx', sheet_name='MMX_Hackathon_2025')
        df25 = pd.read_csv('otto_files/MMX_Hackathon2025_year2025Q1.CSV', sep=';', encoding='latin1')

        df23.rename(columns={'EntPlz': 'Plz'}, inplace=True)
        df23.rename(columns={'EntOrt': 'Ort'}, inplace=True)
        df24.rename(columns={'EntPlz': 'Plz'}, inplace=True)
        df24.rename(columns={'EntOrt': 'Ort'}, inplace=True)

        df = pd.concat([df21, df22, df23, df24, df25], ignore_index=True)

        # Saving the combined dataframe to a pickle file
        df.to_pickle('otto_files/combined_df.pkl')

    df['LiefDatum'] = pd.to_datetime(df['LiefDatum'], format='%d.%m.%Y', errors='coerce')

    total = df.groupby(['LiefDatum', 'DspZenKz'])['CSAnz'].sum().reset_index(name='total_containers')

    # Containers M by date and location
    container_M = df[df['DspGrpKz'] == 'M'].groupby(['LiefDatum', 'DspZenKz'])['CSAnz'].sum().reset_index(name='container_M')

    # Containers C by date and location
    container_C = df[df['DspGrpKz'] == 'C'].groupby(['LiefDatum', 'DspZenKz'])['CSAnz'].sum().reset_index(name='container_C')

    df_grouped = total.merge(container_M, on=['LiefDatum', 'DspZenKz'], how='left') \
                    .merge(container_C, on=['LiefDatum', 'DspZenKz'], how='left')

    df_grouped['container_M'] = df_grouped['container_M'].fillna(0)
    df_grouped['container_C'] = df_grouped['container_C'].fillna(0)

    df_grouped = df_grouped.rename(columns={
        'LiefDatum': 'date',
        'DspZenKz': 'location'
    })

    df_grouped['container_M'] = df_grouped['container_M'].astype(int)
    df_grouped['container_C'] = df_grouped['container_C'].astype(int)

    df_grouped = df_grouped[['date', 'location', 'container_M', 'container_C', 'total_containers']]

    df_grouped = df_grouped.sort_values('date')

    df_grouped['dayofweek'] = df_grouped['date'].dt.dayofweek
    df_grouped['day'] = df_grouped['date'].dt.day
    df_grouped['month'] = df_grouped['date'].dt.month
    df_grouped['year'] = df_grouped['date'].dt.year
    df_grouped['dayofyear'] = df_grouped['date'].dt.dayofyear
    df_grouped['weekofyear'] = df_grouped['date'].dt.isocalendar().week.astype(int)

    le = LabelEncoder() # To encode 'location' as a number
    df_grouped['location_encoded'] = le.fit_transform(df_grouped['location'])

    features = ['dayofweek', 'day', 'month', 'year', 'dayofyear', 'weekofyear', 'location_encoded']
    X = df_grouped[features]
    y = df_grouped[['container_M', 'container_C']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)

    model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

    def predict_next_n_days(start_date: str, location: str, days: int) -> pd.DataFrame:
        start_date = pd.to_datetime(start_date)
        future_dates = pd.date_range(start=start_date, periods=days)
        predictions = []
        location_encoded = le.transform([location])[0]

        for date in future_dates:
            input_features = pd.DataFrame({
                'dayofweek': [date.dayofweek],
                'day': [date.day],
                'month': [date.month],
                'year': [date.year],
                'dayofyear': [date.dayofyear],
                'weekofyear': [date.isocalendar().week],
                'location_encoded': [location_encoded]
            })
            pred = model.predict(input_features)[0]
            pred = np.where(pred < 0, 0, pred)
            predictions.append(tuple(map(int, pred)))
            
        return pd.DataFrame({'date': future_dates, 'container_M': [x[0] for x in predictions], 'container_C': [x[1] for x in predictions]})

    location = 'HH'  # Replace with the desired location
    start_date = '2025-05-01'  # Replace with the desired start date
    days = 7  # Number of days to predict

    predictions = predict_next_n_days(start_date, location, days)
    print(predictions)
