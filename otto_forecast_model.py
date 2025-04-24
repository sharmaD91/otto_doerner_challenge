import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import holidays

class OttoForecaster:
    def __init__(self):
        self.model = None
        self.label_encoder = None

    def load_and_prepare_data(self, files):
        try:
            df = pd.read_pickle('otto_files/combined_df.pkl')
            return df
        except FileNotFoundError:
            dfs = []
            for f in files:
                if f.endswith('.csv'):
                    df = pd.read_csv(f, delimiter=';', encoding='latin1')
                elif f.endswith('.xlsx'):
                    df = pd.read_excel(f, sheet_name='MMX_Hackathon_2025')
                else:
                    continue

                if 'EntPlz' in df.columns:
                    df.rename(columns={'EntPlz': 'Plz', 'EntOrt': 'Ort'}, inplace=True)
                dfs.append(df)

            df = pd.concat(dfs, ignore_index=True)
            df['LiefDatum'] = pd.to_datetime(df['LiefDatum'], format='%d.%m.%Y', errors='coerce')
            df.to_pickle('otto_files/combined_df.pkl')
            return df

    def train_model(self, df):
        german_holidays = holidays.Germany()

        total = df.groupby(['LiefDatum', 'DspZenKz'])['CSAnz'].sum().reset_index(name='total_containers')
        container_M = df[df['DspGrpKz'] == 'M'].groupby(['LiefDatum', 'DspZenKz'])['CSAnz'].sum().reset_index(name='container_M')
        container_C = df[df['DspGrpKz'] == 'C'].groupby(['LiefDatum', 'DspZenKz'])['CSAnz'].sum().reset_index(name='container_C')

        df_grouped = total.merge(container_M, on=['LiefDatum', 'DspZenKz'], how='left') \
                             .merge(container_C, on=['LiefDatum', 'DspZenKz'], how='left')

        df_grouped.fillna(0, inplace=True)
        df_grouped.rename(columns={'LiefDatum': 'date', 'DspZenKz': 'location'}, inplace=True)
        df_grouped['container_M'] = df_grouped['container_M'].astype(int)
        df_grouped['container_C'] = df_grouped['container_C'].astype(int)

        df_grouped['dayofweek'] = df_grouped['date'].dt.dayofweek
        df_grouped['day'] = df_grouped['date'].dt.day
        df_grouped['month'] = df_grouped['date'].dt.month
        df_grouped['year'] = df_grouped['date'].dt.year
        df_grouped['dayofyear'] = df_grouped['date'].dt.dayofyear
        df_grouped['weekofyear'] = df_grouped['date'].dt.isocalendar().week.astype(int)
        df_grouped['is_holiday'] = df_grouped['date'].isin(german_holidays).astype(int)

        # Exclude holidays from training
        df_grouped = df_grouped[df_grouped['is_holiday'] == 0]

        self.label_encoder = LabelEncoder()
        df_grouped['location_encoded'] = self.label_encoder.fit_transform(df_grouped['location'])

        X = df_grouped[['dayofweek', 'day', 'month', 'year', 'dayofyear', 'weekofyear', 'location_encoded']]
        y = df_grouped[['container_M', 'container_C']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)

        self.model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1))
        self.model.fit(X_train, y_train)

    def predict(self, start_date: str, location: str, days: int = 7) -> pd.DataFrame:
        start_date = pd.to_datetime(start_date)
        future_dates = pd.date_range(start=start_date, periods=days)
        location_encoded = self.label_encoder.transform([location])[0]
        german_holidays = holidays.Germany()

        predictions = []
        for date in future_dates:
            if date in german_holidays:
                predictions.append((date, 0, 0))
                continue

            features = pd.DataFrame({
                'dayofweek': [date.dayofweek],
                'day': [date.day],
                'month': [date.month],
                'year': [date.year],
                'dayofyear': [date.dayofyear],
                'weekofyear': [date.isocalendar().week],
                'location_encoded': [location_encoded]
            })
            pred = self.model.predict(features)[0]
            predictions.append((date, max(0, int(pred[0])), max(0, int(pred[1]))))

        return pd.DataFrame(predictions, columns=['date', 'container_M', 'container_C'])
