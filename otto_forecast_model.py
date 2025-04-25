import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import holidays
import re

class OttoForecaster:
    def __init__(self, model_type="xgboost"):
        self.model_type = model_type
        self.model = None
        self.label_encoder = None
        self.all_containers = None
        self.difference_flag = False

    def set_model_type(self, model_type: str):
        if model_type not in ["xgboost", "random_forest"]:
            raise ValueError("Invalid model type. Choose 'random_forest' or 'xgboost'.")
        self.model_type = model_type

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
            df['difference'] = df['CHAnz'] - df['CSAnz']
            df.to_pickle('otto_files/combined_df.pkl')
            return df

    def train_model(self, df, containers):
        german_holidays = holidays.Germany()

        df['ConTyp'] = [contyp[:4] if re.match(r'[a-zA-Z]', contyp[1]) else contyp[:3] for contyp in df['ConTyp']]
        self.all_containers = list(dict.fromkeys([contyp for contyp in df['ConTyp'] if '-' not in contyp]))
        
        if containers=="pick":
            data_selection = 'CSAnz'
        elif containers=="put":
            data_selection = 'CHAnz'
        elif containers=="difference":
            data_selection = 'difference'
            self.difference_flag = True
        else:
            raise ValueError("Invalid container type. Choose 'pick', 'put', or 'difference'.")
        total = df.groupby(['LiefDatum', 'DspZenKz'])[data_selection].sum().reset_index(name='total_containers')
        containers = np.empty(len(self.all_containers), dtype=object)
        for cont in range(len(self.all_containers)):
            containers[cont] = df[df['ConTyp'] == self.all_containers[cont]].groupby(['LiefDatum', 'DspZenKz'])[data_selection].sum().reset_index(name=self.all_containers[cont])
            total = total.merge(containers[cont], on=['LiefDatum', 'DspZenKz'], how='left')

        container_M = df[df['DspGrpKz'] == 'M'].groupby(['LiefDatum', 'DspZenKz'])[data_selection].sum().reset_index(name='container_M')
        container_C = df[df['DspGrpKz'] == 'C'].groupby(['LiefDatum', 'DspZenKz'])[data_selection].sum().reset_index(name='container_C')

        df_grouped = total.merge(container_M, on=['LiefDatum', 'DspZenKz'], how='left') \
                             .merge(container_C, on=['LiefDatum', 'DspZenKz'], how='left')

        df_grouped.fillna(0, inplace=True)
        df_grouped.rename(columns={'LiefDatum': 'date', 'DspZenKz': 'location'}, inplace=True)
        df_grouped['container_M'] = df_grouped['container_M'].astype(int)
        df_grouped['container_C'] = df_grouped['container_C'].astype(int)
        for cont in self.all_containers:
            df_grouped[cont] = df_grouped[cont].astype(int)

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
        y = df_grouped[self.all_containers + ['container_M', 'container_C']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)

        if self.model_type == "random_forest":
            self.model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=-1))
        elif self.model_type == "xgboost":
            self.model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1,n_jobs=4))
        else:
            raise ValueError("Invalid model type. Choose 'random_forest' or 'xgboost'.")
        self.model.fit(X_train, y_train)

    def predict(self, start_date: str, location: str, days: int = 7) -> pd.DataFrame:
        start_date = pd.to_datetime(start_date)
        future_dates = pd.date_range(start=start_date, periods=days)
        location_encoded = self.label_encoder.transform([location])[0]
        german_holidays = holidays.Germany()

        predictions = []
        for date in future_dates:
            if date in german_holidays:
                zero_pred = [0] * (len(self.all_containers)+2)
                predictions.append((date, *zero_pred))
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
            if not self.difference_flag:
                pred = [max(0, int(p)) for p in pred]
            predictions.append((date, *pred))

        return pd.DataFrame(predictions, columns=['date'] + self.all_containers + ['container_M', 'container_C'])
