import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class DataPreprocessor:
    def __init__(self, data_path, target_columns, columns_to_drop=None):
        self.data_path = data_path
        self.target_columns = target_columns
        self.columns_to_drop = columns_to_drop or []
        self.df = None
        self.features = None
        self.targets = None
        self.preprocessor = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            logging.info(f"Data loaded successfully from {self.data_path}")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self):
        self.df.drop(columns=self.columns_to_drop, inplace=True, errors='ignore')

        self.features = self.df.drop(columns=self.target_columns)
        self.targets = self.df[self.target_columns]

        numeric_features = self.features.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.features.select_dtypes(include=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        self.features = self.preprocessor.fit_transform(self.features)
        logging.info("Data preprocessing completed")

    def feature_engineering(self):
        # Add interaction terms
        if 'xG' in self.df.columns and 'Sh' in self.df.columns:
            self.df['xG_per_Sh'] = self.df['xG'] / self.df['Sh'].replace(0, 1)
        if 'onxGA' in self.df.columns and 'SoTA' in self.df.columns:
            self.df['xGA_per_SoTA'] = self.df['onxGA'] / self.df['SoTA'].replace(0, 1)

        # Create rolling averages
        rolling_columns = ['Gls', 'GA', 'xG', 'onxGA']
        for col in rolling_columns:
            if col in self.df.columns:
                self.df[f'{col}_rolling_5'] = self.df[col].rolling(window=5, min_periods=1).mean()
                self.df[f'{col}_rolling_10'] = self.df[col].rolling(window=10, min_periods=1).mean()

        self.features = self.df.drop(columns=self.target_columns)
        self.features = self.preprocessor.transform(self.features)

        logging.info("Feature engineering completed")

    def get_processed_data(self):
        return self.features, self.targets