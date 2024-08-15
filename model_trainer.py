from keras.src.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from tensorflow.keras import regularizers
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout

class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.feature_names = X.columns if hasattr(X, 'columns') else None

    def train_test_split(self, test_size=0.2):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=42)

    @staticmethod
    def train_xgboost(X_train, y_train):
        xgb_model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
        xgb_model.fit(X_train, y_train)
        return xgb_model

    @staticmethod
    def create_nn_model(input_dim, output_dim, learning_rate=0.001):
        model = keras.Sequential([
            Input(shape=(input_dim,)),
            Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            Dropout(0.3),
            Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            Dense(output_dim)
        ])

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=10000,
            decay_rate=0.9
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

        return model

    def train_neural_network(self, X_train, y_train):
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]

        nn_model = self.create_nn_model(input_dim, output_dim)

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        nn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

        return nn_model

    @staticmethod
    def train_random_forest(X_train, y_train):
        rf_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
        rf_model.fit(X_train, y_train)
        return rf_model

    @staticmethod
    def train_linear_regression(X_train, y_train):
        lr_model = MultiOutputRegressor(LinearRegression())
        lr_model.fit(X_train, y_train)
        return lr_model