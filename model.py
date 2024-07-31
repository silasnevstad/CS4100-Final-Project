import os
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout
import matplotlib.pyplot as plt
import joblib


class SportsPredictionModel:
    def __init__(self, data_path, target_columns, columns_to_drop=None):
        self.data_path = data_path
        self.target_columns = target_columns
        self.columns_to_drop = columns_to_drop or []
        self.df = pd.read_csv(data_path)
        self.targets = {}
        self.features = None
        self.scaler = None
        self.preprocess_data()

    def preprocess_data(self):
        # Handle missing values
        self.df.fillna(0, inplace=True)

        # Drop unnecessary columns
        columns_to_drop = [col for col in self.columns_to_drop if col in self.df.columns]
        self.df.drop(columns=columns_to_drop, inplace=True)

        # Separate features and target
        self.features = self.df.drop(columns=[col for col in self.target_columns if col in self.df.columns])
        for target in self.target_columns:
            if target in self.df.columns:
                self.targets[target] = self.df[target]

        # Normalize features
        self.scaler = StandardScaler()
        self.features = pd.DataFrame(self.scaler.fit_transform(self.features), columns=self.features.columns)

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

        # Update features and preprocess again
        self.features = self.df.drop(columns=self.target_columns)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def train_test_split(self, test_size=0.2):
        splits = {}
        for target in self.target_columns:
            X_train, X_test, y_train, y_test = train_test_split(self.features, self.targets[target],
                                                                test_size=test_size, random_state=42)
            splits[target] = (X_train, X_test, y_train, y_test)
        return splits

    @staticmethod
    def train_xgboost(X_train, y_train):
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
        xgb_model.fit(X_train, y_train)
        return xgb_model

    @staticmethod
    def create_nn_model(input_dim, learning_rate=0.001):
        model = keras.Sequential([
            Input(shape=(input_dim,)),
            Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            Dropout(0.3),
            Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            Dense(1)
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

        nn_model = self.create_nn_model(input_dim)

        nn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

        return nn_model

    @staticmethod
    def evaluate_model(model, X_test, y_test, model_type='xgboost'):
        if model_type == 'xgboost':
            y_pred = model.predict(X_test)
        elif model_type == 'neural_network':
            y_pred = model.predict(X_test).flatten()

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'Mean Squared Error: {mse}')
        print(f'R^2 Score: {r2}')

        # Plot predictions vs actual values
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{model_type.capitalize()} Model: Actual vs Predicted")
        plt.show()

    @staticmethod
    def save_model(model, filename):
        # create a directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        file_path = 'models/' + filename
        if 'xgboost' in filename:
            joblib.dump(model, file_path)
        elif 'nn_model' in filename:
            model.save(file_path)

    @staticmethod
    def load_model(filename):
        file_path = 'models/' + filename
        if 'xgboost' in filename:
            return joblib.load(file_path)
        elif 'nn_model' in filename:
            return keras.models.load_model(file_path)

    def align_features(self, team_features):
        # Get all columns from the original dataset
        all_columns = self.df.drop(columns=[col for col in self.columns_to_drop + self.target_columns if col in self.df.columns]).columns

        # Add missing columns with zeros
        for col in all_columns:
            if col not in team_features:
                team_features[col] = 0

        # Remove extra columns
        team_features = team_features[all_columns]

        return team_features

    def predict_game_stats_xg(self, team1_features, team2_features):
        team1_features = pd.Series(team1_features)
        team2_features = pd.Series(team2_features)
        team1_features = self.align_features(team1_features)
        team2_features = self.align_features(team2_features)

        team1_features = self.scaler.transform([team1_features])
        team2_features = self.scaler.transform([team2_features])

        xg_model = self.load_model('xgboost_model_Gls.pkl')
        xga_model = self.load_model('xgboost_model_GA.pkl')

        team1_xG = xg_model.predict(team1_features)[0]
        team1_xGA = xga_model.predict(team1_features)[0]
        team2_xG = xg_model.predict(team2_features)[0]
        team2_xGA = xga_model.predict(team2_features)[0]

        return team1_xG, team1_xGA, team2_xG, team2_xGA

    @staticmethod
    def calculate_poisson_scoreline_prob(team1_xG, team2_xG, show_plot=False):
        max_goals = 5  # Maximum number of goals to consider for each team
        scoreline_probs = np.zeros((max_goals + 1, max_goals + 1))

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p1 = poisson.pmf(i, team1_xG)
                p2 = poisson.pmf(j, team2_xG)
                scoreline_probs[i, j] = p1 * p2

        # Increase probability of 0-0 and 1-1 scorelines
        scoreline_probs[0, 0] *= 1.1
        scoreline_probs[1, 1] *= 1.1
        scoreline_probs /= scoreline_probs.sum()  # Renormalize

        if show_plot:
            plt.imshow(scoreline_probs, cmap='coolwarm')
            plt.xlabel('Team 2 Goals')
            plt.ylabel('Team 1 Goals')
            plt.title('Scoreline Probabilities')
            plt.colorbar()
            plt.show()

        return scoreline_probs

    def predict_game_outcome(self, team1_features, team2_features, show_plot=False):
        team1_xG, team1_xGA, team2_xG, team2_xGA = self.predict_game_stats_xg(team1_features, team2_features)

        # Adjust xG based on the opponent's defense
        team1_adjusted_xG = (team1_xG + team2_xGA) / 2
        team2_adjusted_xG = (team2_xG + team1_xGA) / 2

        # Calculate probabilities for different scorelines
        poisson_probs = self.calculate_poisson_scoreline_prob(team1_adjusted_xG, team2_adjusted_xG, show_plot=show_plot)

        # Calculate outcome probabilities
        team1_win_prob = np.sum(np.tril(poisson_probs, -1))
        team2_win_prob = np.sum(np.triu(poisson_probs, 1))
        draw_prob = np.sum(np.diag(poisson_probs))

        # Predict the most likely outcome
        outcomes = ['Team 1 Win', 'Draw', 'Team 2 Win']
        outcome_probs = [team1_win_prob, draw_prob, team2_win_prob]
        predicted_outcome = outcomes[np.argmax(outcome_probs)]

        # Find the most likely scoreline
        most_likely_score = np.unravel_index(poisson_probs.argmax(), poisson_probs.shape)

        return {
            'predicted_outcome': predicted_outcome,
            'team1_win_prob': team1_win_prob,
            'draw_prob': draw_prob,
            'team2_win_prob': team2_win_prob,
            'most_likely_score': f"{most_likely_score[0]}-{most_likely_score[1]}",
            'team1_adjusted_xG': team1_adjusted_xG,
            'team2_adjusted_xG': team2_adjusted_xG
        }


if __name__ == "__main__":
    data_path = 'data/football_data.csv'
    target_columns = ['Gls', 'GA']
    model_trainer = SportsPredictionModel(data_path, target_columns, columns_to_drop=['Squad', 'Season'])

    # Feature Engineering
    model_trainer.feature_engineering()

    # Train-Test Split
    splits = model_trainer.train_test_split()

    # Train and evaluate XGBoost Models
    for target in target_columns:
        X_train, X_test, y_train, y_test = splits[target]
        xgb_model = model_trainer.train_xgboost(X_train, y_train)
        model_trainer.evaluate_model(xgb_model, X_test, y_test, model_type='xgboost')
        model_trainer.save_model(xgb_model, f'xgboost_model_{target}.pkl')

    # Train and evaluate Neural Network Models
    for target in target_columns:
        X_train, X_test, y_train, y_test = splits[target]
        nn_model = model_trainer.train_neural_network(X_train, y_train)
        model_trainer.evaluate_model(nn_model, X_test, y_test, model_type='neural_network')
        model_trainer.save_model(nn_model, f'nn_model_{target}.h5')

    # Predict Game Outcome
    current_season_data = pd.read_csv('data/current_season_data.csv')
    team1_features = current_season_data[current_season_data['Squad'] == 'Arsenal'].drop(columns=['Squad']).iloc[0]
    team2_features = current_season_data[current_season_data['Squad'] == 'Luton Town'].drop(columns=['Squad']).iloc[0]

    outcome = model_trainer.predict_game_outcome(team1_features, team2_features, show_plot=True)
    print(outcome)
