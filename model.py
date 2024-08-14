import os
import joblib
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import learning_curve
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SportsPredictionModel:
    def __init__(self, data_path, target_columns, columns_to_drop=None):
        self.preprocessor = DataPreprocessor(data_path, target_columns, columns_to_drop)
        self.trainer = None
        self.evaluator = None

    def prepare_data(self):
        try:
            self.preprocessor.load_data()
            self.preprocessor.preprocess_data()
            self.preprocessor.feature_engineering()
            X, y = self.preprocessor.get_processed_data()
            self.trainer = ModelTrainer(X, y)
        except Exception as e:
            logging.error(f"Error in data preparation: {e}")
            raise

    def train_and_evaluate(self):
        X_train, X_test, y_train, y_test = self.trainer.train_test_split()
        self.evaluator = ModelEvaluator(X_test, y_test, self.preprocessor.target_columns)

        models = {
            'Random Forest': self.trainer.train_random_forest(X_train, y_train),
            'XGBoost': self.trainer.train_xgboost(X_train, y_train),
            'Neural Network': self.trainer.train_neural_network(X_train, y_train),
            'Linear Regression': self.trainer.train_linear_regression(X_train, y_train)
        }

        model_performance = {}
        for model_name, model in models.items():
            model_performance[model_name] = self.evaluator.evaluate_model(model, model_type=model_name)
            # if self.trainer.feature_names is not None:
            #     self.evaluator.feature_importance(model, model_name, self.trainer.feature_names)
            # else:
            #     logging.warning(f"Feature names not available for {model_name}. Skipping feature importance.")

        return model_performance

    @staticmethod
    def plot_learning_curves(model, X_train, y_train):
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=5, scoring='neg_mean_squared_error',
            train_sizes=np.linspace(0.1, 1.0, 10)
        )

        train_scores_mean = -np.mean(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', label="Training Error")
        plt.plot(train_sizes, test_scores_mean, 'o-', label="Validation Error")
        plt.title("Learning Curves")
        plt.xlabel("Training Set Size")
        plt.ylabel("MSE")
        plt.legend(loc="best")
        plt.show()

    @staticmethod
    def analyze_feature_correlations(X, y, top_n_features=10, correlation_threshold=0.7):
        df = pd.DataFrame(np.concatenate((X, y), axis=1))
        df.columns = list(range(X.shape[1])) + ['Gls', 'GA']

        # Calculate the correlation matrix
        corr = df.corr()

        # Filter correlations for Gls and GA
        gls_corr = corr['Gls'].abs().sort_values(ascending=False)
        ga_corr = corr['GA'].abs().sort_values(ascending=False)

        # Select top features for Gls and GA based on correlation
        top_gls_features = gls_corr.head(top_n_features).index
        top_ga_features = ga_corr.head(top_n_features).index

        # Filter the correlation matrix for these selected features
        selected_gls_features = top_gls_features.union(['Gls'])
        selected_ga_features = top_ga_features.union(['GA'])

        # Plot the heatmap for Gls
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr.loc[selected_gls_features, selected_gls_features], cmap='coolwarm', annot=True, fmt=".2f")
        plt.title('Top Feature Correlations with Gls')
        plt.show()

        # Plot the heatmap for GA
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr.loc[selected_ga_features, selected_ga_features], cmap='coolwarm', annot=True, fmt=".2f")
        plt.title('Top Feature Correlations with GA')
        plt.show()

    @staticmethod
    def save_model(model, filename):
        os.makedirs('models', exist_ok=True)
        file_path = os.path.join('models', filename)
        try:
            joblib.dump(model, file_path)
            logging.info(f"Model saved successfully: {file_path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    @staticmethod
    def load_model(filename):
        file_path = os.path.join('models', filename)
        try:
            model = joblib.load(file_path)
            logging.info(f"Model loaded successfully: {file_path}")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None


if __name__ == "__main__":
    data_path = 'data/football_data.csv'
    target_columns = ['Gls', 'GA']
    columns_to_drop = ['Squad', 'Season']

    model = SportsPredictionModel(data_path, target_columns, columns_to_drop)
    model.prepare_data()
    model_performance = model.train_and_evaluate()

    # model.plot_learning_curves(
    # model.trainer.train_linear_regression(model.trainer.X, model.trainer.y), model.trainer.X, model.trainer.y)

    logging.info("Model Performance Summary:")
    for model_name, performance in model_performance.items():
        logging.info(f"\n{model_name}:")
        for target, metrics in performance.items():
            logging.info(f"  {target}:")
            for metric_name, value in metrics.items():
                logging.info(f"    {metric_name}: {value}")
