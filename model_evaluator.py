import logging
import os
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelEvaluator:
    def __init__(self, X_test, y_test, target_columns):
        self.X_test = X_test
        self.y_test = y_test
        self.target_columns = target_columns
        self.prepare()

    @staticmethod
    def prepare():
        os.makedirs('evals', exist_ok=True)

    def evaluate_model(self, model, model_type='model'):
        y_pred = model.predict(self.X_test)
        metrics = {}

        for i, target in enumerate(self.target_columns):
            mse = mean_squared_error(self.y_test.iloc[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test.iloc[:, i], y_pred[:, i])
            r2 = r2_score(self.y_test.iloc[:, i], y_pred[:, i])
            metrics[target] = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

            logging.info(f'{model_type} {target} Metrics:')
            logging.info(f'  Mean Squared Error: {mse}')
            logging.info(f'  Root Mean Squared Error: {rmse}')
            logging.info(f'  Mean Absolute Error: {mae}')
            logging.info(f'  R^2 Score: {r2}')

            # Plot predictions vs actual values
            plt.figure(figsize=(10, 5))
            plt.scatter(self.y_test.iloc[:, i], y_pred[:, i])
            plt.xlabel(f"Actual {target}")
            plt.ylabel(f"Predicted {target}")
            plt.title(f"{model_type.capitalize()} Model: {target} Actual vs Predicted")
            plt.savefig(f"evals/{model_type}_{target}_actual_vs_predicted.png")
            plt.close()

        return metrics

    def feature_importance(self, model, model_type, feature_names):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            # Plot top 10 and bottom 10 features
            plt.figure(figsize=(10, 6))
            plt.title(f'{model_type} Top 10 & Bottom 10 Feature Importances')
            plt.bar(range(20), np.concatenate((importances[indices[:10]], importances[indices[-10:]])), align='center')
            plt.xticks(range(20), np.concatenate((feature_names[indices[:10]], feature_names[indices[-10:]])), rotation=45)
            plt.tight_layout()
            plt.savefig(f"evals/{model_type}_feature_importance.png")
            plt.show()
            plt.close()
        else:
            print(f"{model_type} does not support native feature importance. Using SHAP values.")
            explainer = shap.Explainer(model.predict, self.X_test)
            shap_values = explainer(self.X_test)
            shap.summary_plot(shap_values, self.X_test, feature_names=feature_names)