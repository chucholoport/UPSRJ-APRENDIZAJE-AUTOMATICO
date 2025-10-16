# ============================================================
# Politécnica de Santa Rosa
#
# Materia: Aprendizaje automático
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC02
# Archivo: logistic_regression.py
# Descripción: Definición de clase LogisticRegression
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

from regression_models.data_source import DataSource as ds

class LogisticRegressionCompare:
    def __init__(self, url: str, base: str, out: str):
        self.source = ds(url, churn=True)
        # ===== CORRECCIÓN AQUÍ =====
        # Guardar los nombres de las columnas para usarlos más tarde en el gráfico
        feature_cols = ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']
        self.x = np.asarray(self.source.data[feature_cols])
        self.y = np.asarray(self.source.data[base])
        
        self.std_scaler, self.x_std = self.standarize(x=self.x)
        self.d = self.prepare_data(x=self.x_std, y=self.y, prc=0.2, random_state=4)
        self.m = self.create_model()
        self.train_model(self.m, self.d)
        
        # ===== CORRECCIÓN AQUÍ =====
        # Se pasa la lista correcta de 7 nombres de características al parámetro 'index'
        self.plot_model_and_predict(self.m, index=feature_cols, x=self.d[1],
                                    y=self.d[3], out=os.path.join(out, "logistic_regression_churn_coefficients.png"))

    # (El resto de la clase no cambia)
    def standarize(self, x: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
        std_scaler = StandardScaler()
        x_std = std_scaler.fit_transform(x)
        return std_scaler, x_std
    def prepare_data(self, x:np.ndarray, y:np.ndarray, prc: float, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=prc, random_state=random_state)
        return (x_train, x_test, y_train, y_test)
    def create_model(self) -> LogisticRegression:
        return LogisticRegression(C=0.01, solver='liblinear')
    def train_model(self, model: LogisticRegression, data: tuple) -> None:
        x_train, y_train = data[0], data[2]
        model.fit(x_train, y_train)
    def plot_model_and_predict(self, model: LogisticRegression, index, x: np.ndarray, y:np.ndarray, out: str) -> None:
        try:
            yhat_prob = model.predict_proba(x)
            coefficients = pd.Series(model.coef_[0], index=index)
            plt.figure(figsize=(10, 6))
            coefficients.sort_values().plot(kind='barh')
            plt.title("Feature Coefficients in Logistic Regression Churn Model")
            plt.xlabel("Coefficient Value")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.savefig(out)
            plt.close()
            print(f"Se creó gráfico de coeficientes en {out}")
            loss = log_loss(y, yhat_prob)
            print(f"LogLoss (pérdida logarítmica) en el conjunto de prueba: {loss:.4f}")
        except Exception as e:
            print(f"Error: no se pudo crear gráfico de coeficientes en {out}: {e}")