# ============================================================
# Politécnica de Santa Rosa
#
# Materia: Aprendizaje automático
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC02
# Archivo: linear_regression.py
# Descripción: Definición de clase LinearRegression
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from linear_regression.data_source import DataSource as ds

class LinearRegressionCompare:
    def __init__(self, url: str, hist: str, base: str, f1: str, f2: str, out: str):
        self.source = ds(url)
        self.histogram = hist
        # Característica base de comparación
        self.base = base
        # Opciones de características a comparar
        self.f1 = f1
        self.f2 = f2
        # Histograma
        self.source.set_histogram(self.histogram)
        # Seleccionar las características para regresión lineal
        self.x1 = self.source.get_relevant_feature(self.f1)
        self.x2 = self.source.get_relevant_feature(self.f2)
        self.y = self.source.get_relevant_feature(self.base)
        # Relación entre opciones y base
        self.source.plot_relationship(x=self.x1, y=self.y, x_label=self.f1.capitalize(), y_label=self.base.capitalize(), 
                             out=os.path.join(out, f"relationship_{self.f1.lower()}_{self.base.lower()}.png"))
        self.source.plot_relationship(x=self.x2, y=self.y, x_label=self.f2.capitalize(), y_label=self.base.capitalize(), 
                             out=os.path.join(out, f"relationship_{self.f2.lower()}_{self.base.lower()}.png"))
        # Preparar información al 80% para entrenamiento y 20% para pruebas
        self.d1 = self.prepare_data(x=self.x1, y=self.y, prc=0.2, random_state=42)
        self.d2 = self.prepare_data(x=self.x2, y=self.y, prc=0.2, random_state=42)
        # Creación de modelos de regresión lineal para las opciones
        self.m1 = self.create_model()
        self.m2 = self.create_model()
        # Entrenamiento de modelos
        self.train_model(self.m1, self.d1)
        self.train_model(self.m2, self.d2)
        # Coeficientes de regresor y la intercepción
        self.get_coef_and_int(self.m1)
        self.get_coef_and_int(self.m2)
        # Predicciones con modelos
        self.p1 = self.predict(self.m1, self.x1)
        self.p2 = self.predict(self.m2, self.x2)
        # Evaluación de modelos
        self.evaluate(self.y, self.p1)
        self.evaluate(self.y, self.p2)
        # Gráfico de Regresión lineal
        self.plot_model(model=self.m1, x=self.x1, y=self.y, x_label=self.f1.capitalize(), y_label=self.base.capitalize(),
                        out=os.path.join(out, f"linear_regression_{self.f1.lower()}_{self.base.lower()}.png"))        
        self.plot_model(model=self.m2, x=self.x2, y=self.y, x_label=self.f2.capitalize(), y_label=self.base.capitalize(),
                        out=os.path.join(out, f"linear_regression_{self.f2.lower()}_{self.base.lower()}.png"))
        
    # TODO: Define un método que prepare la información para ser analizada por regresión lineal.
    #       Recuerda que al hacer un modelo de aprendizaje automático debemos dividir la información disponible en
    #       datos de entrenamiento y datos de pruebas, por lo tanto, a la salida debe 
    #       haber un tuple(x_train, x_test, y_train, y_test) de arreglos de numpy.
    # NOTE: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    def prepare_data(self, x:np.ndarray, y:np.ndarray, prc: float, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Divide los datos en conjuntos de entrenamiento y prueba.

        Args:
            x (np.ndarray): Característica independiente.
            y (np.ndarray): Variable dependiente.
            prc (float): Proporción para prueba (entre 0 y 1).
            random_state (int): Semilla para reproducibilidad.

        Returns:
            tuple: (x_train, x_test, y_train, y_test)
        """
        # Usar train_test_split para dividir los datos
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=prc, random_state=random_state)
        return (x_train, x_test, y_train, y_test)
    
    # TODO: Define un método que devuelva un objeto "linear_model.LinearRegression" de scikit-learn.
    # NOTE: https://scikit-learn.org/stable/modules/linear_model.html
    def create_model(self) -> linear_model.LinearRegression:
        """
        
        Crea un modelo de regresión lineal.

        Returns:
            LinearRegression: Modelo vacío listo para entrenar.
        """
        model = linear_model.LinearRegression()
        return model
    
    # TODO: Define un método que entrene un modelo de entrada "linear_model" de scikit-learn
    #       con la información de entrada "data".
    # NOTE: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.reshape.html
    #       https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.fit
    def train_model(self, model: linear_model.LinearRegression, data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        """
        Entrena el modelo con los datos de entrenamiento.

        Args:
            model (LinearRegression): Modelo a entrenar.
            data (tuple): (x_train, x_test, y_train, y_test)
        """
        x_train = data[0]
        y_train = data[2]
    
        # Reshape x_train si es necesario (debe ser 2D para sklearn)
        if len(x_train.shape) == 1:
            x_train = x_train.reshape(-1, 1)
    
        # Entrenar el modelo
        model.fit(x_train, y_train)
    
    # TODO: Define un método que obtenga los coeficientes de regresor y la intercepción
    #       de un modelo de entrada "linear_model" de scikit-learn.
    # NOTE: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression
    def get_coef_and_int(self, model: linear_model.LinearRegression) -> None:
        """
        Imprime los coeficientes y la intercepción del modelo.

        Args:
            model (LinearRegression): Modelo entrenado.
        """
        print("Coeficientes del modelo:")
        print(f"- Coeficiente(s): {model.coef_}")
        print(f"- Intercepción: {model.intercept_}")
    
    # TODO: Define un método que haga una predicción con el modelo y carácteristica de entrada.
    # NOTE: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.predict
    def predict(self, model: linear_model.LinearRegression, x: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo.

        Args:
            model (LinearRegression): Modelo entrenado.
            x (np.ndarray): Datos de entrada.

        Returns:
            np.ndarray: Predicciones.
        """
        # Reshape x si es necesario (debe ser 2D para sklearn)
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
    
        prediction = model.predict(x)
        return prediction
    
    def evaluate(self, y: np.ndarray, prediction: np.ndarray) -> None:
        print("Evaluación de modelo:")
        print("- Mean absolute error: %.2f" % mean_absolute_error(y, prediction))
        print("- Mean squared error: %.2f" % mean_squared_error(y, prediction))
        print("- Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y, prediction)))
        print("- R2-score: %.2f" % r2_score(y, prediction))
        
    def plot_model(self, model: linear_model.LinearRegression, x: np.ndarray, y: np.ndarray, x_label: str, y_label: str, out: str) -> None:
        try:
            plt.scatter(x, y, color='blue')
            plt.plot(x, model.coef_[0] * x + model.intercept_, '-r')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.savefig(out)
            print(f"Se creó gráfico de regresión lineal en {out}")
        except Exception as e:
            print(f"Error: no se pudo crear gráfico del modelo: {e}")