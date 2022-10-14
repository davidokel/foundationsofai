from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class LinearRegressionModel:
    model = None

    x_train = []
    y_train = []

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.model = LinearRegression().fit(X=x_train, y=y_train)

    def get_line_fit(self):
        return {
            "bias": self.model.intercept_,
            "weights": self.model.coef_,
        }

    def predict(self, X_test, y_test=None):
        y_predicted = self.model.predict(X_test)
        if type(y_test) == type(None):
            return y_predicted
        else:
            MSE = mean_squared_error(y_test, y_predicted)
            return y_predicted, MSE
