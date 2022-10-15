from sklearn.linear_model import Ridge

from notebooks.week_1.LinearRegression import LinearRegressionModel


class RidgeRegressionModel(LinearRegressionModel):
    alpha = []

    def __init__(self, alpha, x_train, y_train):
        self.alpha = alpha
        self.x_train = x_train
        self.y_train = y_train
        self.model = Ridge(alpha).fit(X=x_train, y=y_train)
