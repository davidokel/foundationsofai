from sklearn.linear_model import Ridge

from AI_library.LinearRegression import LinearRegressionModel


class RidgeRegressionModel(LinearRegressionModel):
    alpha = []

    def __init__(self, alpha, x_train, y_train):
        super().__init__(x_train, y_train)
        self.alpha = alpha
        self.model = Ridge(alpha).fit(X=x_train, y=y_train)

# todo find best alpha in init