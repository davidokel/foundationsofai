from sklearn.kernel_ridge import KernelRidge

from AI_library.RidgeRegression import RidgeRegressionModel


class KernelRidgeRegressionModel(RidgeRegressionModel):

    def __init__(self, alpha, x_train, y_train, **kwargs):
        super().__init__(alpha, x_train, y_train)
        self.model = KernelRidge(alpha, **kwargs).fit(x_train, y_train)
