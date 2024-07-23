class TrivialModel:
    def __init__(self, column: str):
        """A trivial model that predicts y(t) = y(t-1)"""
        self.column = column

    def fit(self, X, y):
        pass

    def predict(self, X):
        return X[self.column].iloc[-1]
