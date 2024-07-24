import joblib
import pandas as pd


class ValidationModule:
    def __init__(
        self,
        model,
        ar_order: tuple[int, int],
        target: str,
        exog: list[str],
        actuator_delay: int = 8,  # measured empirically
        actuator_command: str | None = "steerFiltered",
    ):
        self.model = model
        self.ar_order = ar_order
        if self.ar_order[0] != self.ar_order[1]:
            raise NotImplementedError("Different orders for endogenous and exogenous variables are untested")
        self.target = target
        self.exog = exog
        self.actuator_delay = actuator_delay
        self.actuator_command = actuator_command

    def add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        # endogenous lags
        df["target"] = df[self.target].shift(-1)
        for i in range(1, self.ar_order[0] + 1):
            df[f"{self.target}_{i}"] = df[self.target].shift(i)

        # exogenous lags
        for col in self.exog:
            if col == self.actuator_command:
                df[col] = df[col].shift(-self.actuator_delay)
            else:
                for i in range(1, self.ar_order[1] + 1):
                    df.loc[:, f"{col}_{i}"] = df[col].shift(i)

        df = df.dropna(subset=[c for c in df.columns if c != "target"])
        return df

    def fit(self, dfs: list[pd.DataFrame] | pd.DataFrame):
        if isinstance(dfs, pd.DataFrame):
            dfs = [dfs]
        with joblib.Parallel(n_jobs=-1) as parallel:
            dfs = parallel(joblib.delayed(self.add_lags)(df) for df in dfs)
        df = pd.concat(dfs).reset_index(drop=True).dropna()
        X = df.drop(columns=["target"])
        y = df["target"]
        print(f"Fitting model with {len(X):,} samples")
        return self.model.fit(X, y)

    def predict(self, df: pd.DataFrame, n: int = 20) -> pd.Series:
        df = df.copy()
        start = self.ar_order[0] + 1
        df.loc[start:, self.target] = None  # avoid inadvertent data leakage

        for i in range(start, start + n):
            X = df.iloc[i-start:i]
            X = self.add_lags(X).drop(columns=["target"])
            pred = self.model.predict(X)
            df.loc[i, self.target] = pred
        return df[self.target].iloc[:start + n]
