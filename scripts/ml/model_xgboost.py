"""
Model type 3: gradient-boosted trees (XGBoost).

Trees are added one at a time, each fitting the remaining error. The number
of trees is chosen by early stopping on 10 % of the TRAINING set (the
validation set is kept untouched for model comparison).

Run alone:  python scripts/ml/model_xgboost.py   (prints validation metrics)
"""
import sys
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

NAME = "xgboost"
LABEL = "XGBoost"
HYPERPARAMS = ("up to 6000 trees, lr=0.03, max_depth=6, subsample=0.8, colsample=0.8, "
               "early stopping (200 rounds) on 10 % of the training set")


class XGBEarlyStop:
    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs
        self.model = None

    def fit(self, x, y):
        idx = np.random.default_rng(0).permutation(len(x))
        n_es = len(x) // 10
        es, tr = idx[:n_es], idx[n_es:]
        self.model = XGBRegressor(n_estimators=6000, learning_rate=0.03, max_depth=6,
                                  subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                                  reg_lambda=1.0, tree_method="hist",
                                  early_stopping_rounds=200, eval_metric="rmse",
                                  n_jobs=self.n_jobs, random_state=0)
        self.model.fit(x[tr], y[tr], eval_set=[(x[es], y[es])], verbose=False)
        return self

    def predict(self, x):
        return self.model.predict(x)


def make_regressor(n_jobs: int = 1):
    return XGBEarlyStop(n_jobs=n_jobs if n_jobs > 0 else 0)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common
    common.run_standalone(sys.modules[__name__])
