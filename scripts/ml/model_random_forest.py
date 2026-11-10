"""
Model type 2: random forest.

An average of 300 decision trees, each grown on a bootstrap sample with a
random half of the features considered at every split.

Run alone:  python scripts/ml/model_random_forest.py   (prints validation metrics)
"""
import sys
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

NAME = "random_forest"
LABEL = "Random forest"
HYPERPARAMS = "300 trees, max_features=0.5, min_samples_leaf=1"


def make_regressor(n_jobs: int = 1):
    return RandomForestRegressor(n_estimators=300, max_features=0.5, min_samples_leaf=1,
                                 n_jobs=n_jobs, random_state=0)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common
    common.run_standalone(sys.modules[__name__])
