"""
Model type 1: polynomial ridge regression (baseline).

Standardised features -> all polynomial terms up to degree 3 -> ridge
(L2-regularised linear) regression on y = asinh(I/I0).

Run alone:  python scripts/ml/model_poly_ridge.py   (prints validation metrics)
"""
import sys
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

NAME = "poly_ridge"
LABEL = "Polynomial ridge (d=3)"
HYPERPARAMS = "all terms up to degree 3 of the 31 standardised inputs (25 features + 6 physics-equation values, ~6,000 terms), ridge alpha=1e-3"


def make_regressor(n_jobs: int = 1):
    return make_pipeline(StandardScaler(),
                         PolynomialFeatures(degree=3, include_bias=False),
                         StandardScaler(),
                         Ridge(alpha=1e-3))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common
    common.run_standalone(sys.modules[__name__])
