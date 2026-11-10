"""
Model type 4: multilayer perceptron (fully-connected neural network).

Standardised features -> 3 hidden layers of 128 tanh neurons -> asinh(I/I0)
(standardised). Trained with Adam; training stops when the score on 10 % of
the TRAINING set stops improving.

Run alone:  python scripts/ml/model_mlp.py   (prints validation metrics)
"""
import sys
from pathlib import Path

from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NAME = "mlp"
LABEL = "MLP neural network"
HYPERPARAMS = ("3 x 128 tanh, Adam lr=1e-3, batch 64, L2 alpha=1e-6, up to 3000 epochs, "
               "early stopping (patience 60) on 10 % of the training set")


def make_regressor(n_jobs: int = 1):
    mlp = MLPRegressor(hidden_layer_sizes=(128, 128, 128), activation="tanh",
                       solver="adam", learning_rate_init=1e-3, batch_size=64,
                       alpha=1e-6, max_iter=3000, tol=1e-8,
                       early_stopping=True, validation_fraction=0.1, n_iter_no_change=60,
                       random_state=0)
    return make_pipeline(StandardScaler(),
                         TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler()))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common
    common.run_standalone(sys.modules[__name__])
