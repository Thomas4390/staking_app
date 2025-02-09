"""
utils.py

Contains utility functions for the Streamlit application, plus suggested approaches
for ensuring strictly positive fitted weights.
"""

import numpy as np
from typing import Callable, Tuple
from scipy.optimize import curve_fit

def polynomial_fit(x_data: np.ndarray, y_data: np.ndarray, degree: int) -> np.ndarray:
    """
    Fits a polynomial of a given degree to the data using numpy.polyfit.

    NOTE on positivity:
      - Polynomials can become negative within the domain, so positivity is not guaranteed
        unless you reparameterize or clamp. For a strictly positive fit over a given range,
        consider exponentiating the polynomial or using advanced constraints.
    """
    coeffs = np.polyfit(x_data, y_data, deg=degree)
    return coeffs

def polynomial_eval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Evaluates the polynomial (given by its coefficients) at points x.

    :param coeffs: Coefficients (as returned by np.polyfit).
    :param x: Array of x-values.
    :return: Evaluated polynomial values.
    """
    return np.polyval(coeffs, x)

def exponential_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Exponential function: y = a * exp(b * x).

    For positivity:
      - If you use curve_fit with bounds=( (0, -np.inf), (np.inf, np.inf) ), you can ensure a > 0.
    """
    return a * np.exp(b * x)

def logarithmic_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Logarithmic function: y = a + b * ln(x).

    This can become negative if 'a + b * ln(x)' < 0 for certain x.
    Use bounds or reparameterize if you need guaranteed positivity.
    """
    return a + b * np.log(x + 1e-9)

def power_law_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Power law function: y = a * x^b.

    With x>0, if a>0, this is guaranteed positive.
    You can ensure a>0 via bounds in curve_fit, e.g., bounds=((0, -np.inf), (np.inf, np.inf)).
    """
    return a * np.power(x, b)

def sigmoid_func(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    """
    Sigmoid (logistic) function: y = L / (1 + exp(-k*(x - x0))).

    This is always positive for L>0.
    """
    return L / (1 + np.exp(-k * (x - x0)))

def curve_fit_function(
    x_data: np.ndarray,
    y_data: np.ndarray,
    func: Callable[..., np.ndarray],
    p0: Tuple[float, ...] = (1.0, 0.1),
    bounds: Tuple = (-np.inf, np.inf),
    method: str = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uses scipy.optimize.curve_fit to fit a function to data.

    If you want strictly positive predictions:
      - For certain functions (exponential, power law), use parameter bounds so that
        the function remains positive. E.g. bounds=((0, -np.inf), (np.inf, np.inf)).
      - For polynomials or logs, consider reparameterizing or post-fit clamping.

    :param x_data: Array of x-values.
    :param y_data: Array of y-values.
    :param func: The model function, f(x, ...).
    :param p0: Initial guess for the parameters.
    :param bounds: Lower and upper bounds on parameters. (a tuple of arrays)
    :param method: Optimization method (e.g., 'trf', 'dogbox').
    :param kwargs: Additional keyword arguments passed to curve_fit.
    :return: (popt, pcov) where popt are the optimal parameters.
    """
    popt, pcov = curve_fit(func, x_data, y_data, p0=p0, bounds=bounds, method=method, **kwargs)
    return popt, pcov

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Mean Squared Error (MSE).
    """
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Mean Absolute Error (MAE).
    """
    return np.mean(np.abs(y_true - y_pred))

def eval_custom_function(x: np.ndarray, user_expr: str) -> np.ndarray:
    """
    Evaluates a custom function expression provided as a string.

    WARNING: Using eval() can be dangerous. This is for demonstration only.

    :param x: Array of x-values.
    :param user_expr: A string expression (e.g., "0.01*x**2 + 0.2*x + 1").
    :return: The evaluated function values.
    """
    allowed_names = {
        'x': x,
        'np': np
    }
    return eval(user_expr, {"__builtins__": {}}, allowed_names)
