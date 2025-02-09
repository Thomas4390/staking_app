"""
1_Staking_Distribution_Fit.py

Allows the user to enter raw weights for each staking tier (normalized to 100%).
Chooses a model type, fits the data, and displays:
 - The distribution table (including a totals row).
 - The fitted formula in code and LaTeX form.
 - The chosen fit metric (MSE/MAE) displayed in LaTeX.

Adjustments from the previous version:
 - Default polynomial degree is 3.
 - After any clamping, we normalize the 8 fitted weights so that their sum is 100%.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils import (
    polynomial_fit, polynomial_eval,
    exponential_func, logarithmic_func,
    power_law_func, sigmoid_func,
    curve_fit_function,
    mean_squared_error, mean_absolute_error
)

def main():
    st.title("Staking Distribution Fit")

    st.write(
        """
        Enter the raw weights for each staking tier below.
        These weights will be normalized to sum to 100%.
        Next, choose a model type, select a fit metric (MSE by default),
        and click the **Fit Model** button to see:
         - The fitted curve,
         - A distribution table (with totals),
         - The fitted formula (both code & LaTeX),
         - The fit metric rendered in LaTeX.
        """
    )

    # Staking tiers
    tiers = np.array([1, 3, 7, 14, 30, 90, 180, 365], dtype=int)
    nb_tiers = len(tiers)

    # Default raw weights
    default_raw_weights = [1.0, 1.2, 1.5, 2.0, 6.0, 24.0, 60.0, 150.0]

    st.header("Input Weights")
    raw_user_weights = []
    for i, day in enumerate(tiers):
        val = st.number_input(
            f"Weight for {day} days",
            min_value=0.0, max_value=10000.0,
            value=default_raw_weights[i], step=0.1
        )
        raw_user_weights.append(val)

    sum_raw = sum(raw_user_weights)
    if sum_raw == 0:
        st.error("All input weights are zero. Please enter some values.")
        return
    normalized_weights = [(w / sum_raw) * 100 for w in raw_user_weights]

    st.header("Model & Metric Selection")
    model_options = ["Polynomial", "Exponential", "Logarithmic", "Power Law", "Sigmoid"]
    model_choice = st.selectbox("Choose model type", model_options)

    degree = None
    if model_choice == "Polynomial":
        # Default degree is now 3
        degree = st.slider("Polynomial degree", 1, 8, 3)

    metric_options = ["MSE", "MAE"]
    metric_choice = st.selectbox("Select fit metric", metric_options, index=0)

    # Toggle: do we clamp negative fitted weights to a small positive value?
    clamp_negatives = st.checkbox("Clamp negative fitted weights to a small positive value?", value=False)

    if st.button("Fit Model"):
        x_data = tiers.astype(float)
        y_data = np.array(normalized_weights, dtype=float)

        total_tokens = 3.46e9

        # --- Fit the chosen model ---
        if model_choice == "Polynomial":
            coeffs = polynomial_fit(x_data, y_data, degree=degree)
            x_fit = np.linspace(min(x_data), max(x_data), 200)
            y_fit = polynomial_eval(coeffs, x_fit)
            y_pred_points = polynomial_eval(coeffs, x_data)

            # Build polynomial formula strings
            poly_terms = []
            poly_latex_terms = []
            for i, a in enumerate(coeffs):
                power = degree - i
                if power > 0:
                    poly_terms.append(f"({a:.5g})*x**({power})")
                    poly_latex_terms.append(f"{a:.5g}x^{{{power}}}")
                else:
                    poly_terms.append(f"({a:.5g})")
                    poly_latex_terms.append(f"{a:.5g}")
            formula_code = "y = " + " + ".join(poly_terms)
            formula_latex = "y = " + " + ".join(poly_latex_terms)

        elif model_choice == "Exponential":
            try:
                popt, _ = curve_fit_function(x_data, y_data, exponential_func, p0=(1.0, 0.01), method="trf")
            except Exception as e:
                st.error(f"Exponential fit error: {e}")
                return
            a_opt, b_opt = popt
            x_fit = np.linspace(min(x_data), max(x_data), 200)
            y_fit = exponential_func(x_fit, a_opt, b_opt)
            y_pred_points = exponential_func(x_data, a_opt, b_opt)
            formula_code = f"y = ({a_opt:.5g})*np.exp(({b_opt:.5g})*x)"
            formula_latex = f"y = {a_opt:.5g}\\,\\exp({b_opt:.5g}x)"

        elif model_choice == "Logarithmic":
            try:
                popt, _ = curve_fit_function(x_data, y_data, logarithmic_func, p0=(1.0, 1.0), method="trf")
            except Exception as e:
                st.error(f"Logarithmic fit error: {e}")
                return
            a_opt, b_opt = popt
            x_fit = np.linspace(min(x_data), max(x_data), 200)
            y_fit = logarithmic_func(x_fit, a_opt, b_opt)
            y_pred_points = logarithmic_func(x_data, a_opt, b_opt)
            formula_code = f"y = ({a_opt:.5g}) + ({b_opt:.5g})*np.log(x)"
            formula_latex = f"y = {a_opt:.5g} + {b_opt:.5g}\\,\\ln(x)"

        elif model_choice == "Power Law":
            try:
                popt, _ = curve_fit_function(x_data, y_data, power_law_func, p0=(1.0, 0.1), method="trf")
            except Exception as e:
                st.error(f"Power law fit error: {e}")
                return
            a_opt, b_opt = popt
            x_fit = np.linspace(min(x_data), max(x_data), 200)
            y_fit = power_law_func(x_fit, a_opt, b_opt)
            y_pred_points = power_law_func(x_data, a_opt, b_opt)
            formula_code = f"y = ({a_opt:.5g})*x**({b_opt:.5g})"
            formula_latex = f"y = {a_opt:.5g}\\,x^{{{b_opt:.5g}}}"

        else:  # Sigmoid
            try:
                popt, _ = curve_fit_function(x_data, y_data, sigmoid_func, p0=(100, 0.1, np.median(x_data)), method="trf")
            except Exception as e:
                st.error(f"Sigmoid fit error: {e}")
                return
            L_opt, k_opt, x0_opt = popt
            x_fit = np.linspace(min(x_data), max(x_data), 200)
            y_fit = sigmoid_func(x_fit, L_opt, k_opt, x0_opt)
            y_pred_points = sigmoid_func(x_data, L_opt, k_opt, x0_opt)
            formula_code = f"y = ({L_opt:.5g})/(1+np.exp(-({k_opt:.5g})*(x-({x0_opt:.5g}))))"
            formula_latex = f"y = \\frac{{{L_opt:.5g}}}{{1+\\exp(-{k_opt:.5g}(x-{x0_opt:.5g}))}}"

        # --- Optional clamp then normalize the fitted weights so they sum to 100 ---
        if clamp_negatives:
            y_pred_points = np.clip(y_pred_points, 1e-9, None)  # remplacer les valeurs négatives

        fitted_sum = np.sum(y_pred_points)
        if fitted_sum > 0:
            y_pred_points = (y_pred_points / fitted_sum) * 100

        # Compute metric
        if metric_choice == "MSE":
            fit_metric = mean_squared_error(y_data, y_pred_points)
            metric_name = "MSE"
        else:
            fit_metric = mean_absolute_error(y_data, y_pred_points)
            metric_name = "MAE"

        # Distribution table
        user_tokens = (y_data / 100) * total_tokens
        fitted_tokens = (y_pred_points / 100) * total_tokens

        df = pd.DataFrame({
            "Tier": [f"Tier {i+1}" for i in range(nb_tiers)],
            "Days": tiers,
            "User Weights (%)": y_data,
            "User Tokens": user_tokens,
            "Fitted Weights (%)": y_pred_points,
            "Fitted Tokens": fitted_tokens
        })

        df.loc[len(df)] = {
            "Tier": "Total",
            "Days": "",
            "User Weights (%)": np.sum(y_data),
            "User Tokens": np.sum(user_tokens),
            "Fitted Weights (%)": np.sum(y_pred_points),
            "Fitted Tokens": np.sum(fitted_tokens)
        }

        st.subheader("Distribution Table")
        st.dataframe(df.style.format({
            "User Weights (%)": "{:.2f}",
            "User Tokens": "{:,.0f}",
            "Fitted Weights (%)": "{:.2f}",
            "Fitted Tokens": "{:,.0f}"
        }))

        st.subheader("Total Tokens")
        st.write(f"**{total_tokens:,.0f} tokens**")

        st.subheader("Fitted Formula")
        st.write("Copy‑paste version (for Custom Fit page):")
        st.code(formula_code, language="python")

        st.write("LaTeX-rendered equation:")
        st.latex(formula_latex)

        st.subheader("Fit Metric")
        st.latex(fr"{metric_name} = {fit_metric:.5g}")

        # Plot
        # We'll show y_fit *without normalization for illustrative shape
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data,
            mode="markers", name="User (Normalized)",
            marker=dict(color="red", size=8)
        ))
        fig.add_trace(go.Scatter(
            x=x_fit, y=y_fit,
            mode="lines", name=f"{model_choice} Fit (Raw)",
            line=dict(color="blue")
        ))
        fig.update_layout(
            title=f"{model_choice} Fit on Staking Tiers",
            xaxis_title="Days of Staking",
            yaxis_title="Weight (%)"
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
