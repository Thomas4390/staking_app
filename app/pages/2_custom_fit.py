"""
2_Custom_Fit.py

Allows the user to input raw weights for each staking tier (normalized to 100%)
and a custom function expression in Python (using 'x' for the days array).
It computes a chosen fit metric (MSE or MAE) between the normalized weights
and the output of the custom function. We then clamp (if desired), normalize the 8 fitted points
so they sum to 100, and display a distribution table with totals, the total tokens, the metric in LaTeX, etc.

Adjustments from the previous version:
 - Default custom function is now (0.03)*x**(1.3).
 - After optional clamp, we normalize the custom function outputs so they sum to 100.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils import eval_custom_function, mean_squared_error, mean_absolute_error

def main():
    st.title("Custom Fit & Fit Metric")
    st.write(
        """
        Enter raw weights for each of the 8 staking tiers below. 
        These weights will be normalized to sum to 100%. 
        Then, input your custom function expression using `x` 
        (days array: [1, 3, 7, 14, 30, 90, 180, 365]).
        
        Choose a fit metric (MSE or MAE) and click "Compute Fit Metric" to compare
        your normalized weights with the function output. We'll clamp if desired,
        normalize so the custom outputs sum to 100, and then display:
         - Distribution table (with totals),
         - Total tokens,
         - The chosen metric in LaTeX,
         - An interactive Plotly chart.
        """
    )

    tiers = np.array([1, 3, 7, 14, 30, 90, 180, 365], dtype=int)
    nb_tiers = len(tiers)

    default_raw_weights = [1.0, 1.2, 1.5, 2.0, 6.0, 24.0, 60.0, 150.0]
    raw_weights = []
    for i, day in enumerate(tiers):
        val = st.number_input(
            f"Weight for {day} days",
            min_value=0.0,
            max_value=10000.0,
            value=default_raw_weights[i],
            step=0.1
        )
        raw_weights.append(val)

    sum_raw = sum(raw_weights)
    if sum_raw == 0:
        st.warning("All input weights are zero. Please input some values.")
        return

    normalized_weights = [(w / sum_raw) * 100 for w in raw_weights]

    st.subheader("Custom Function")
    st.write("Use the variable `x` to refer to the days array.")
    # New default custom function
    user_function_expr = st.text_input(
        "Custom function (e.g., 0.03*x**1.3)",
        value="(0.03)*x**(1.3)"
    )

    metric_options = ["MSE", "MAE"]
    metric_choice = st.selectbox("Select fit metric", metric_options, index=0)

    clamp_negatives = st.checkbox("Clamp negative outputs to a small positive value?", value=False)

    if st.button("Compute Fit Metric"):
        try:
            x_data = tiers.astype(float)
            y_true = np.array(normalized_weights)
            y_pred = eval_custom_function(x_data, user_function_expr)

            # Optional clamp
            if clamp_negatives:
                y_pred = np.clip(y_pred, 1e-9, None)

            # Then normalize so sum = 100
            y_pred_sum = np.sum(y_pred)
            if y_pred_sum > 0:
                y_pred = (y_pred / y_pred_sum) * 100

            if metric_choice == "MSE":
                fit_metric = mean_squared_error(y_true, y_pred)
                metric_name = "MSE"
            else:
                fit_metric = mean_absolute_error(y_true, y_pred)
                metric_name = "MAE"

            total_tokens = 3.46e9
            user_tokens = (y_true / 100) * total_tokens
            custom_tokens = (y_pred / 100) * total_tokens

            df = pd.DataFrame({
                "Tier": [f"Tier {i+1}" for i in range(nb_tiers)],
                "Days": tiers,
                "User Weights (%)": y_true,
                "User Tokens": user_tokens,
                "Custom Func (%)": y_pred,
                "Custom Tokens": custom_tokens
            })

            # Totals row
            df.loc[len(df)] = {
                "Tier": "Total",
                "Days": "",
                "User Weights (%)": np.sum(y_true),
                "User Tokens": np.sum(user_tokens),
                "Custom Func (%)": np.sum(y_pred),
                "Custom Tokens": np.sum(custom_tokens)
            }

            st.subheader("Fit Metric")
            st.latex(fr"{metric_name} = {fit_metric:.5g}")

            st.subheader("Total Tokens")
            st.write(f"**{total_tokens:,.0f} tokens**")

            st.write("### Distribution Table")
            st.dataframe(df.style.format({
                "User Weights (%)": "{:.2f}",
                "User Tokens": "{:,.0f}",
                "Custom Func (%)": "{:.2f}",
                "Custom Tokens": "{:,.0f}"
            }))

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=tiers,
                y=y_true,
                mode="markers",
                name="User (Normalized)",
                marker=dict(color="red", size=8)
            ))
            fig.add_trace(go.Scatter(
                x=tiers,
                y=y_pred,
                mode="lines+markers",
                name="Custom Function (Normalized)",
                line=dict(color="blue")
            ))
            fig.update_layout(
                title="Comparison of Custom Function vs User Weights",
                xaxis_title="Days of Staking",
                yaxis_title="Weight (%)"
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error evaluating the custom function: {e}")


if __name__ == "__main__":
    main()
