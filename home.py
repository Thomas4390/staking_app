"""
Home.py

Landing page of the Streamlit application.
"""

import streamlit as st

def main():
    st.title("Staking Pools Distribution Fit Application")
    st.write(
        """
        Welcome to the demo application for fitting and visualizing distribution
        functions for staking pools.

        Navigate through the following features:

        - **Staking Distribution Fit**:
          Define weights for each tier, choose a model type (Polynomial, Exponential,
          Logarithmic, Power Law, or Sigmoid) and see the fitted curve, formula, and
          distribution table.
        - **Custom Fit & MSE**:
          Enter your own function expression and compare it to your normalized weights
          for the same 8 tiers.
        """
    )

if __name__ == "__main__":
    main()
