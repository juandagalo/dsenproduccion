"""Streamlit demo app for churn prediction."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src/ to path so inference module can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import plotly.express as px  # type: ignore[import-untyped]
import streamlit as st

from inference.predict import (
    build_customer_dataframe,
    get_risk_level,
    load_evaluation_results,
    load_model,
    load_serving_config,
    predict_churn,
)

RISK_COLORS = {"Low": "green", "Medium": "orange", "High": "red"}


@st.cache_data
def cached_config() -> dict:  # type: ignore[type-arg]
    """Load and cache the serving configuration."""
    config: dict = load_serving_config()  # type: ignore[type-arg]
    return config


@st.cache_resource
def cached_model(model_path: str) -> object:
    """Load and cache the trained pipeline."""
    return load_model(model_path)


@st.cache_data
def cached_eval_results(results_path: str) -> dict:  # type: ignore[type-arg]
    """Load and cache the evaluation results."""
    results: dict = load_evaluation_results(results_path)  # type: ignore[type-arg]
    return results


def render_individual_tab(config: dict, pipeline: object, eval_results: dict) -> None:  # type: ignore[type-arg]
    """Render the Individual Prediction tab."""
    threshold = eval_results["chosen_threshold"]
    risk_cfg = config["risk_levels"]
    input_cols = config["input_columns"]

    with st.form("customer_form"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Account")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly = st.slider("Monthly Charges ($)", 18.0, 118.0, 50.0, step=0.5)

        with col2:
            st.subheader("Demographics")
            senior = st.toggle("Senior Citizen")
            partner = st.toggle("Partner")
            dependents = st.toggle("Dependents")
            paperless = st.toggle("Paperless Billing")

        with col3:
            st.subheader("Services")
            services: dict[str, bool] = {}
            for svc in input_cols["services"]:
                label = svc.replace("_", " ")
                services[svc] = st.toggle(label, key=f"svc_{svc}")

        with col4:
            st.subheader("Contract")
            contract = st.selectbox(
                "Contract",
                input_cols["categorical"]["Contract"],
            )
            internet = st.selectbox(
                "Internet Service",
                input_cols["categorical"]["InternetService"],
            )
            payment = st.selectbox(
                "Payment Method",
                input_cols["categorical"]["PaymentMethod"],
            )

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        customer_data = {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "PaperlessBilling": paperless,
            "Contract": contract,
            "InternetService": internet,
            "PaymentMethod": payment,
        }
        for svc_name, svc_val in services.items():
            customer_data[svc_name] = "Yes" if svc_val else "No"

        df = build_customer_dataframe(customer_data)
        result = predict_churn(pipeline, df, threshold)  # type: ignore[arg-type]

        prob = float(result["churn_probability"].iloc[0])
        pred = int(result["churn_predicted"].iloc[0])
        risk = get_risk_level(prob, risk_cfg["low_max"], risk_cfg["medium_max"])

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn Probability", f"{prob:.1%}")
        m2.metric("Prediction", "Churn" if pred == 1 else "No Churn")
        m3.markdown(f"**Risk Level:** :{RISK_COLORS[risk]}[{risk}]")


def render_batch_tab(config: dict, pipeline: object, eval_results: dict) -> None:  # type: ignore[type-arg]
    """Render the Batch Prediction tab."""
    threshold = eval_results["chosen_threshold"]

    sample_path = Path(__file__).parent / "sample_customers.csv"
    with open(sample_path, "rb") as f:
        sample_csv = f.read()
    st.download_button(
        "Download sample CSV template",
        data=sample_csv,
        file_name="sample_customers.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload a CSV file with customer data", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)

        input_cols = config["input_columns"]
        expected = (
            input_cols["numeric"]
            + input_cols["binary"]
            + input_cols["services"]
            + list(input_cols["categorical"].keys())
        )
        missing = [c for c in expected if c not in batch_df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            return

        results = predict_churn(
            pipeline,
            batch_df[expected],
            threshold,  # type: ignore[arg-type]
        )

        churn_count = int(results["churn_predicted"].sum())
        total = len(results)
        no_churn_count = total - churn_count
        churn_rate = churn_count / total if total > 0 else 0.0

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Customers", total)
        m2.metric("Predicted Churn", churn_count)
        m3.metric("Churn Rate", f"{churn_rate:.1%}")

        churn_df = results[results["churn_predicted"] == 1]
        no_churn_df = results[results["churn_predicted"] == 0]

        tab_churn, tab_no_churn, tab_all = st.tabs(
            [
                f"Likely to Churn ({churn_count})",
                f"Likely to Stay ({no_churn_count})",
                f"All Customers ({total})",
            ]
        )

        with tab_churn:
            if churn_df.empty:
                st.info("No customers predicted to churn.")
            else:
                st.dataframe(
                    churn_df.sort_values("churn_probability", ascending=False),
                    use_container_width=True,
                )

        with tab_no_churn:
            if no_churn_df.empty:
                st.info("All customers predicted to churn.")
            else:
                st.dataframe(
                    no_churn_df.sort_values("churn_probability", ascending=False),
                    use_container_width=True,
                )

        with tab_all:
            st.dataframe(
                results.sort_values("churn_probability", ascending=False),
                use_container_width=True,
            )

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Results CSV",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv",
        )


def render_insights_tab(eval_results: dict) -> None:  # type: ignore[type-arg]
    """Render the Model Insights tab."""
    metrics = eval_results["metrics_at_chosen_threshold"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{eval_results['roc_auc']:.3f}")
    m2.metric("PR-AUC", f"{eval_results['pr_auc']:.3f}")
    m3.metric("F1 Score", f"{metrics['f1']:.3f}")
    m4.metric("Threshold", f"{eval_results['chosen_threshold']}")

    st.subheader("Top 15 Feature Coefficients")
    coef_df = pd.DataFrame(eval_results["coefficients"][:15])
    coef_df["sign"] = coef_df["coefficient"].apply(
        lambda x: "Increases Churn" if x > 0 else "Decreases Churn"
    )
    fig = px.bar(
        coef_df,
        x="coefficient",
        y="feature",
        color="sign",
        orientation="h",
        color_discrete_map={
            "Increases Churn": "#EF553B",
            "Decreases Churn": "#636EFA",
        },
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=450,
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Threshold Comparison")
    comparison = pd.DataFrame(
        {
            "Threshold": [0.50, eval_results["chosen_threshold"]],
            "Precision": [
                eval_results["metrics_at_default_05"]["precision"],
                metrics["precision"],
            ],
            "Recall": [
                eval_results["metrics_at_default_05"]["recall"],
                metrics["recall"],
            ],
            "F1": [
                eval_results["metrics_at_default_05"]["f1"],
                metrics["f1"],
            ],
        }
    )
    st.dataframe(
        comparison.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    """Entry point for the Streamlit app."""
    st.set_page_config(
        page_title="Churn Prediction",
        page_icon=":chart_with_downwards_trend:",
        layout="wide",
    )
    st.title("Churn Prediction Demo")

    config = cached_config()
    pipeline = cached_model(config["model_path"])
    eval_results = cached_eval_results(config["evaluation_results_path"])

    tab1, tab2, tab3 = st.tabs(["Individual Prediction", "Batch Prediction", "Model Insights"])

    with tab1:
        render_individual_tab(config, pipeline, eval_results)

    with tab2:
        render_batch_tab(config, pipeline, eval_results)

    with tab3:
        render_insights_tab(eval_results)


if __name__ == "__main__":
    main()
