import streamlit as st
import pandas as pd
import time, random, io
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import shap

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ----------------------------
# Helper functions
# ----------------------------

def benchmark_models(df, target_column):
    """Run multiple ML models and return performance metrics."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "LightGBM": lgb.LGBMClassifier()
    }

    results = []
    model_objects = {}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        runtime = round(time.time() - start, 3)

        try:
            auc = roc_auc_score(y_test, preds)
        except:
            auc = 0.0

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, preds, average="weighted", zero_division=0),
            "F1": f1_score(y_test, preds, average="weighted"),
            "ROC-AUC": auc,
            "Runtime (s)": runtime
        })
        model_objects[name] = (model, X_test, y_test)

    return pd.DataFrame(results), model_objects


def pick_best_model(results_df):
    """Pick the best model based on Accuracy, then F1, then Runtime."""
    results_df = results_df.sort_values(
        by=["Accuracy", "F1", "Runtime (s)"], ascending=[False, False, True]
    )
    return results_df.iloc[0]


def simulate_failover():
    """Simulate multi-cloud failover sequence."""
    providers = ["AWS", "GCP", "Azure"]
    fail_sequence = []
    for cloud in providers:
        if random.random() < 0.3:  # 30% chance a provider "fails"
            fail_sequence.append((cloud, "❌ Failed"))
        else:
            fail_sequence.append((cloud, "✅ Success"))
            return cloud, fail_sequence
    return "Local Backup", fail_sequence


def simulate_costs_and_carbon():
    """Simulate cost and carbon footprint per run."""
    cost = round(random.uniform(0.10, 2.0), 2)  # $
    carbon = round(random.uniform(0.05, 0.5), 3)  # kg CO2
    return cost, carbon


def generate_summary(best_model, cloud, results_df, cost, carbon):
    """Generate an executive-style summary."""
    summary = f"""
    ✅ Best Model: **{best_model['Model']}**  
    - Accuracy: {best_model['Accuracy']:.2f}, F1: {best_model['F1']:.2f}  
    - Runtime: {best_model['Runtime (s)']}s  

    🌐 Deployed on: **{cloud}**  
    💰 Estimated Cost: ${cost}  
    🌍 Carbon Footprint: {carbon} kg CO₂  

    📊 Benchmark Overview (Top 3):  
    {results_df.head(3).to_dict(orient="records")}
    """
    return summary


def agentic_conversation(best_model, cloud):
    """Two agents discuss decisions."""
    convo = f"""
    **Data Scientist Agent 🤖**: After benchmarking, I found **{best_model['Model']}** had the best accuracy/F1.  

    **Cloud Ops Agent ☁️**: Understood. I'll deploy it on **{cloud}** because it passed availability checks.  

    **Data Scientist Agent 🤖**: Great. That balances accuracy, runtime, and resilience.  
    """
    return convo


def plot_confusion(model, X_test, y_test):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


def explain_features(model, X):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        st.subheader("🔍 Feature Importance (SHAP)")
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.info("Explainability not available for this model.")


def export_pdf(summary_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = [Paragraph("Executive Report", styles["Title"]), Spacer(1, 12)]
    for line in summary_text.split("\n"):
        if line.strip():
            story.append(Paragraph(line, styles["Normal"]))
            story.append(Spacer(1, 8))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="AI R&D Multi-Cloud Orchestrator", layout="wide")

st.title("🌐 AI R&D Multi-Cloud Orchestrator")
st.markdown("A demo platform that benchmarks models, selects best-fit AI + Cloud, explains decisions, and generates executive reports.")

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Data Preview:", df.head())

    target_column = st.selectbox("🎯 Select target column (label)", df.columns)

    if st.button("🚀 Run Benchmark"):
        with st.spinner("Benchmarking models..."):
            results_df, model_objects = benchmark_models(df, target_column)
            st.success("✅ Benchmarking complete!")

            st.subheader("📊 Benchmark Results")
            st.dataframe(results_df)

            best_model = pick_best_model(results_df)
            cloud, failover = simulate_failover()
            cost, carbon = simulate_costs_and_carbon()

            st.subheader("🏆 Best Model Selection")
            st.write(f"**{best_model['Model']}** deployed on **{cloud}**")

            st.subheader("🛡️ Failover Simulation")
            for step in failover:
                st.write(f"{step[0]} → {step[1]}")

            st.subheader("📑 Executive Summary")
            summary_text = generate_summary(best_model, cloud, results_df, cost, carbon)
            st.markdown(summary_text)

            st.download_button(
                "📥 Download Executive Report (PDF)",
                data=export_pdf(summary_text),
                file_name="executive_report.pdf",
                mime="application/pdf"
            )

            st.subheader("🤝 Agentic Collaboration")
            st.markdown(agentic_conversation(best_model, cloud))

            # Confusion matrix + explainability
            model, X_test, y_test = model_objects[best_model["Model"]]
            st.subheader("📉 Confusion Matrix")
            plot_confusion(model, X_test, y_test)

            explain_features(model, X_test)
else:
    st.info("Please upload a CSV file to begin.")
