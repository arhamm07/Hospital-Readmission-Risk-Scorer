import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.explainability.shap_explainer import SHAPExplainer
from src.features.selector import FeatureSelector

logger = get_logger(__name__, log_file="logs/report_generator.log")


# HTML TEMPLATES
GLOBAL_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Readmission Model — Global Explanation Report</title>
  <style>
    body  {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
    h1    {{ color: #1565C0; }}
    h2    {{ color: #1976D2; border-bottom: 2px solid #1976D2; padding-bottom: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th    {{ background: #1565C0; color: white; padding: 10px; text-align: left; }}
    td    {{ padding: 8px 10px; border: 1px solid #ddd; }}
    tr:nth-child(even) {{ background: #f5f5f5; }}
    .metric-box {{
      display: inline-block; background: #E3F2FD;
      border: 1px solid #1565C0; border-radius: 8px;
      padding: 12px 20px; margin: 8px; text-align: center;
    }}
    .metric-val {{ font-size: 24px; font-weight: bold; color: #1565C0; }}
    .metric-lbl {{ font-size: 13px; color: #555; }}
    img {{ max-width: 100%; margin: 12px 0; border: 1px solid #ddd; border-radius: 4px; }}
    .increases {{ color: #c62828; font-weight: bold; }}
    .decreases {{ color: #2e7d32; font-weight: bold; }}
  </style>
</head>
<body>
  <h1>🏥 Hospital Readmission Risk — Model Explanation Report</h1>
  <p><b>Model:</b> XGBoost + Isotonic Calibration &nbsp;|&nbsp;
     <b>Dataset:</b> Diabetes 130-US Hospitals (UCI) &nbsp;|&nbsp;
     <b>Generated:</b> {timestamp}</p>

  <h2>📊 Model Performance</h2>
  <div>
    {metric_boxes}
  </div>

  <h2>🔍 Top 20 Global SHAP Features</h2>
  <p>Mean absolute SHAP value — higher means the feature has more
  influence on predictions across all patients.</p>
  {feature_table}

  <h2>📈 SHAP Visualisations</h2>
  <h3>Feature Importance (Bar)</h3>
  <img src="../reports/shap_summary_bar.png" />
  <h3>Impact Distribution (Beeswarm)</h3>
  <img src="../reports/shap_beeswarm.png" />
</body>
</html>
"""

PATIENT_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Patient Explanation — {patient_id}</title>
  <style>
    body  {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
    h1    {{ color: #1565C0; }}
    h2    {{ color: #1976D2; border-bottom: 2px solid #1976D2; }}
    .risk-badge {{
      display: inline-block; padding: 8px 20px;
      border-radius: 20px; font-size: 20px; font-weight: bold;
      color: white; background: {risk_color};
    }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th    {{ background: #1565C0; color: white; padding: 10px; text-align: left; }}
    td    {{ padding: 8px 10px; border: 1px solid #ddd; }}
    tr:nth-child(even) {{ background: #f5f5f5; }}
    .bar-pos {{
      display: inline-block; height: 16px; background: #c62828; border-radius: 2px;
    }}
    .bar-neg {{
      display: inline-block; height: 16px; background: #2e7d32; border-radius: 2px;
    }}
    .increases {{ color: #c62828; font-weight: bold; }}
    .decreases {{ color: #2e7d32; font-weight: bold; }}
    img {{ max-width: 100%; margin: 12px 0; }}
  </style>
</head>
<body>
  <h1>🏥 Patient Readmission Risk Explanation</h1>
  <p><b>Patient ID:</b> {patient_id} &nbsp;|&nbsp;
     <b>Generated:</b> {timestamp}</p>

  <h2>Risk Score</h2>
  <p>
    <span class="risk-badge">{risk_tier} — {risk_score:.1%}</span>
    &nbsp; Base rate: {base_value:.1%}
  </p>
  <p><i>Score interpretation: probability of readmission within 30 days.</i></p>

  <h2>🔍 Top Factors Driving This Prediction</h2>
  <table>
    <tr>
      <th>#</th>
      <th>Feature</th>
      <th>Patient Value</th>
      <th>Impact</th>
      <th>Direction</th>
      <th>Visual</th>
    </tr>
    {factor_rows}
  </table>

  <h2>📊 SHAP Waterfall</h2>
  <img src="{waterfall_path}" />
</body>
</html>
"""


# REPORT GENERATOR CLASS
class ExplanationReportGenerator:
    """
    Generates global model report + per-patient HTML reports.
    """

    def __init__(self, cfg):
        self.cfg     = cfg
        self.reports = Path("reports")
        self.reports.mkdir(exist_ok=True)

    # Global report

    def generate_global_report(
        self,
        shap_explainer: SHAPExplainer,
        model_metrics:  dict,
        save_path: str = "reports/global_explanation.html",
    ) -> str:
        from datetime import datetime

        # Metric boxes
        key_metrics = {
            "AUC-ROC":   f"{model_metrics.get('test_auc_roc', 0):.3f}",
            "AUC-PR":    f"{model_metrics.get('test_auc_pr', 0):.3f}",
            "ECE":       f"{model_metrics.get('test_ece', 0):.4f}",
            "KS Stat":   f"{model_metrics.get('test_ks_stat', 0):.3f}",
            "Gini":      f"{model_metrics.get('test_gini', 0):.3f}",
            "NNT":       f"{model_metrics.get('test_nnt', 0):.1f}",
        }
        boxes_html = "".join([
            f'<div class="metric-box">'
            f'<div class="metric-val">{v}</div>'
            f'<div class="metric-lbl">{k}</div>'
            f'</div>'
            for k, v in key_metrics.items()
        ])

        # Feature importance table
        if hasattr(shap_explainer, "global_importance"):
            gi = shap_explainer.global_importance.head(20)
            rows = "".join([
                f"<tr><td>{i+1}</td>"
                f"<td>{row['feature']}</td>"
                f"<td>{row['mean_abs_shap']:.4f}</td></tr>"
                for i, (_, row) in enumerate(gi.iterrows())
            ])
            feat_table = (
                "<table><tr><th>Rank</th>"
                "<th>Feature</th><th>Mean |SHAP|</th></tr>"
                + rows + "</table>"
            )
        else:
            feat_table = "<p>Run explain_global() first.</p>"

        html = GLOBAL_HTML.format(
            timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M"),
            metric_boxes = boxes_html,
            feature_table= feat_table,
        )

        Path(save_path).parent.mkdir(exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Global explanation report saved → {save_path}")
        return save_path

    # Per-patient report

    def generate_patient_report(
        self,
        explanation:  dict,
        save_dir:     str = "reports/patients",
    ) -> str:
        from datetime import datetime

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        patient_id = explanation["patient_id"]
        risk_score = explanation["predicted_risk"]

        # Risk tier + colour
        tiers = self.cfg.risk_tiers.thresholds
        if risk_score < tiers.low:
            tier, color = "Low Risk",      "#2e7d32"
        elif risk_score < tiers.medium:
            tier, color = "Medium Risk",   "#f57f17"
        elif risk_score < tiers.high:
            tier, color = "High Risk",     "#e65100"
        else:
            tier, color = "Critical Risk", "#b71c1c"

        # Factor rows
        max_impact = max(
            abs(f["shap_impact"])
            for f in explanation["top_factors"]
        ) or 1.0

        rows_html = ""
        for i, factor in enumerate(explanation["top_factors"]):
            bar_width = int(
                abs(factor["shap_impact"]) / max_impact * 150
            )
            bar_class = (
                "bar-pos" if factor["direction"] == "increases_risk"
                else "bar-neg"
            )
            dir_class = (
                "increases" if factor["direction"] == "increases_risk"
                else "decreases"
            )
            rows_html += (
                f"<tr>"
                f"<td>{i+1}</td>"
                f"<td>{factor['feature']}</td>"
                f"<td>{factor['value']}</td>"
                f"<td>{factor['shap_impact']:+.4f}</td>"
                f"<td class='{dir_class}'>"
                f"{'↑ Increases' if factor['direction']=='increases_risk' else '↓ Decreases'} risk"
                f"</td>"
                f"<td><span class='{bar_class}' style='width:{bar_width}px'></span></td>"
                f"</tr>"
            )

        waterfall = explanation.get(
            "waterfall_plot",
            "reports/shap_waterfall_default.png"
        )

        html = PATIENT_HTML.format(
            patient_id     = patient_id,
            timestamp      = datetime.now().strftime("%Y-%m-%d %H:%M"),
            risk_score     = risk_score,
            risk_tier      = tier,
            risk_color     = color,
            base_value     = explanation["base_value"],
            factor_rows    = rows_html,
            waterfall_path = f"../{waterfall}",
        )

        save_path = str(Path(save_dir) / f"{patient_id}_explanation.html")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Patient report saved → {save_path}")
        return save_path


# STANDALONE RUNNER
def main():
    cfg = load_config()

    mlflow.set_tracking_uri(cfg.project.mlflow_tracking_url)
    mlflow.set_experiment(cfg.project.mlflow_experiment)

    # Load artefacts
    proc   = Path(cfg.data.processed_dir)
    TARGET = cfg.data.target_binary_col

    train_df = pd.read_parquet(proc / "train.parquet")
    test_df  = pd.read_parquet(proc / "test.parquet")

    selected = FeatureSelector.load(str(proc / "selected_features.json"))
    selected = [f for f in selected if f in train_df.columns]

    X_train = train_df[selected].values
    X_test  = test_df[selected].values
    y_test  = test_df[TARGET].values

    # Load base XGBoost model (not calibrated — for SHAP TreeExplainer)
    import xgboost as xgb
    base_model = xgb.XGBClassifier()
    base_model.load_model("models/xgb_base.ubj")

    # Load metrics
    metrics_path = Path("reports/metrics.json")
    model_metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            model_metrics = json.load(f)

    # Build SHAP explainer
    explainer = SHAPExplainer(base_model, selected, cfg)
    explainer.fit(X_train)

    # Global explanations
    with mlflow.start_run(run_name="shap_explainability"):
        saved = explainer.explain_global(X_test, save_prefix="shap")
        for path in saved.values():
            mlflow.log_artifact(path)

        # Local explanation for highest-risk patient
        test_probs = base_model.predict_proba(X_test)[:, 1]
        high_risk_idx = int(np.argmax(test_probs))

        local_exp = explainer.explain_local(
            X_test[high_risk_idx],
            patient_id=f"patient_{high_risk_idx}",
            save_plot=True,
        )
        logger.info(f"High-risk patient explanation:\n{json.dumps(local_exp, indent=2)}")

        # Generate HTML reports
        generator = ExplanationReportGenerator(cfg)

        global_report = generator.generate_global_report(
            explainer, model_metrics
        )
        patient_report = generator.generate_patient_report(local_exp)

        mlflow.log_artifact(global_report)
        mlflow.log_artifact(patient_report)

    logger.info("Explainability pipeline complete")


if __name__ == "__main__":
    main()