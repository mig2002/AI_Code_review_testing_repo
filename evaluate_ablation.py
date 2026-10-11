import os
import sys
import yaml
import argparse
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from scipy import stats

# Ensure root directory is on path
sys.path.insert(0, os.path.abspath("."))

from utils.logger import get_logger
from utils.metrics import compute_binary_metrics
from data.preprocessing import NBaIoTDataPreprocessor
from data.nbaiot_dataset import create_dataloaders, NBaIoTDataset
from data.download_nbaiot import get_nbaiot_feature_names, prepare_nbaiot_data
from models.lstm import LSTMIDS
from federated.client import IoTClient
from federated.fedavg import fedavg_aggregate
from security.model_trust import ModelTrustEvaluator
from security.explanation_trust import ExplanationTrustEvaluator
from security.temporal_trust import TemporalTrustManager
from security.trust_aggregation import (
    compute_all_joint_trust,
    compute_trust_aggregation_weights,
    trust_weighted_fedavg
)
from security.attacks import (
    apply_model_poisoning,
    apply_explanation_manipulation
)
from xai.shap_explainer import SHAPExplainer

logger = get_logger("Step11_Ablation")

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def evaluate_model_on_loader(model: LSTMIDS, test_loader: DataLoader, device: str) -> Dict[str, float]:
    """Evaluates a model over a DataLoader and computes metrics."""
    model.eval()
    dev = torch.device(device)
    model.to(dev)
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(dev)
            logits = model(x_batch)
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy().ravel())
            all_targets.extend(y_batch.numpy().ravel())
            
    return compute_binary_metrics(all_targets, all_preds)

def run_ablation_study(
    config_path: str = "configs/config.yaml",
    device: str = "cuda"
):
    output_dir = "results/ablation"
    os.makedirs(output_dir, exist_ok=True)
    cfg = load_config(config_path)
    feature_names = get_nbaiot_feature_names()
    dev_type = device if torch.cuda.is_available() and device == "cuda" else "cpu"

    logger.info("=" * 95)
    logger.info("   STEP 11 (PART 2): COMPONENT ABLATION STUDY & STATISTICAL SIGNIFICANCE")
    logger.info("=" * 95)

    seeds = [42, 101, 555, 777, 999]
    logger.info(f"Running across 5 independent random seeds: {seeds}")

    # Configurations for component ablation
    ablation_configs = [
        {"name": "Standard FedAvg (No Trust)", "use_trust": False, "alpha": 0.0, "beta": 0.0, "gamma": 0.0},
        {"name": "Model Trust Only (M)",       "use_trust": True,  "alpha": 1.0, "beta": 0.0, "gamma": 0.0},
        {"name": "Explanation Trust Only (E)", "use_trust": True,  "alpha": 0.0, "beta": 1.0, "gamma": 0.0},
        {"name": "Static Joint Trust (M+E)",  "use_trust": True,  "alpha": 0.5, "beta": 0.5, "gamma": 0.0},
        {"name": "Model + Temporal (M+H)",     "use_trust": True,  "alpha": 0.6, "beta": 0.0, "gamma": 0.4},
        {"name": "Explanation + Temporal (E+H)","use_trust": True,"alpha": 0.0, "beta": 0.6, "gamma": 0.4},
        {"name": "Full Proposed (M+E+H)",      "use_trust": True,  "alpha": 0.4, "beta": 0.3, "gamma": 0.3},
    ]

    all_seed_records = []

    for seed_idx, seed in enumerate(seeds):
        logger.info(f"\n=========================================================================")
        logger.info(f"   SEED {seed_idx + 1}/{len(seeds)}: seed={seed}")
        logger.info(f"=========================================================================")

        # 1. Prepare data for this seed
        data_dir = cfg["dataset"]["data_dir"]
        prepare_nbaiot_data(data_dir=data_dir, sample_size=cfg["dataset"].get("sample_size_per_client", 3000), seed=seed)
        
        preprocessor = NBaIoTDataPreprocessor(
            data_dir=data_dir,
            train_ratio=cfg["dataset"]["train_ratio"],
            val_ratio=cfg["dataset"]["val_ratio"],
            test_ratio=cfg["dataset"]["test_ratio"],
            seed=seed
        )
        device_names = cfg["clients"]["device_names"]
        client_data, (global_test_x, global_test_y) = preprocessor.load_and_preprocess_all(device_names)
        
        batch_size = cfg["federated"]["batch_size"]
        global_test_dataset = NBaIoTDataset(global_test_x, global_test_y)
        global_test_loader = DataLoader(global_test_dataset, batch_size=batch_size, shuffle=False)
        
        benign_idx = np.where(global_test_y == 0)[0][:40]
        bg_samples = global_test_x[benign_idx]
        eval_x_sample = global_test_x[:50]

        # Initial weights
        checkpoint_path = "results/global_lstm_step1.pt"
        if os.path.exists(checkpoint_path):
            initial_weights = torch.load(checkpoint_path, map_location="cpu")
        else:
            base_init = LSTMIDS(cfg["model"]["input_dim"], cfg["model"]["hidden_dim"], cfg["model"]["num_layers"], cfg["model"]["dense_dim"]).to(dev_type)
            initial_weights = base_init.get_weights()

        # Clean reference explanation
        ref_model = LSTMIDS(cfg["model"]["input_dim"], cfg["model"]["hidden_dim"], cfg["model"]["num_layers"], cfg["model"]["dense_dim"]).to("cpu")
        ref_model.load_state_dict(initial_weights)
        ref_model.eval()
        ref_explainer = SHAPExplainer(model=ref_model, background_data=bg_samples, feature_names=feature_names, device="cpu")
        ref_shap_vals = ref_explainer.compute_shap_values(eval_x_sample)
        clean_ref_expl = np.mean(np.abs(ref_shap_vals), axis=0)

        # 2 Attackers: Client 7 (Model Poisoning), Client 8 (Decoy XAI)
        attacker_poison_dev = device_names[7]
        attacker_decoy_dev = device_names[8]

        client_weights = {}
        client_sample_counts = {}
        client_models = {}
        client_explanations = {}
        client_eval_x = {}

        for client_id, dev in enumerate(device_names):
            splits = client_data[dev]
            tr_l, va_l, te_l = create_dataloaders(
                splits["x_train"], splits["y_train"], splits["x_val"], splits["y_val"], splits["x_test"], splits["y_test"],
                batch_size=batch_size
            )
            client = IoTClient(
                client_id=client_id,
                device_name=dev,
                train_loader=tr_l,
                val_loader=va_l,
                test_loader=te_l,
                model_config=cfg["model"],
                device=dev_type
            )
            client.receive_global_weights(initial_weights)
            client.train_local(epochs=2, lr=cfg["federated"]["learning_rate"])
            w_i, n_i = client.get_model_update()

            # Apply Model Poisoning on client 7
            if dev == attacker_poison_dev:
                delta_i = {k: (w_i[k].cpu() - initial_weights[k].cpu()) for k in w_i.keys()}
                delta_adv = apply_model_poisoning(delta_i, scale=4.0)
                w_i = {k: (initial_weights[k].cpu() + delta_adv[k]) for k in w_i.keys()}

            client_weights[dev] = w_i
            client_sample_counts[dev] = n_i
            client_models[dev] = client.model
            client_eval_x[dev] = splits["x_test"][:50]

            # Compute SHAP explanation
            c_explainer = SHAPExplainer(model=client.model, background_data=bg_samples, feature_names=feature_names, device="cpu")
            shap_vals = c_explainer.compute_shap_values(splits["x_test"][:50])
            expl_i = np.mean(np.abs(shap_vals), axis=0)

            # Apply Decoy XAI on client 8
            if dev == attacker_decoy_dev:
                expl_i = apply_explanation_manipulation(expl_i, strategy="decoy_swap", top_k=10)

            client_explanations[dev] = expl_i

        # Precompute individual trust evaluations
        deltas = {d: {k: (client_weights[d][k].cpu() - initial_weights[k].cpu()) for k in client_weights[d].keys()} for d in device_names}
        model_eval = ModelTrustEvaluator(w_direction=0.5, w_norm=0.2, w_layer=0.3)
        m_trust_res, _ = model_eval.evaluate_all(deltas)
        m_trust_scores = {d: res["model_trust"] for d, res in m_trust_res.items()}

        expl_eval = ExplanationTrustEvaluator(w_similarity=0.4, w_jaccard=0.3, w_fidelity=0.3, top_k=10)
        e_trust_res, _ = expl_eval.evaluate_all(client_explanations, client_models, client_eval_x, feature_names, device="cpu")
        e_trust_scores = {d: res["explanation_trust"] for d, res in e_trust_res.items()}

        # Evaluate all ablation configurations
        client_weights_list = [client_weights[d] for d in device_names]
        sample_counts_list = [client_sample_counts[d] for d in device_names]

        for config in ablation_configs:
            cfg_name = config["name"]
            use_trust = config["use_trust"]

            if not use_trust:
                # FedAvg
                w_agg = fedavg_aggregate(client_weights_list, sample_counts_list)
                total_samples = sum(sample_counts_list)
                attacker_weight_pct = (client_sample_counts[attacker_poison_dev] + client_sample_counts[attacker_decoy_dev]) / total_samples * 100.0
            else:
                # Initialize TemporalTrustManager with custom alpha, beta, gamma
                alpha = config["alpha"]
                beta = config["beta"]
                gamma = config["gamma"]
                temp_mgr = TemporalTrustManager(lambda_decay=0.7, alpha=alpha, beta=beta, gamma=gamma)
                t_res = temp_mgr.update_round(m_trust_scores, e_trust_scores, round_idx=1)
                final_trust = {d: res["final_trust (T_final)"] for d, res in t_res.items()}

                w_agg = trust_weighted_fedavg(
                    client_weights_list,
                    sample_counts_list,
                    [final_trust[d] for d in device_names]
                )
                norm_weights = compute_trust_aggregation_weights(client_sample_counts, final_trust)
                attacker_weight_pct = (norm_weights[attacker_poison_dev] + norm_weights[attacker_decoy_dev]) * 100.0

            # Evaluate Global Model
            eval_model = LSTMIDS(cfg["model"]["input_dim"], cfg["model"]["hidden_dim"], cfg["model"]["num_layers"], cfg["model"]["dense_dim"]).to(dev_type)
            eval_model.load_state_dict(w_agg)
            metrics = evaluate_model_on_loader(eval_model, global_test_loader, dev_type)

            # Global Model SHAP explanation
            eval_explainer = SHAPExplainer(model=eval_model, background_data=bg_samples, feature_names=feature_names, device="cpu")
            global_shap_vals = eval_explainer.compute_shap_values(eval_x_sample)
            global_expl = np.mean(np.abs(global_shap_vals), axis=0)

            # Cosine Consensus & Jaccard
            norm_g = np.linalg.norm(global_expl)
            norm_ref = np.linalg.norm(clean_ref_expl)
            expl_cosine = float(np.dot(global_expl, clean_ref_expl) / (norm_g * norm_ref)) if norm_g > 0 and norm_ref > 0 else 0.0

            top10_g = set(np.argsort(global_expl)[-10:])
            top10_ref = set(np.argsort(clean_ref_expl)[-10:])
            top10_jaccard = float(len(top10_g.intersection(top10_ref)) / len(top10_g.union(top10_ref)))

            record = {
                "Seed": seed,
                "Configuration": cfg_name,
                "Accuracy (%)": metrics["accuracy"] * 100.0,
                "Precision (%)": metrics["precision"] * 100.0,
                "Recall (%)": metrics["recall"] * 100.0,
                "F1-Score (%)": metrics["f1"] * 100.0,
                "FPR (%)": metrics["fpr"] * 100.0,
                "Attacker Weight Share (%)": attacker_weight_pct,
                "Explanation Cosine": expl_cosine,
                "Top-10 Jaccard": top10_jaccard
            }
            all_seed_records.append(record)

            logger.info(
                f"[{cfg_name:<28}] Acc: {record['Accuracy (%)']:6.2f}% | "
                f"F1: {record['F1-Score (%)']:6.2f}% | "
                f"FPR: {record['FPR (%)']:6.2f}% | "
                f"Atk-Wt: {record['Attacker Weight Share (%)']:5.1f}% | "
                f"SHAP-Cos: {record['Explanation Cosine']:.4f} | "
                f"Jaccard: {record['Top-10 Jaccard']:.4f}"
            )

    df_seeds = pd.DataFrame(all_seed_records)
    csv_raw_path = os.path.join(output_dir, "ablation_study_all_seeds.csv")
    df_seeds.to_csv(csv_raw_path, index=False)

    # ---------------------------------------------------------
    # Compute Mean +/- Standard Error of the Mean (SEM)
    # ---------------------------------------------------------
    agg_funcs = {
        "Accuracy (%)": ["mean", "sem"],
        "Precision (%)": ["mean", "sem"],
        "Recall (%)": ["mean", "sem"],
        "F1-Score (%)": ["mean", "sem"],
        "FPR (%)": ["mean", "sem"],
        "Attacker Weight Share (%)": ["mean", "sem"],
        "Explanation Cosine": ["mean", "sem"],
        "Top-10 Jaccard": ["mean", "sem"],
    }
    df_summary = df_seeds.groupby("Configuration").agg(agg_funcs).reset_index()
    csv_summary_path = os.path.join(output_dir, "ablation_study_results.csv")
    df_summary.to_csv(csv_summary_path, index=False)

    # ---------------------------------------------------------
    # Perform Paired Student's t-tests against Full Proposed
    # ---------------------------------------------------------
    full_cfg_name = "Full Proposed (M+E+H)"
    full_df = df_seeds[df_seeds["Configuration"] == full_cfg_name].sort_values("Seed")

    ttest_records = []
    key_metrics = ["Accuracy (%)", "F1-Score (%)", "FPR (%)", "Attacker Weight Share (%)", "Explanation Cosine"]

    for cfg_item in ablation_configs:
        cfg_name = cfg_item["name"]
        if cfg_name == full_cfg_name:
            continue
            
        comp_df = df_seeds[df_seeds["Configuration"] == cfg_name].sort_values("Seed")
        
        for metric in key_metrics:
            vals_full = full_df[metric].values
            vals_comp = comp_df[metric].values
            
            # Paired Student's t-test
            diff = vals_full - vals_comp
            if np.allclose(diff, 0.0):
                t_stat = 0.0
                p_val = 1.0
            else:
                t_stat, p_val = stats.ttest_rel(vals_full, vals_comp)
                
            sig_star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            
            ttest_records.append({
                "Comparison": f"{full_cfg_name} vs {cfg_name}",
                "Metric": metric,
                "Proposed Mean": float(np.mean(vals_full)),
                "Baseline Mean": float(np.mean(vals_comp)),
                "Mean Delta": float(np.mean(vals_full) - np.mean(vals_comp)),
                "t-statistic": float(t_stat),
                "p-value": float(p_val),
                "Significance": sig_star
            })

    df_ttest = pd.DataFrame(ttest_records)
    csv_ttest_path = os.path.join(output_dir, "statistical_significance_tests.csv")
    df_ttest.to_csv(csv_ttest_path, index=False)
    logger.info(f"\nSaved statistical significance tests to: {csv_ttest_path}")

    # ---------------------------------------------------------
    # Generate Publication-Quality Visualizations
    # ---------------------------------------------------------
    generate_ablation_plots(df_seeds, output_dir)
    return df_seeds, df_ttest

def generate_ablation_plots(df_seeds: pd.DataFrame, output_dir: str):
    """Generates a 4-panel publication-grade ablation bar chart with error bars."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    cfg_order = [
        "Standard FedAvg (No Trust)",
        "Model Trust Only (M)",
        "Explanation Trust Only (E)",
        "Static Joint Trust (M+E)",
        "Model + Temporal (M+H)",
        "Explanation + Temporal (E+H)",
        "Full Proposed (M+E+H)"
    ]
    palette = sns.color_palette("Set2", n_colors=len(cfg_order))

    # Panel 1: Accuracy under Mixed Attack
    sns.barplot(
        data=df_seeds, x="Configuration", y="Accuracy (%)",
        order=cfg_order, ax=axes[0, 0], palette=palette, capsize=0.1, ci=68
    )
    axes[0, 0].set_title("A. Classification Accuracy under Mixed Attacks (Mean ± SEM)", fontweight="bold")
    axes[0, 0].set_xticklabels(cfg_order, rotation=35, ha="right")
    axes[0, 0].set_ylim(0, 105)

    # Panel 2: Attacker Weight Share (%)
    sns.barplot(
        data=df_seeds, x="Configuration", y="Attacker Weight Share (%)",
        order=cfg_order, ax=axes[0, 1], palette=palette, capsize=0.1, ci=68
    )
    axes[0, 1].set_title("B. Attacker Weight Infiltration (Lower is Better)", fontweight="bold")
    axes[0, 1].set_xticklabels(cfg_order, rotation=35, ha="right")
    axes[0, 1].set_ylim(0, 30)

    # Panel 3: False Positive Rate (FPR %)
    sns.barplot(
        data=df_seeds, x="Configuration", y="FPR (%)",
        order=cfg_order, ax=axes[1, 0], palette=palette, capsize=0.1, ci=68
    )
    axes[1, 0].set_title("C. False Positive Rate (FPR) (Lower is Better)", fontweight="bold")
    axes[1, 0].set_xticklabels(cfg_order, rotation=35, ha="right")
    axes[1, 0].set_ylim(0, 105)

    # Panel 4: Explanation Cosine Consensus
    sns.barplot(
        data=df_seeds, x="Configuration", y="Explanation Cosine",
        order=cfg_order, ax=axes[1, 1], palette=palette, capsize=0.1, ci=68
    )
    axes[1, 1].set_title("D. Explanation Integrity under Decoy Attacks (Cosine Consensus)", fontweight="bold")
    axes[1, 1].set_xticklabels(cfg_order, rotation=35, ha="right")
    axes[1, 1].set_ylim(0, 1.1)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "ablation_component_barchart.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ablation figure to: {chart_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 11: Component Ablation Study & Statistical Significance")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda/cpu)")
    args = parser.parse_args()

    run_ablation_study(config_path=args.config, device=args.device)
