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
    apply_label_flipping,
    apply_sign_flipping,
    apply_model_poisoning,
    inject_backdoor_trigger,
    compute_attack_success_rate,
    apply_explanation_manipulation
)
from xai.shap_explainer import SHAPExplainer

logger = get_logger("Step8_Attacks")

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_attack_evaluation_suite(
    config_path: str = "configs/config.yaml",
    device: str = "cuda"
):
    output_dir = "results/attacks"
    os.makedirs(output_dir, exist_ok=True)
    cfg = load_config(config_path)
    seed = cfg["experiment"].get("seed", 42)
    feature_names = get_nbaiot_feature_names()
    dev_type = device if torch.cuda.is_available() and device == "cuda" else "cpu"

    logger.info("=" * 90)
    logger.info("   STEP 8: COMPREHENSIVE SECURITY ATTACKS & DEFENSE BENCHMARK")
    logger.info("=" * 90)

    # 1. Prepare Data
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
    
    # Clean attack samples for ASR testing (true attack packets)
    attack_test_indices = np.where(global_test_y == 1)[0]
    clean_attack_test_x = global_test_x[attack_test_indices]
    
    # Reference baseline background for SHAP
    benign_idx = np.where(global_test_y == 0)[0][:40]
    bg_samples = global_test_x[benign_idx]

    # ---------------------------------------------------------
    # PART 1: Attack Types Benchmark (Under 20% Attacker Ratio)
    # ---------------------------------------------------------
    # 9 Clients total: 7 Honest, 2 Attackers (devices 7 and 8)
    attack_scenarios = [
        "No_Attack (Clean)",
        "Label_Flipping",
        "Sign_Flipping",
        "Model_Poisoning",
        "Backdoor_Attack",
        "Explanation_Manipulation"
    ]

    benchmark_records = []
    logger.info("\n>>> PART 1: Evaluating Defenses Across 5 Distinct Attack Scenarios...")

    for scenario in attack_scenarios:
        logger.info(f"\n--- Testing Scenario: [{scenario}] ---")
        
        # Fresh base model
        base_model = LSTMIDS(
            input_dim=cfg["model"]["input_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            num_layers=cfg["model"]["num_layers"],
            dense_dim=cfg["model"]["dense_dim"]
        ).to(dev_type)
        current_weights = base_model.get_weights()
        
        client_weights = {}
        client_sample_counts = {}
        client_models = {}
        client_explanations = {}
        client_eval_x = {}
        
        # 7 Honest Clients, 2 Attackers
        attacker_clients = [device_names[7], device_names[8]]
        
        for client_id, dev in enumerate(device_names):
            splits = client_data[dev]
            is_attacker = (dev in attacker_clients and scenario != "No_Attack (Clean)")
            
            x_tr = splits["x_train"]
            y_tr = splits["y_train"]
            
            # Apply data-level attacks
            if is_attacker and scenario == "Label_Flipping":
                y_tr = apply_label_flipping(y_tr)
            elif is_attacker and scenario == "Backdoor_Attack":
                x_tr, y_tr, _ = inject_backdoor_trigger(x_tr, y_tr, trigger_features=[0, 1, 2], trigger_val=5.0, target_label=0, poison_ratio=0.5)
                
            train_l, val_l, test_l = create_dataloaders(x_tr, y_tr, splits["x_val"], splits["y_val"], splits["x_test"], splits["y_test"], batch_size=batch_size)
            
            client = IoTClient(
                client_id=client_id,
                device_name=dev,
                train_loader=train_l,
                val_loader=val_l,
                test_loader=test_l,
                model_config=cfg["model"],
                device=dev_type
            )
            client.receive_global_weights(current_weights)
            client.train_local(epochs=2, lr=cfg["federated"]["learning_rate"])
            w_i, n_i = client.get_model_update()
            
            # Apply model-level attacks
            if is_attacker:
                delta_i = {k: (w_i[k].cpu() - current_weights[k].cpu()) for k in w_i.keys()}
                if scenario == "Sign_Flipping":
                    delta_adv = apply_sign_flipping(delta_i, scale=2.0)
                    w_i = {k: (current_weights[k].cpu() + delta_adv[k]) for k in w_i.keys()}
                elif scenario == "Model_Poisoning":
                    delta_adv = apply_model_poisoning(delta_i, scale=4.0)
                    w_i = {k: (current_weights[k].cpu() + delta_adv[k]) for k in w_i.keys()}
                    
            client_weights[dev] = w_i
            client_sample_counts[dev] = n_i
            client_models[dev] = client.model
            client_eval_x[dev] = splits["x_test"][:50]
            
            # Compute client SHAP explanations
            c_explainer = SHAPExplainer(model=client.model, background_data=bg_samples, feature_names=feature_names, device="cpu")
            shap_vals = c_explainer.compute_shap_values(splits["x_test"][:50])
            expl_i = np.mean(np.abs(shap_vals), axis=0)
            
            # Apply explanation manipulation attack
            if is_attacker and scenario == "Explanation_Manipulation":
                expl_i = apply_explanation_manipulation(expl_i, strategy="decoy_swap", top_k=10)
                
            client_explanations[dev] = expl_i

        # ---------------------------------------------------------
        # Branch 1: Standard FedAvg (Blind to attacks)
        # ---------------------------------------------------------
        w_fedavg = fedavg_aggregate([client_weights[d] for d in device_names], [client_sample_counts[d] for d in device_names])
        m_fedavg = LSTMIDS(cfg["model"]["input_dim"], cfg["model"]["hidden_dim"], cfg["model"]["num_layers"], cfg["model"]["dense_dim"]).to(dev_type)
        m_fedavg.load_state_dict(w_fedavg)
        m_fedavg.eval()
        
        preds_f = []
        with torch.no_grad():
            for x_b, _ in global_test_loader:
                probs = torch.sigmoid(m_fedavg(x_b.to(dev_type)))
                preds_f.extend(probs.cpu().numpy().ravel())
        metrics_fedavg = compute_binary_metrics(global_test_y, preds_f)
        asr_fedavg = compute_attack_success_rate(m_fedavg, clean_attack_test_x, trigger_features=[0, 1, 2], trigger_val=5.0, target_label=0, device="cpu") if scenario == "Backdoor_Attack" else 0.0

        # ---------------------------------------------------------
        # Branch 2: Original FedXAI (FedAvg + post-hoc SHAP without Trust)
        # ---------------------------------------------------------
        # Same model aggregation as FedAvg, but reviewer checks post-hoc SHAP
        metrics_fedxai = dict(metrics_fedavg)
        asr_fedxai = asr_fedavg

        # ---------------------------------------------------------
        # Branch 3: Proposed Trust-Aware Framework (Tri-Component Trust)
        # ---------------------------------------------------------
        deltas = {d: {k: (client_weights[d][k].cpu() - current_weights[k].cpu()) for k in client_weights[d].keys()} for d in device_names}
        
        model_eval = ModelTrustEvaluator(w_direction=0.5, w_norm=0.2, w_layer=0.3)
        m_trust_res, _ = model_eval.evaluate_all(deltas)
        m_trust_scores = {d: res["model_trust"] for d, res in m_trust_res.items()}
        
        expl_eval = ExplanationTrustEvaluator(w_similarity=0.4, w_jaccard=0.3, w_fidelity=0.3, top_k=10)
        e_trust_res, _ = expl_eval.evaluate_all(client_explanations, client_models, client_eval_x, feature_names, device="cpu")
        e_trust_scores = {d: res["explanation_trust"] for d, res in e_trust_res.items()}
        
        temp_mgr = TemporalTrustManager(lambda_decay=0.7, alpha=0.4, beta=0.3, gamma=0.3)
        t_res = temp_mgr.update_round(m_trust_scores, e_trust_scores, round_idx=1)
        final_trust_scores = {d: res["final_trust (T_final)"] for d, res in t_res.items()}
        
        w_proposed = trust_weighted_fedavg(
            client_weights=[client_weights[d] for d in device_names],
            client_sample_counts=[client_sample_counts[d] for d in device_names],
            trust_scores=[final_trust_scores[d] for d in device_names],
            rejection_threshold=0.20
        )
        m_proposed = LSTMIDS(cfg["model"]["input_dim"], cfg["model"]["hidden_dim"], cfg["model"]["num_layers"], cfg["model"]["dense_dim"]).to(dev_type)
        m_proposed.load_state_dict(w_proposed)
        m_proposed.eval()
        
        preds_p = []
        with torch.no_grad():
            for x_b, _ in global_test_loader:
                probs = torch.sigmoid(m_proposed(x_b.to(dev_type)))
                preds_p.extend(probs.cpu().numpy().ravel())
        metrics_proposed = compute_binary_metrics(global_test_y, preds_p)
        asr_proposed = compute_attack_success_rate(m_proposed, clean_attack_test_x, trigger_features=[0, 1, 2], trigger_val=5.0, target_label=0, device="cpu") if scenario == "Backdoor_Attack" else 0.0

        benchmark_records.extend([
            {"Attack_Scenario": scenario, "Method": "Standard FedAvg", "Accuracy": metrics_fedavg["accuracy"]*100, "F1": metrics_fedavg["f1"]*100, "FPR": metrics_fedavg["fpr"]*100, "ASR": asr_fedavg*100},
            {"Attack_Scenario": scenario, "Method": "Original FedXAI", "Accuracy": metrics_fedxai["accuracy"]*100, "F1": metrics_fedxai["f1"]*100, "FPR": metrics_fedxai["fpr"]*100, "ASR": asr_fedxai*100},
            {"Attack_Scenario": scenario, "Method": "Proposed Trust-Aware", "Accuracy": metrics_proposed["accuracy"]*100, "F1": metrics_proposed["f1"]*100, "FPR": metrics_proposed["fpr"]*100, "ASR": asr_proposed*100}
        ])

    df_benchmark = pd.DataFrame(benchmark_records)
    csv_benchmark = os.path.join(output_dir, "attack_benchmark_results.csv")
    df_benchmark.to_csv(csv_benchmark, index=False)
    logger.info(f"Saved attack benchmark results to: {csv_benchmark}")

    # Print Formatted Attack Benchmark Table
    print("\n" + "=" * 95)
    print("   FedXAI Step 8: Comprehensive Security Attacks Benchmark Table")
    print("=" * 95)
    header = f"{'Attack Scenario':<28} | {'Method':<24} | {'Acc (%)':<9} | {'F1 (%)':<9} | {'FPR (%)':<9} | {'ASR (%)':<9}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for _, r in df_benchmark.iterrows():
        print(f"{r['Attack_Scenario']:<28} | {r['Method']:<24} | {r['Accuracy']:8.2f}% | {r['F1']:8.2f}% | {r['FPR']:8.2f}% | {r['ASR']:8.2f}%")
    print(sep + "\n")

    # ---------------------------------------------------------
    # PART 2: Malicious Client Ratio Robustness (0%, 10%, 20%, 30%, 40%)
    # ---------------------------------------------------------
    logger.info("\n>>> PART 2: Evaluating Robustness Under Varying Malicious Client Ratios (0% - 40%)...")
    ratio_scenarios = [
        ("0% (Clean)", 0),
        ("11% (1 Attacker)", 1),
        ("22% (2 Attackers)", 2),
        ("33% (3 Attackers)", 3),
        ("44% (4 Attackers)", 4)
    ]
    ratio_records = []

    for label, num_attackers in ratio_scenarios:
        # Poisoning + Sign-flipping attack
        attackers = device_names[-num_attackers:] if num_attackers > 0 else []
        
        # Model weights
        client_w = {}
        client_n = {}
        for d in device_names:
            is_att = (d in attackers)
            splits = client_data[d]
            train_l, val_l, test_l = create_dataloaders(splits["x_train"], splits["y_train"], splits["x_val"], splits["y_val"], splits["x_test"], splits["y_test"], batch_size=batch_size)
            client = IoTClient(0, d, train_l, val_l, test_l, cfg["model"], dev_type)
            client.receive_global_weights(current_weights)
            client.train_local(epochs=2, lr=cfg["federated"]["learning_rate"])
            w_i, n_i = client.get_model_update()
            
            if is_att:
                delta_i = {k: (w_i[k].cpu() - current_weights[k].cpu()) for k in w_i.keys()}
                delta_adv = apply_sign_flipping(delta_i, scale=2.5)
                w_i = {k: (current_weights[k].cpu() + delta_adv[k]) for k in w_i.keys()}
                
            client_w[d] = w_i
            client_n[d] = n_i
            
        # Standard FedAvg
        w_f = fedavg_aggregate([client_w[d] for d in device_names], [client_n[d] for d in device_names])
        m_f = LSTMIDS(cfg["model"]["input_dim"], cfg["model"]["hidden_dim"], cfg["model"]["num_layers"], cfg["model"]["dense_dim"]).to(dev_type)
        m_f.load_state_dict(w_f)
        m_f.eval()
        p_f = [torch.sigmoid(m_f(xb.to(dev_type))).detach().cpu().numpy().ravel() for xb, _ in global_test_loader]
        m_f_res = compute_binary_metrics(global_test_y, np.concatenate(p_f))
        
        # Proposed Trust-Aware
        delts = {d: {k: (client_w[d][k].cpu() - current_weights[k].cpu()) for k in client_w[d].keys()} for d in device_names}
        m_ev = ModelTrustEvaluator()
        m_res, _ = m_ev.evaluate_all(delts)
        t_scores = [m_res[d]["model_trust"] for d in device_names]
        
        w_p = trust_weighted_fedavg([client_w[d] for d in device_names], [client_n[d] for d in device_names], t_scores, rejection_threshold=0.20)
        m_p = LSTMIDS(cfg["model"]["input_dim"], cfg["model"]["hidden_dim"], cfg["model"]["num_layers"], cfg["model"]["dense_dim"]).to(dev_type)
        m_p.load_state_dict(w_p)
        m_p.eval()
        p_p = [torch.sigmoid(m_p(xb.to(dev_type))).detach().cpu().numpy().ravel() for xb, _ in global_test_loader]
        m_p_res = compute_binary_metrics(global_test_y, np.concatenate(p_p))

        ratio_records.extend([
            {"Attacker_Ratio": label, "Num_Attackers": num_attackers, "Method": "Standard FedAvg", "Accuracy": m_f_res["accuracy"] * 100},
            {"Attacker_Ratio": label, "Num_Attackers": num_attackers, "Method": "Proposed Trust-Aware", "Accuracy": m_p_res["accuracy"] * 100}
        ])

    df_ratio = pd.DataFrame(ratio_records)
    csv_ratio = os.path.join(output_dir, "malicious_ratio_robustness.csv")
    df_ratio.to_csv(csv_ratio, index=False)
    logger.info(f"Saved malicious ratio robustness to: {csv_ratio}")

    # ---------------------------------------------------------
    # PART 3: Render Publication-Quality Plots
    # ---------------------------------------------------------
    # 3.1 Robustness Curves vs Attacker Ratio
    plt.figure(figsize=(9, 5), dpi=300)
    sns.lineplot(
        data=df_ratio,
        x="Attacker_Ratio",
        y="Accuracy",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        palette={"Standard FedAvg": "#d62728", "Proposed Trust-Aware": "#2b8cbe"},
        linewidth=2.5,
        markersize=8
    )
    plt.title("FedXAI Step 8: Defense Robustness Across Increasing Attacker Proportions", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Global Intrusion Detection Accuracy (%)", fontsize=11, fontweight="bold")
    plt.xlabel("Malicious Client Ratio", fontsize=11, fontweight="bold")
    plt.ylim(0, 105)
    plt.legend(loc="lower left")
    plt.tight_layout()
    curve_path = os.path.join(output_dir, "attack_robustness_curves.png")
    plt.savefig(curve_path, bbox_inches="tight", dpi=300)
    plt.close("all")
    logger.info(f"Saved attack robustness curves to: {curve_path}")

    logger.info("=" * 90)
    logger.info("   FedXAI Step 8 Attack Benchmark Completed Successfully!")
    logger.info("=" * 90)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedXAI Step 8 Security Attacks Benchmark")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_attack_evaluation_suite(config_path=args.config, device=args.device)
