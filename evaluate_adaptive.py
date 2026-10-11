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
from sklearn.metrics import roc_auc_score

# Ensure root directory is on path
sys.path.insert(0, os.path.abspath("."))

from utils.logger import get_logger
from utils.metrics import compute_binary_metrics, print_metrics_table
from data.preprocessing import NBaIoTDataPreprocessor
from data.download_nbaiot import get_nbaiot_feature_names, prepare_nbaiot_data
from data.nbaiot_dataset import create_dataloaders, NBaIoTDataset
from models.lstm import LSTMIDS
from federated.client import IoTClient
from federated.aggregation import (
    compute_model_trust_weights,
    compute_joint_trust_weights,
    compute_temporal_trust_weights,
    aggregate_models
)
from security.model_trust import (
    ModelTrustEvaluator,
    calculate_client_update
)
from security.explanation_trust import (
    ExplanationTrustEvaluator,
    get_shap_representation
)
from security.joint_trust import compute_client_joint_trust
from security.temporal_trust import TemporalTrustManager
from security.attacks import (
    BaseAttack,
    NoAttack,
    AdaptiveAttack,
    SCHEDULE_A,
    SCHEDULE_B,
    SCHEDULE_C,
    select_malicious_clients,
    create_attack
)
from xai.shap_explainer import SHAPExplainer

logger = get_logger("StageG6_Adaptive")

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def create_lstm_model(model_config: dict, device: str = "cpu") -> LSTMIDS:
    return LSTMIDS(
        input_dim=model_config.get("input_dim", 115),
        hidden_dim=model_config.get("hidden_dim", 64),
        num_layers=model_config.get("num_layers", 2),
        dense_dim=model_config.get("dense_dim", 32),
        dropout=model_config.get("dropout", 0.2)
    ).to(device)

def evaluate_model_on_test_loader(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Evaluates a PyTorch model on test_loader and computes binary IDS metrics."""
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = model.to(dev)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b = x_b.to(dev)
            probs = torch.sigmoid(model(x_b)).cpu().numpy().ravel()
            all_preds.extend(probs)
            all_targets.extend(y_b.numpy().ravel())
            
    return compute_binary_metrics(all_targets, all_preds)

def compute_detection_metrics(
    true_malicious_mask: List[int],
    final_trust_scores: List[float],
    threshold: float = 0.50
) -> Dict[str, float]:
    """
    Evaluates malicious client detection where anomaly score = (1.0 - FinalTrust).
    A client is flagged as malicious if FinalTrust <= threshold.
    """
    y_true = np.asarray(true_malicious_mask, dtype=int)
    anomaly_scores = 1.0 - np.asarray(final_trust_scores, dtype=float)
    y_pred = (anomaly_scores >= (1.0 - threshold)).astype(int)
    
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, anomaly_scores))
        else:
            auc = 1.0 if tp > 0 else 0.5
    except Exception:
        auc = 0.5
        
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": prec,
        "recall_detection_rate": rec,
        "false_positive_rate": fpr,
        "f1": f1,
        "auc": auc
    }

def run_adaptive_benchmark(
    config_path: str = "configs/config.yaml",
    rounds: int = 6,
    malicious_ratio: float = 0.10,
    base_attack: str = "model_poisoning",
    schedule_name: str = "schedule_a",
    start_round: int = 3,
    seed: int = 42,
    device: str = "cuda",
    sweep_schedules: bool = True
):
    output_dir = "results/attacks/adaptive"
    os.makedirs(output_dir, exist_ok=True)
    cfg = load_config(config_path)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    feature_names = get_nbaiot_feature_names()
    dev_type = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    
    # Resolve Schedule
    schedules_map = {
        "schedule_a": SCHEDULE_A,
        "schedule_b": SCHEDULE_B,
        "schedule_c": SCHEDULE_C
    }
    active_schedule = schedules_map.get(schedule_name.lower().strip(), SCHEDULE_A)
    
    logger.info("=" * 115)
    logger.info("   STAGE G6: SECURITY / ADVERSARIAL EVALUATION — ADAPTIVE / GRADUAL ATTACK")
    logger.info(
        f"   Attack: Adaptive | Base Attack: {base_attack} | Schedule: {schedule_name} ({active_schedule}) "
        f"| Malicious Ratio: {malicious_ratio:.2f} | Start Round: {start_round} | Seed: {seed}"
    )
    logger.info("=" * 115)
    
    # 1. Load Data
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
    num_clients = len(device_names)
    client_data, (global_test_x, global_test_y) = preprocessor.load_and_preprocess_all(device_names)
    
    batch_size = cfg["federated"]["batch_size"]
    global_test_dataset = NBaIoTDataset(global_test_x, global_test_y)
    global_test_loader = DataLoader(global_test_dataset, batch_size=batch_size, shuffle=False)
    
    # Background baseline samples for SHAP
    benign_idx = np.where(global_test_y == 0)[0][:40]
    attack_idx = np.where(global_test_y == 1)[0][:20]
    bg_samples = global_test_x[benign_idx]
    eval_x_sample = np.vstack([global_test_x[benign_idx[:20]], global_test_x[attack_idx]])

    # 2. Malicious Client Selection
    malicious_device_names = select_malicious_clients(
        total_clients=num_clients,
        malicious_ratio=malicious_ratio,
        seed=seed,
        client_ids=device_names
    )
    logger.info(f"Selected Malicious Clients ({len(malicious_device_names)}/{num_clients}): {malicious_device_names}")
    malicious_mask = [1 if dev in malicious_device_names else 0 for dev in device_names]
    
    # 3. Instantiate Attack
    attack = AdaptiveAttack(
        malicious_client_ids=malicious_device_names,
        base_attack=base_attack,
        schedule=active_schedule,
        start_round=start_round,
        seed=seed
    )

    # 4. Client Pool Factory
    def build_client_pool():
        clients = []
        for client_id, dev in enumerate(device_names):
            splits = client_data[dev]
            train_l, val_l, test_l = create_dataloaders(
                splits["x_train"], splits["y_train"],
                splits["x_val"], splits["y_val"],
                splits["x_test"], splits["y_test"],
                batch_size=batch_size
            )
            c = IoTClient(
                client_id=client_id,
                device_name=dev,
                train_loader=train_l,
                val_loader=val_l,
                test_loader=test_l,
                model_config=cfg["model"],
                device=dev_type
            )
            clients.append(c)
        return clients

    initial_model = create_lstm_model(cfg["model"], device=dev_type)
    initial_weights = {k: v.cpu().clone() for k, v in initial_model.state_dict().items()}
    
    clean_init_metrics = evaluate_model_on_test_loader(initial_model, global_test_loader, dev_type)
    logger.info(f"Initial Untrained Model F1: {clean_init_metrics['f1'] * 100:.2f}% | Acc: {clean_init_metrics['accuracy'] * 100:.2f}%")

    # =========================================================================
    # PHASE A: CLEAN BASELINE EXPERIMENT (0% Malicious, FedAvg)
    # =========================================================================
    logger.info(f"\n>>> PHASE A: Running Clean Baseline Experiment ({rounds} Rounds, 0% Malicious, FedAvg)...")
    clean_clients = build_client_pool()
    clean_server_model = create_lstm_model(cfg["model"], device=dev_type)
    clean_server_model.load_state_dict(initial_weights)
    
    clean_round_metrics = []
    for r in range(1, rounds + 1):
        g_weights = {k: v.cpu().clone() for k, v in clean_server_model.state_dict().items()}
        updates, counts = [], []
        for client in clean_clients:
            client.receive_global_weights(g_weights)
            client.train_local(epochs=1, lr=cfg["federated"]["learning_rate"])
            w_i, n_i = client.get_model_update()
            updates.append(w_i)
            counts.append(n_i)
        agg_w, _ = aggregate_models(updates, counts, None, method="fedavg")
        clean_server_model.load_state_dict(agg_w)
        m = evaluate_model_on_test_loader(clean_server_model, global_test_loader, dev_type)
        m["round"] = r
        clean_round_metrics.append(m)
        logger.info(f"Clean Round {r} FedAvg -> F1: {m['f1']*100:.2f}% | Acc: {m['accuracy']*100:.2f}% | FPR: {m['fpr']*100:.2f}%")

    final_clean_f1 = clean_round_metrics[-1]["f1"]
    final_clean_acc = clean_round_metrics[-1]["accuracy"]

    # =========================================================================
    # PHASE B: ADVERSARIAL EVALUATION UNDER ADAPTIVE / GRADUAL ATTACK
    # Comparing: FedAvg vs Model Trust vs Joint Trust vs Temporal Trust
    # =========================================================================
    logger.info(f"\n>>> PHASE B: Multi-Round Comparative Evaluation across 4 Aggregation Schemes...")
    aggregation_schemes = ["fedavg", "model_trust", "joint_trust", "temporal_trust"]
    
    comparison_results = []
    client_trajectories = []
    schedule_records = []
    detection_records = []

    scheme_models = {
        scheme: create_lstm_model(cfg["model"], device=dev_type) for scheme in aggregation_schemes
    }
    for sm in scheme_models.values():
        sm.load_state_dict(initial_weights)
        
    scheme_temporal_managers = {
        scheme: TemporalTrustManager(lambda_decay=0.70, beta=0.50) for scheme in aggregation_schemes
    }

    model_evaluator = ModelTrustEvaluator(w_direction=0.50, w_norm=0.20, w_layer=0.30, reference_strategy="leave_one_out")
    expl_evaluator = ExplanationTrustEvaluator(w_similarity=0.40, w_jaccard=0.30, w_fidelity=0.30, top_k=5, reference_strategy="leave_one_out")
    
    clients = build_client_pool()

    first_instant_detect_round = None
    first_temporal_detect_round = None

    for r in range(1, rounds + 1):
        current_intensity = attack.get_round_intensity(r)
        attack_is_active = (current_intensity > 0.0)
        phase_str = "HONEST PHASE" if not attack_is_active else f"ACTIVE ATTACK (c={current_intensity:.1f})"
        
        print(f"\n" + "=" * 125)
        print(f"   --- COMMUNICATION ROUND {r}/{rounds} [{phase_str}] ---")
        print("=" * 125)
        
        main_global_weights = {k: v.cpu().clone() for k, v in scheme_models["temporal_trust"].state_dict().items()}
        
        client_updates = {}
        client_deltas = {}
        sample_counts = []
        client_models_dict = {}
        client_eval_x_dict = {}

        for i, client in enumerate(clients):
            dev = client.device_name
            client.receive_global_weights(main_global_weights)
            client.train_local(epochs=1, lr=cfg["federated"]["learning_rate"])
            honest_w, n_i = client.get_model_update()
            sample_counts.append(n_i)
            
            # Honest delta relative to pre-round global weights
            honest_delta = calculate_client_update(main_global_weights, honest_w)
            
            # Apply Adaptive Attack Hook
            adv_delta, adv_w = attack.apply_update_attack(
                delta=honest_delta,
                global_weights=main_global_weights,
                client_id=dev,
                round_idx=r
            )
            
            # Numerical validation
            for pk, pv in adv_delta.items():
                assert not torch.isnan(pv).any(), f"NaN in delta '{pk}' for {dev}"
                assert not torch.isinf(pv).any(), f"Inf in delta '{pk}' for {dev}"
                
            client_updates[dev] = adv_w
            client_deltas[dev] = adv_delta
            
            # Load into client model for SHAP evaluation
            client.model.load_state_dict({k: v.to(dev_type) for k, v in adv_w.items()})
            client_models_dict[dev] = client.model
            client_eval_x_dict[dev] = eval_x_sample

        # Trust Calculations
        # 1. Model Trust (M_i)
        m_scores, _ = model_evaluator.evaluate_all(client_deltas)
        m_trust_list = [m_scores[c.device_name]["model_trust"] for c in clients]

        # 2. Explanation Trust (E_i)
        client_explanations = {}
        for client in clients:
            explainer = SHAPExplainer(model=client.model, background_data=bg_samples, feature_names=feature_names, device="cpu")
            shap_vals = explainer.compute_shap_values(eval_x_sample)
            client_explanations[client.device_name] = get_shap_representation(shap_vals)

        e_scores, _ = expl_evaluator.evaluate_all(
            client_explanations=client_explanations,
            client_models=client_models_dict,
            x_eval_dict=client_eval_x_dict,
            feature_names=feature_names,
            device=dev_type
        )
        e_trust_list = [e_scores[c.device_name]["explanation_trust"] for c in clients]

        # 3. Joint Trust (T_i) = alpha * M_i + (1 - alpha) * E_i
        j_trust_dict = compute_client_joint_trust(m_scores, e_scores, alpha=0.50)
        j_trust_list = [j_trust_dict[c.device_name] for c in clients]

        # 4. Historical (H_i) and Final Trust (F_i)
        h_trust_list = []
        f_trust_list = []
        for i, client in enumerate(clients):
            rec = scheme_temporal_managers["temporal_trust"].update_client(
                client_name=client.device_name,
                joint_trust=j_trust_list[i],
                round_idx=r,
                model_trust=m_trust_list[i],
                explanation_trust=e_trust_list[i]
            )
            h_trust_list.append(rec["historical_trust"])
            f_trust_list.append(rec["final_trust"])

        # Compute Weights for All 4 Methods
        fedavg_w, model_w = compute_model_trust_weights(sample_counts, m_trust_list)
        _, joint_w = compute_joint_trust_weights(sample_counts, j_trust_list)
        _, final_w = compute_temporal_trust_weights(sample_counts, f_trust_list)

        # Print Client Trust Table
        header = f"{'ID':<2} | {'Device':<32} | {'Role':<10} | {'Model M_i':<9} | {'Expl E_i':<9} | {'Joint T_i':<9} | {'Hist H_i':<9} | {'Final F_i':<9} | {'FedAvg W':<8} | {'Final W':<8}"
        print(header)
        print("-" * len(header))
        for i, client in enumerate(clients):
            is_m_str = "MALICIOUS" if client.device_name in malicious_device_names else "Honest"
            print(
                f"{i:<2} | {client.device_name:<32} | {is_m_str:<10} | "
                f"{m_trust_list[i]:9.4f} | {e_trust_list[i]:9.4f} | {j_trust_list[i]:9.4f} | "
                f"{h_trust_list[i]:9.4f} | {f_trust_list[i]:9.4f} | "
                f"{fedavg_w[i]:8.4f} | {final_w[i]:8.4f}"
            )
            client_trajectories.append({
                "round": r,
                "client_id": i,
                "device_name": client.device_name,
                "malicious": 1 if client.device_name in malicious_device_names else 0,
                "attack_type": "adaptive",
                "base_attack": base_attack,
                "attack_intensity": current_intensity,
                "attack_active": 1 if attack_is_active else 0,
                "sample_count": sample_counts[i],
                "model_trust": m_trust_list[i],
                "explanation_trust": e_trust_list[i],
                "joint_trust": j_trust_list[i],
                "historical_trust": h_trust_list[i],
                "final_trust": f_trust_list[i],
                "fedavg_weight": fedavg_w[i],
                "model_trust_weight": model_w[i],
                "joint_trust_weight": joint_w[i],
                "final_trust_weight": final_w[i]
            })

        # Track Malicious Reaction
        mal_final = [f_trust_list[idx] for idx, is_m in enumerate(malicious_mask) if is_m == 1]
        mal_joint = [j_trust_list[idx] for idx, is_m in enumerate(malicious_mask) if is_m == 1]
        if mal_final and mal_joint:
            avg_mf = float(np.mean(mal_final))
            avg_mj = float(np.mean(mal_joint))
            if attack_is_active:
                if avg_mj < 0.50 and first_instant_detect_round is None:
                    first_instant_detect_round = r
                if avg_mf < 0.50 and first_temporal_detect_round is None:
                    first_temporal_detect_round = r

        # Detection Metrics based on Final Trust (threshold 0.50)
        det = compute_detection_metrics(malicious_mask, f_trust_list, threshold=0.50)
        det["round"] = r
        det["attack_intensity"] = current_intensity
        detection_records.append(det)

        schedule_records.append({
            "round": r,
            "attack_intensity": current_intensity,
            "attack_active": attack_is_active,
            "avg_malicious_model_trust": float(np.mean([m_trust_list[idx] for idx, is_m in enumerate(malicious_mask) if is_m == 1])) if any(malicious_mask) else 0.0,
            "avg_malicious_joint_trust": float(np.mean(mal_joint)) if mal_joint else 0.0,
            "avg_malicious_historical_trust": float(np.mean([h_trust_list[idx] for idx, is_m in enumerate(malicious_mask) if is_m == 1])) if any(malicious_mask) else 0.0,
            "avg_malicious_final_trust": float(np.mean(mal_final)) if mal_final else 0.0
        })

        # Multi-Scheme Aggregation & Evaluation
        updates_list = [client_updates[c.device_name] for c in clients]
        scheme_weights_map = {
            "fedavg": fedavg_w,
            "model_trust": model_w,
            "joint_trust": joint_w,
            "temporal_trust": final_w
        }

        print(f"\n   >>> Performance Comparison across Aggregation Schemes (Round {r}):")
        for scheme in aggregation_schemes:
            if scheme == "fedavg":
                agg_dict, _ = aggregate_models(updates_list, sample_counts, None, method="fedavg")
            elif scheme == "model_trust":
                agg_dict, _ = aggregate_models(updates_list, sample_counts, m_trust_list, method="model_trust")
            elif scheme == "joint_trust":
                agg_dict, _ = aggregate_models(updates_list, sample_counts, j_trust_list, method="joint_trust")
            elif scheme == "temporal_trust":
                agg_dict, _ = aggregate_models(updates_list, sample_counts, f_trust_list, method="temporal_trust")
                
            scheme_models[scheme].load_state_dict(agg_dict)
            eval_metrics = evaluate_model_on_test_loader(scheme_models[scheme], global_test_loader, dev_type)
            
            mal_w_sum = sum([scheme_weights_map[scheme][idx] for idx, is_m in enumerate(malicious_mask) if is_m == 1])
            hon_w_sum = sum([scheme_weights_map[scheme][idx] for idx, is_m in enumerate(malicious_mask) if is_m == 0])
            
            avg_mal_trust = float(np.mean([f_trust_list[idx] for idx, is_m in enumerate(malicious_mask) if is_m == 1])) if any(malicious_mask) else 0.0
            avg_hon_trust = float(np.mean([f_trust_list[idx] for idx, is_m in enumerate(malicious_mask) if is_m == 0]))
            
            f1_degradation = max(0.0, final_clean_f1 - eval_metrics["f1"])
            
            comparison_results.append({
                "round": r,
                "aggregation_method": scheme,
                "malicious_ratio": malicious_ratio,
                "attack_intensity": current_intensity,
                "malicious_weight_sum": mal_w_sum,
                "honest_weight_sum": hon_w_sum,
                "avg_malicious_final_trust": avg_mal_trust,
                "avg_honest_final_trust": avg_hon_trust,
                "accuracy": eval_metrics["accuracy"],
                "precision": eval_metrics["precision"],
                "recall": eval_metrics["recall"],
                "f1": eval_metrics["f1"],
                "fpr": eval_metrics["fpr"],
                "fnr": eval_metrics["fnr"],
                "roc_auc": eval_metrics["roc_auc"],
                "clean_f1_baseline": final_clean_f1,
                "f1_degradation": f1_degradation
            })
            
            print(
                f"   * {scheme:<14} | F1: {eval_metrics['f1']*100:6.2f}% | "
                f"Acc: {eval_metrics['accuracy']*100:6.2f}% | "
                f"FPR: {eval_metrics['fpr']*100:6.2f}% | "
                f"Malicious Wt: {mal_w_sum*100:5.2f}% | "
                f"F1 Drop: {f1_degradation*100:5.2f}%"
            )

    logger.info(
        f"\n>>> REACTION DELAY ANALYSIS (Start Round = {start_round}):\n"
        f"   - Instantaneous Joint Trust Detection (< 0.50): Round {first_instant_detect_round} "
        f"(Delay = {first_instant_detect_round - start_round if first_instant_detect_round else 'None'} rounds)\n"
        f"   - Temporal/Final Trust Detection (< 0.50): Round {first_temporal_detect_round} "
        f"(Delay = {first_temporal_detect_round - start_round if first_temporal_detect_round else 'None'} rounds)"
    )

    # 5. Optional Schedule Sweep: Compare Schedule A vs B vs C
    sweep_results = []
    if sweep_schedules:
        logger.info("\n>>> Running Adaptive Schedule Comparison Sweep (Schedule A vs B vs C)...")
        for s_name, s_dict in [("Schedule_A", SCHEDULE_A), ("Schedule_B", SCHEDULE_B), ("Schedule_C", SCHEDULE_C)]:
            s_att = AdaptiveAttack(malicious_client_ids=malicious_device_names, schedule=s_dict, start_round=start_round)
            for r in range(1, rounds + 1):
                c_val = s_att.get_round_intensity(r)
                sweep_results.append({
                    "schedule_name": s_name,
                    "round": r,
                    "intensity": c_val,
                    "attack_active": 1 if c_val > 0 else 0
                })

    # 6. Save Structured CSV Results
    df_traj = pd.DataFrame(client_trajectories)
    traj_path = os.path.join(output_dir, "adaptive_client_trajectories.csv")
    df_traj.to_csv(traj_path, index=False)
    logger.info(f"Saved client trajectories to: {traj_path}")

    df_comp = pd.DataFrame(comparison_results)
    comp_path = os.path.join(output_dir, "adaptive_aggregation_comparison.csv")
    df_comp.to_csv(comp_path, index=False)
    logger.info(f"Saved aggregation comparisons to: {comp_path}")

    df_sched = pd.DataFrame(schedule_records)
    sched_path = os.path.join(output_dir, "adaptive_schedule_tracking.csv")
    df_sched.to_csv(sched_path, index=False)
    logger.info(f"Saved schedule tracking to: {sched_path}")

    if sweep_results:
        df_swp = pd.DataFrame(sweep_results)
        swp_path = os.path.join(output_dir, "adaptive_schedule_sweep.csv")
        df_swp.to_csv(swp_path, index=False)
        logger.info(f"Saved schedule sweep to: {swp_path}")

    # 7. Generate 6-Panel Validation Visual
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axs = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
    
    palette = {"fedavg": "#d9534f", "model_trust": "#f0ad4e", "joint_trust": "#5bc0de", "temporal_trust": "#2e7d32"}

    # Panel 1: Model Trust (M_i) vs Round
    hon_m = df_traj[df_traj["malicious"] == 0].groupby("round")["model_trust"].mean()
    mal_m = df_traj[df_traj["malicious"] == 1].groupby("round")["model_trust"].mean()
    axs[0, 0].plot(hon_m.index, hon_m.values, marker='o', linewidth=2.5, color="#2e7d32", label="Honest Clients (Avg)")
    axs[0, 0].plot(mal_m.index, mal_m.values, marker='^', linewidth=2.5, color="#b71c1c", linestyle="--", label="Malicious Client (Adaptive)")
    axs[0, 0].axvline(start_round, color="gray", linestyle=":", label=f"Attack Starts (R{start_round})")
    axs[0, 0].set_title("1. Model Trust (M_i) vs Round", fontsize=11, fontweight="bold")
    axs[0, 0].set_xlabel("Communication Round", fontsize=9)
    axs[0, 0].set_ylabel("Model Trust Score", fontsize=9)
    axs[0, 0].set_ylim(-0.05, 1.05)
    axs[0, 0].legend(loc="lower left", frameon=True, fontsize=8)

    # Panel 2: Joint Trust (T_i) vs Round
    hon_t = df_traj[df_traj["malicious"] == 0].groupby("round")["joint_trust"].mean()
    mal_t = df_traj[df_traj["malicious"] == 1].groupby("round")["joint_trust"].mean()
    axs[0, 1].plot(hon_t.index, hon_t.values, marker='o', linewidth=2.5, color="#2e7d32", label="Honest Clients")
    axs[0, 1].plot(mal_t.index, mal_t.values, marker='s', linewidth=2.5, color="#b71c1c", linestyle="--", label="Malicious Client")
    axs[0, 1].axvline(start_round, color="gray", linestyle=":", label=f"Attack Starts (R{start_round})")
    axs[0, 1].set_title("2. Joint Trust (T_i) vs Round", fontsize=11, fontweight="bold")
    axs[0, 1].set_xlabel("Communication Round", fontsize=9)
    axs[0, 1].set_ylabel("Joint Trust Score", fontsize=9)
    axs[0, 1].set_ylim(-0.05, 1.05)
    axs[0, 1].legend(loc="lower left", frameon=True, fontsize=8)

    # Panel 3: Historical Trust (H_i) vs Round
    hon_h = df_traj[df_traj["malicious"] == 0].groupby("round")["historical_trust"].mean()
    mal_h = df_traj[df_traj["malicious"] == 1].groupby("round")["historical_trust"].mean()
    axs[0, 2].plot(hon_h.index, hon_h.values, marker='o', linewidth=2.5, color="#2e7d32", label="Honest Clients")
    axs[0, 2].plot(mal_h.index, mal_h.values, marker='D', linewidth=2.5, color="#b71c1c", linestyle="--", label="Malicious Client (EMA Decay)")
    axs[0, 2].axvline(start_round, color="gray", linestyle=":", label=f"Attack Starts (R{start_round})")
    axs[0, 2].set_title("3. Historical Trust (H_i) vs Round (Reputation Buffer)", fontsize=11, fontweight="bold")
    axs[0, 2].set_xlabel("Communication Round", fontsize=9)
    axs[0, 2].set_ylabel("Historical Trust Score", fontsize=9)
    axs[0, 2].set_ylim(-0.05, 1.05)
    axs[0, 2].legend(loc="lower left", frameon=True, fontsize=8)

    # Panel 4: Final Trust (F_i) vs Round
    hon_f = df_traj[df_traj["malicious"] == 0].groupby("round")["final_trust"].mean()
    mal_f = df_traj[df_traj["malicious"] == 1].groupby("round")["final_trust"].mean()
    axs[1, 0].plot(hon_f.index, hon_f.values, marker='o', linewidth=2.5, color="#2e7d32", label="Honest Clients")
    axs[1, 0].plot(mal_f.index, mal_f.values, marker='v', linewidth=2.5, color="#b71c1c", linestyle="--", label="Malicious Client")
    axs[1, 0].axhline(0.50, color="gray", linestyle="--", alpha=0.7, label="Detection Threshold (0.50)")
    axs[1, 0].axvline(start_round, color="gray", linestyle=":", label=f"Attack Starts (R{start_round})")
    axs[1, 0].set_title("4. Final Trust (F_i) vs Round", fontsize=11, fontweight="bold")
    axs[1, 0].set_xlabel("Communication Round", fontsize=9)
    axs[1, 0].set_ylabel("Final Trust Score", fontsize=9)
    axs[1, 0].set_ylim(-0.05, 1.05)
    axs[1, 0].legend(loc="lower left", frameon=True, fontsize=8)

    # Panel 5: Malicious Client Aggregated Weight (%) vs Round
    for scheme in aggregation_schemes:
        sub = df_comp[df_comp["aggregation_method"] == scheme]
        axs[1, 1].plot(sub["round"], sub["malicious_weight_sum"] * 100, marker='d', linewidth=2.5, label=scheme.replace("_", " ").title(), color=palette.get(scheme))
    axs[1, 1].axhline(100.0 / num_clients, color="gray", linestyle="--", alpha=0.7, label=f"FedAvg Equal Weight ({100.0/num_clients:.1f}%)")
    axs[1, 1].axvline(start_round, color="gray", linestyle=":", label=f"Attack Starts (R{start_round})")
    axs[1, 1].set_title("5. Malicious Client Aggregated Weight (%)", fontsize=11, fontweight="bold")
    axs[1, 1].set_xlabel("Communication Round", fontsize=9)
    axs[1, 1].set_ylabel("Aggregation Weight (%)", fontsize=9)
    axs[1, 1].set_ylim(0, 15)
    axs[1, 1].legend(loc="upper right", frameon=True, fontsize=8)

    # Panel 6: Global Clean IDS F1-Score (%) vs Round
    for scheme in aggregation_schemes:
        sub = df_comp[df_comp["aggregation_method"] == scheme]
        axs[1, 2].plot(sub["round"], sub["f1"] * 100, marker='o', linewidth=2.5, label=scheme.replace("_", " ").title(), color=palette.get(scheme))
    axs[1, 2].axhline(final_clean_f1 * 100, color="#333333", linestyle="--", alpha=0.7, label=f"Clean Baseline ({final_clean_f1*100:.1f}%)")
    axs[1, 2].axvline(start_round, color="gray", linestyle=":", label=f"Attack Starts (R{start_round})")
    axs[1, 2].set_title("6. Global Clean IDS F1-Score vs Round", fontsize=11, fontweight="bold")
    axs[1, 2].set_xlabel("Communication Round", fontsize=9)
    axs[1, 2].set_ylabel("Global F1-Score (%)", fontsize=9)
    axs[1, 2].set_ylim(0, 105)
    axs[1, 2].legend(loc="lower left", frameon=True, fontsize=8)

    plt.suptitle("Stage G6: Adaptive / Gradual Attack Defense Benchmark — FedXAI Trust Pipeline", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig_path = os.path.join(output_dir, "stage_g6_adaptive_validation.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved 6-panel validation figure to: {fig_path}")

    # Copy figure to artifacts directory if available
    artifact_dir = os.path.abspath("C:/Users/mg631/.gemini/antigravity-ide/brain/2fc60168-5d84-4e19-92ec-ec184bfb8405")
    if os.path.isdir(artifact_dir):
        dest_path = os.path.join(artifact_dir, "stage_g6_adaptive_validation.png")
        try:
            import shutil
            shutil.copyfile(fig_path, dest_path)
            logger.info(f"Copied validation figure to artifact directory: {dest_path}")
        except Exception as e:
            logger.warning(f"Could not copy figure to artifact directory: {e}")

    logger.info("\n" + "=" * 115)
    logger.info("   STAGE G6: ADAPTIVE ATTACK BENCHMARK COMPLETED SUCCESSFULLY")
    logger.info("=" * 115)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage G6 Adaptive Attack Benchmark")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    parser.add_argument("--rounds", type=int, default=6, help="Number of communication rounds")
    parser.add_argument("--malicious_ratio", type=float, default=0.10, help="Ratio of malicious clients")
    parser.add_argument("--base_attack", type=str, default="model_poisoning", help="Base attack type")
    parser.add_argument("--schedule", type=str, default="schedule_a", help="Schedule name (schedule_a, schedule_b, schedule_c)")
    parser.add_argument("--start_round", type=int, default=3, help="Round when attack begins")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda or cpu)")
    parser.add_argument("--sweep_schedules", action="store_true", default=True, help="Compare schedule presets")
    args = parser.parse_args()

    run_adaptive_benchmark(
        config_path=args.config,
        rounds=args.rounds,
        malicious_ratio=args.malicious_ratio,
        base_attack=args.base_attack,
        schedule_name=args.schedule,
        start_round=args.start_round,
        seed=args.seed,
        device=args.device,
        sweep_schedules=args.sweep_schedules
    )
