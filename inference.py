import time, numpy as np, pandas as pd, bnlearn as bn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, log_loss, brier_score_loss

def _extract_pos_proba_from_bn_predict(out, target_col, pos_label):
    """Prova a estrarre P(target=pos_label) dall'output di bn.predict (DataFrame/dict)."""
    if isinstance(out, pd.DataFrame):
        # 1) Colonna esplicita tipo "P(target=pos_label)"
        col = f"P({target_col}={pos_label})"
        if col in out.columns:
            return out[col].to_numpy(dtype=float)
        # 2) Colonne tipo "target=state" (scegli quella del pos_label)
        cand = [c for c in out.columns if str(c).startswith(f"{target_col}=")]
        if cand:
            cpos = [c for c in cand if str(c).endswith(f"={pos_label}")]
            if cpos:
                return out[cpos[0]].to_numpy(dtype=float)
        # 3) Se c'è la classe MAP (non probabilità), converti a 0/1 (fallback)
        if target_col in out.columns:
            return (out[target_col].astype(str) == str(pos_label)).astype(float).to_numpy()
    elif isinstance(out, dict):
        for k in ("proba", "prob", "probabilities"):
            if k in out:
                return np.asarray(out[k], dtype=float)
        if "y_pred" in out:
            return (pd.Series(out["y_pred"]).astype(str) == str(pos_label)).astype(float).to_numpy()
    raise RuntimeError("Impossibile estrarre P(target=pos_label) dall'output di bn.predict.")

def evaluate_bn(params, test_discrete, target_col, pos_label, threshold=0.5, n_bins=10):
    """
    Usa bn.predict per ottenere le proba del target positivo e stampa le metriche richieste.
    Ritorna (y_pred, y_proba, metrics_dict, test_time_seconds).
    """
    X_test = test_discrete.drop(columns=[target_col])

    t0 = time.time()
    out = bn.predict(params, X_test, variables=[target_col])
    secs = time.time() - t0

    probs = np.clip(_extract_pos_proba_from_bn_predict(out, target_col, pos_label), 1e-12, 1-1e-12)
    y_true = (test_discrete[target_col].astype(str) == str(pos_label)).astype(int).to_numpy()
    y_pred = (probs >= float(threshold)).astype(int)

    # Metriche
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc     = accuracy_score(y_true, y_pred)
    try:
        auc  = roc_auc_score(y_true, probs)
    except ValueError:
        auc  = float("nan")  # AUC non definibile se una sola classe nel test

    # KL divergence media (vs one-hot) = cross-entropy (nats)
    kl_div = log_loss(y_true, np.c_[1 - probs, probs], labels=[0, 1])
    brier  = brier_score_loss(y_true, probs)

    # Expected Calibration Error (ECE) con n_bins uniformi
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx  = np.digitize(probs, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = (idx == b)
        if mask.any():
            conf = probs[mask].mean()
            accb = y_true[mask].mean()
            ece += abs(accb - conf) * mask.mean()

    # Stampa compatta
    print(f"Balanced Accuracy : {bal_acc:.3f}")
    print(f"AUC               : {auc:.3f}")
    print(f"Accuracy (score)  : {acc:.3f}")
    print(f"KL Divergence     : {kl_div:.3f} (nats)")
    print(f"Brier score       : {brier:.3f}")
    print(f"Expected Cal. Loss: {ece:.3f}")
    print(f"Test time (s)     : {secs:.3f}")

    metrics = {
        "balanced_accuracy": float(bal_acc),
        "auc": float(auc),
        "accuracy": float(acc),
        "kl_divergence": float(kl_div),
        "brier": float(brier),
        "ece": float(ece),
    }
    # etichette finali
    y_pred_labels = np.where(y_pred == 1, str(pos_label), f"not_{pos_label}")
    return y_pred_labels, probs, metrics, secs
