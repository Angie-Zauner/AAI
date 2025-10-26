import time, numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, log_loss, brier_score_loss
import bnlearn as bn

# --- parametri ---
target_col = "target"      # nome della tua variabile target
pos_label  = "1"           # etichetta positiva (es. "1", "yes", "True")

# --- prepara X_test (solo feature) e y_true binaria ---
X_test = test_discrete.drop(columns=[target_col])
y_true = (test_discrete[target_col].astype(str) == str(pos_label)).astype(int).values

# --- predict_proba (P(target=pos_label | evidence)) ---
states = list((params["model"] if isinstance(params, dict) else params).get_cpds(target_col).state_names[target_col])
pos_idx = states.index(pos_label)

t0 = time.time()
probs = []
for _, row in X_test.iterrows():
    q = bn.inference.fit(params, variables=[target_col], evidence={k: str(v) for k, v in row.items()})
    p = (q.values.get(pos_label, 0.0) if isinstance(q.values, dict)
         else float(np.ravel(q.values)[pos_idx]))
    probs.append(p)
test_time_sec = time.time() - t0
probs = np.clip(np.array(probs, float), 1e-12, 1-1e-12)

# --- metriche ---
y_pred = (probs >= 0.5).astype(int)
bal_acc = balanced_accuracy_score(y_true, y_pred)
acc     = accuracy_score(y_true, y_pred)
try:
    auc  = roc_auc_score(y_true, probs)
except ValueError:
    auc  = float("nan")   # AUC non definibile se c'è una sola classe nel test

# KL(P||Q): per etichette one-hot coincide con la cross-entropy (in "nats")
kl_div = log_loss(y_true, np.c_[1-probs, probs], labels=[0,1])   # = CE = KL (nats)
brier  = brier_score_loss(y_true, probs)

# Expected Calibration Loss (ECE, 10 bin)
bins = np.linspace(0,1,11)
idx  = np.digitize(probs, bins) - 1
ece = 0.0
for b in range(10):
    mask = (idx == b)
    if mask.any():
        conf = probs[mask].mean()
        accb = y_true[mask].mean()
        ece += np.abs(accb - conf) * mask.mean()

# --- stampa ---
print(f"Balanced Accuracy: {bal_acc:.3f}")
print(f"AUC:               {auc:.3f}")
print(f"Accuracy (score):  {acc:.3f}")
print(f"KL Divergence:     {kl_div:.3f}  (nats)")
print(f"Brier score:       {brier:.3f}")
print(f"Expected Cal. Loss:{ece:.3f}")
print(f"Test time (s):     {test_time_sec:.3f}")
