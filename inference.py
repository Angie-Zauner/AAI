import numpy as np
import math
import time
import bnlearn as bn
from sklearn import metrics
from sklearn.calibration import calibration_curve

def infer_and_evaluate(test_set, model, model_name='BayesianNetwork', labels = False):

    # Expected Calibration Loss
    def expected_calibration_loss(y_true, y_prob, n_bins=None):
        if n_bins is None:
            N = len(y_true)
            n_bins = math.ceil(math.log2(N) + 1)  # regola di Sturges

        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

        bin_counts, _ = np.histogram(y_prob, bins=n_bins, range=(0, 1))
        nonempty = bin_counts > 0
        bin_weights = bin_counts[nonempty] / np.sum(bin_counts[nonempty])

        return np.sum(bin_weights * np.abs(prob_true - prob_pred))


    # Filter the features that are in the DAG
    test_set = test_set[[col for col in test_set.columns if col in model['model'].nodes()]]

    start = time.time()

    # Predict
    prediction = bn.predict(model, test_set, variables=['target'])

    inference_time = time.time() - start

    # Extract y_true, y_probs and y_pred
    y_pred = prediction['target']
    y_probs = prediction['p']
    y_true= test_set['target']


    # Balanced Accuracy
    bal_acc = metrics.balanced_accuracy_score(y_true, y_pred)

    # F1 Score
    f1 = metrics.f1_score(y_true, y_pred)

    # ROC Curve + AUC
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # Brier Score
    brier = metrics.brier_score_loss(y_true, y_pred)

    # KL Divergence (tra distribuzioni empiriche di pred vs true)
    eps = 1e-12
    P = np.clip(y_true.mean(), eps, 1 - eps)
    Q = np.clip(y_true.mean(), eps, 1 - eps)
    kl_div = np.sum([P * np.log(P / Q) + (1 - P) * np.log((1 - P) / (1 - Q))])

    # Expected Calibration Loss
    ec_loss = expected_calibration_loss(y_true, y_probs)

    # Output
    results = {
        "Model": model_name,
        "Balanced Accuracy": round(bal_acc, 4),
        "F1 Score": round(f1, 4),
        "AUC": round(auc, 4),
        "Brier Score": round(brier, 4),
        "KL Divergence": round(kl_div, 4),
        "Expected Calibration Loss": round(ec_loss, 4),
        "Inference Time (s)": f"{round(inference_time, 4)} s"
    }

    if labels:
        return results, y_pred, y_probs
    
    return results
