import numpy as np
import math
import time
from sklearn import metrics
from sklearn.calibration import calibration_curve

def evaluate_bn_metrics(Y_true, Y_pred, Y_prob, model_name='BayesianNetwork'):
    """
    Calcola le metriche richieste per un modello bayesiano:
    Balanced Accuracy, F1, AUC, Brier Score, KL Divergence,
    Expected Calibration Loss e Inference Time.

    Parametri
    ----------
    Y_true : array-like
        Etichette vere.
    Y_pred : array-like
        Etichette predette (classi 0/1).
    Y_prob : array-like
        Probabilità predette per la classe positiva.
    model_name : str, opzionale
        Nome del modello (solo per stampe/log).

    Ritorna
    -------
    dict : metriche calcolate.
    """

    # ========================
    # Funzione interna: Expected Calibration Loss
    # ========================
    def expected_calibration_loss(y_true, y_prob, n_bins=None):
        if n_bins is None:
            N = len(y_true)
            n_bins = math.ceil(math.log2(N) + 1)  # regola di Sturges

        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

        bin_counts, _ = np.histogram(y_prob, bins=n_bins, range=(0, 1))
        nonempty = bin_counts > 0
        bin_weights = bin_counts[nonempty] / np.sum(bin_counts[nonempty])

        # Expected Calibration Loss
        return np.sum(bin_weights * np.abs(prob_true - prob_pred))

    # ========================
    # Calcolo metriche
    # ========================

    start = time.time()

    # Balanced Accuracy
    bal_acc = metrics.balanced_accuracy_score(Y_true, Y_pred)

    # F1 Score
    f1 = metrics.f1_score(Y_true, Y_pred)

    # ROC Curve + AUC
    fpr, tpr, _ = metrics.roc_curve(Y_true, Y_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # Brier Score
    brier = metrics.brier_score_loss(Y_true, Y_prob)

    # KL Divergence (tra distribuzioni empiriche di pred vs true)
    eps = 1e-12
    P = np.clip(Y_true.mean(), eps, 1 - eps)
    Q = np.clip(Y_prob.mean(), eps, 1 - eps)
    kl_div = np.sum([P * np.log(P / Q) + (1 - P) * np.log((1 - P) / (1 - Q))])

    # Expected Calibration Loss
    ec_loss = expected_calibration_loss(Y_true, Y_prob)

    inference_time = time.time() - start

    # ========================
    # Output finale
    # ========================
    results = {
        "Model": model_name,
        "Balanced Accuracy": bal_acc,
        "F1 Score": f1,
        "AUC": auc,
        "Brier Score": brier,
        "KL Divergence": kl_div,
        "Expected Calibration Loss": ec_loss,
        "Inference Time (s)": inference_time
    }

    return results
