from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# ---------- KS ----------
def ks_stat(y_true, y_prob):
    """Compute KS statistic = max(TPR - FPR)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]
    # cumulative distributions
    cum_bad = np.cumsum(y_true_sorted) / (y_true_sorted.sum() + 1e-12)
    cum_good = np.cumsum(1 - y_true_sorted) / ((1 - y_true_sorted).sum() + 1e-12)
    ks = np.max(np.abs(cum_bad - cum_good))
    return float(ks)

# ---------- PSI (optional drift check) ----------
def psi(expected, actual, bins=10):
    """Population Stability Index between expected (train) and actual (oot)."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    # bin edges based on expected
    quantiles = np.quantile(expected, np.linspace(0, 1, bins+1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    
    exp_counts, _ = np.histogram(expected, bins=quantiles)
    act_counts, _ = np.histogram(actual, bins=quantiles)
    
    exp_perc = exp_counts / (len(expected) + 1e-12)
    act_perc = act_counts / (len(actual) + 1e-12)
    
    # avoid division by zero
    exp_perc = np.clip(exp_perc, 1e-6, None)
    act_perc = np.clip(act_perc, 1e-6, None)
    
    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))

# ---------- Score scaling ----------
def pd_to_score(pd, base_score=600, base_odds=50, pdo=20):
    """
    Convert PD to credit score using log-odds scaling.
    base_score at base_odds (good:bad). PDO = points to double odds.
    """
    pd = np.clip(pd, 1e-6, 1-1e-6)
    odds = (1 - pd) / pd
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    score = offset + factor * np.log(odds)
    return score

def cv_auc(model, X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
   
    for tr, te in skf.split(X, y):
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
    return float(np.mean(aucs))
       
def evaluate_auc(model, X, y):
 
    p = model.predict_proba(X)[:, 1]
 
    if isinstance(p, cp.ndarray):
        p = cp.asnumpy(p)
 
    return float(
        roc_auc_score(y, p)
    )

def evaluate_gini(model, X, y):
    auc = evaluate_auc (model, X, y)
    return 2 * auc - 1

def gini_from_preds(y_true, y_pred):
    return 2 * roc_auc_score(y_true, y_pred) - 1