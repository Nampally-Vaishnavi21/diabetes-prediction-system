"""
Model zoo: the candidate classifiers and their hyperparameter grids.

WHAT: Each entry = an untrained sklearn classifier + the grid of settings
      GridSearchCV will try.
WHY:  Comparing several model families on the same folds is the fair way to
      pick a model. Grids are kept small on purpose (only 614 training rows;
      huge grids just overfit the validation folds).
NOTE: Keys starting with "model__" are classifier settings.
      "prep__engineer__enabled" lets cross-validation decide whether the
      engineered features help each model (feature engineering as a
      tuned hyperparameter).
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
ENGINEER = {"prep__engineer__enabled": [False, True]}

MODEL_ZOO = {
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "param_grid": {
            **ENGINEER,
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
        "why": "Simple, interpretable linear baseline; coefficients are log-odds.",
    },
    "decision_tree": {
        "display_name": "Decision Tree",
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            **ENGINEER,
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__min_samples_leaf": [1, 5, 10, 20],
            "model__class_weight": [None, "balanced"],
        },
        "why": "Human-readable if/else rules; high variance on its own.",
    },
    "random_forest": {
        "display_name": "Random Forest",
        "estimator": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "param_grid": {
            **ENGINEER,
            "model__max_depth": [4, 6, 8, None],
            "model__min_samples_leaf": [1, 3, 5],
            "model__max_features": ["sqrt", 0.5],
        },
        "why": "Bagging ensemble of trees; reduces the variance of a single tree.",
    },
    "svm": {
        "display_name": "Support Vector Machine (RBF)",
        # An SVM outputs distances, not probabilities. Platt scaling (a sigmoid fitted
        # with internal 5-fold CV) turns them into probabilities. This is what the
        # deprecated SVC(probability=True) did internally.
        "estimator": CalibratedClassifierCV(
            SVC(kernel="rbf", random_state=RANDOM_STATE),
            method="sigmoid", cv=5, ensemble=False),
        "param_grid": {
            **ENGINEER,
            "model__estimator__C": [0.1, 1.0, 10.0],
            "model__estimator__gamma": ["scale", 0.01, 0.1],
            "model__estimator__class_weight": [None, "balanced"],
        },
        "why": "Maximum-margin classifier; RBF kernel captures non-linear boundaries. "
               "Probabilities via Platt (sigmoid) scaling.",
    },
    "gradient_boosting": {
        "display_name": "Gradient Boosting",
        "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            **ENGINEER,
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
            "model__subsample": [0.8, 1.0],
        },
        "why": "Boosting ensemble: each tree corrects the errors of the previous ones.",
    },
    "knn": {
        "display_name": "K-Nearest Neighbours",
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            **ENGINEER,
            "model__n_neighbors": [5, 11, 15, 21, 31],
            "model__weights": ["uniform", "distance"],
        },
        "why": "Distance-based baseline with no training phase.",
    },
}

# Soft-voting ensemble of the top-N tuned models (built in train.py)
ENSEMBLE_KEY = "soft_voting_ensemble"
ENSEMBLE_DISPLAY = "Soft-Voting Ensemble"
ENSEMBLE_TOP_N = 3

# Metrics computed during cross-validation (sklearn scorer names)
CV_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "accuracy": "accuracy",
    # zero_division=0: a config that predicts no positives scores 0 instead of warning
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "brier": "neg_brier_score",   # sklearn negates it so that "higher is better"
}
PRIMARY_METRIC = "roc_auc"
# Models whose CV ROC-AUC is within this margin of the best are treated as tied;
# the tie is broken by the lower (better) CV Brier score.
TIE_TOLERANCE = 0.005
