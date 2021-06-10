import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

SEED = 42
FOLDS = 10

PROCESED = "../../data/step_3_processed_ground_truth.csv"
PROCESED_FS = "../../data/step_3_processed_ground_truth_fs.csv"
PLOTS_OUT = "../../data/plots/"

VERBOSE = True

data = pd.read_csv(PROCESED, delimiter=";")
data_fs = pd.read_csv(PROCESED_FS, delimiter=";")

X = data.iloc[:, :-5]
Y = data.iloc[:, -5:]

X_fs = data_fs.iloc[:, :-6]
Y_fs = data_fs.iloc[:, -6:]

rf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
baseline = DummyClassifier("stratified", random_state=SEED)
nb_baseline = MultinomialNB()
skf = KFold(n_splits=FOLDS, random_state=SEED, shuffle=True)
rf_clf = OneVsRestClassifier(rf, n_jobs=-1)
baseline_clf = OneVsRestClassifier(baseline, n_jobs=-1)
nb_clf = OneVsRestClassifier(nb_baseline, n_jobs=-1)
gb_clf = OneVsRestClassifier(GradientBoostingClassifier(), n_jobs=-1)
