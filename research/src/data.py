import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
<<<<<<< HEAD
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier # AdaBoost model
=======
from sklearn.multioutput import ClassifierChain

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import *
>>>>>>> xgboost

SEED = 42
FOLDS = 10

BOW_TUNING = "/inputData/BoW-tuning.csv"
PROCESED = "/data/step_3_processed_ground_truth.csv"
PROCESED_FS = "/inputData/processed_ground_truth_fullstack.csv"

VERBOSE = True

PLOTS_OUT = "/outputData/plots/"

data = pd.read_csv(PROCESED, delimiter=";")
data_fs = pd.read_csv(PROCESED_FS, delimiter=";")

X = data.loc[:,:"zone.js (dep)"]
Y = data.loc[:,"Backend":]

X_fs = data_fs.loc[:,:"yup (dep.)"]
Y_fs = data_fs.loc[:,"Backend":]

rf = RandomForestClassifier(n_estimators=500, random_state=SEED)
baseline = DummyClassifier("stratified", random_state=SEED)
ada = AdaBoostClassifier (n_estimators = 500, random_state = SEED)
nb_baseline = MultinomialNB()
skf = KFold(n_splits=FOLDS, random_state=SEED, shuffle=True)
rf_clf = OneVsRestClassifier(rf)
baseline_clf = OneVsRestClassifier(baseline)
nb_clf = OneVsRestClassifier(nb_baseline)
<<<<<<< HEAD
ada_clf = OneVsRestClassifier(ada)
gb_clf = OneVsRestClassifier(GradientBoostingClassifier())
=======

# load data
dataset = loadtxt(PROCESED, delimiter=",")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=FOLDS, random_state=SEED)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
>>>>>>> xgboost
