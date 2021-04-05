import sklearn
import platform
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain

from utils import *

SEED = 42
FOLDS = 10

BOW_TUNING = "/inputData/BoW-tuning.csv"
PROCESED = "/inputData/processed_ground_truth.csv"
PROCESED_FS = "/inputData/processed_ground_truth_fullstack.csv"

PLOTS_OUT = "/outputData/plots/"

data = pd.read_csv(PROCESED, delimiter=";")
data_fs = pd.read_csv(PROCESED_FS, delimiter=";")

X = data.loc[:,:"yup (dep.)"]
Y = data.loc[:,"Backend":]

X_fs = data_fs.loc[:,:"yup (dep.)"]
Y_fs = data_fs.loc[:,"Backend":]

rf = RandomForestClassifier(n_estimators=500, random_state=SEED)
baseline = DummyClassifier("stratified", random_state=SEED)
nb_baseline = MultinomialNB()
skf = KFold(n_splits=FOLDS, random_state=SEED)
rf_clf = OneVsRestClassifier(rf)
baseline_clf = OneVsRestClassifier(baseline)
nb_clf = OneVsRestClassifier(nb_baseline)