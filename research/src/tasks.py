import itertools
from threading import Thread

from sklearn.multioutput import ClassifierChain
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from data import *
from utils import *


# RQ.1: How accurate are machine learning classifiers in identifying technical roles?


# noinspection DuplicatedCode
def task1():
    run_classification(X, Y)


def run_classification(_x, _y):
    def classify_in_thread(name, thread_id, results, clf):
        log(f"Begin of {name}")
        scores, folds = classify(_x, _y, skf, clf, average="micro", name=name)
        results[thread_id] = (name, scores, folds)
        log(f"End of {name}")

    _results = [None] * 4

    _threads = [
        Thread(target=classify_in_thread, args=("Random Forest", 0, _results, rf_clf)),
        Thread(target=classify_in_thread, args=("Native Bayes", 1, _results, nb_clf)),
        Thread(target=classify_in_thread, args=("Baseline", 2, _results, baseline_clf)),
        Thread(target=classify_in_thread, args=("GradientBoost", 3, _results, gb_clf))
    ]

    for t in _threads:
        t.start()

    for t in _threads:
        t.join()

    for r in _results:
        if r is not None:
            print(f"******** {r[0]} ********")
            classify_report(r[1], Y.columns)


# RQ.2: What are the most relevant features to identify technical roles?


def task2():
    var_imp = feature_importances_rank(X, Y, clone(rf))

    var_imp["order"] = var_imp.groupby("role").rank(
        method="first", ascending=False)
    var_imp[var_imp.category == "Dependency"].groupby("role").tail(10)

    p = top_10_features(var_imp)
    ggsave(plot=p, filename="ex2 wykres top 10", path=PLOTS_OUT)

    for r in Y.columns:
        features_df = build_histogram_data(X, Y, var_imp, r)
        p = plot_histogram_data(features_df, r)
        ggsave(plot=p, filename="ex2 wykres - " + r, path=PLOTS_OUT)


# RQ.3: Do technical roles influence each other during classification?

def task3():
    Y_rq3 = Y.loc[:, :]
    permutations = itertools.permutations(range(0, Y_rq3.shape[1]))

    iterations = []
    for i, p in enumerate(permutations, start=1):
        p = list(p)
        order = np.array(Y_rq3.columns.tolist())[p]
        print(f"============= {order} =============", flush=True)

        chain_clf = ClassifierChain(rf, order=list(p), random_state=SEED)
        cc_scores, _ = classify(X, Y_rq3, skf, chain_clf, average="micro")
        classify_report(cc_scores, Y_rq3)

        iteration = {i: r for i, r in enumerate(order)}
        iteration.update({
            "index": i,
            "precision": cc_scores["precision"],
            "recall": cc_scores["recall"],
            "f1": cc_scores["f1"],
            "auc": cc_scores["auc"],
            "jaccard": cc_scores["jaccard"],
            "hamming_loss": cc_scores["hamming_loss"]
        })
        for role in list(Y_rq3.columns):
            iteration.update({
                f"precision_{role}": cc_scores[f"precision_{role}"],
                f"recall_{role}": cc_scores[f"recall_{role}"],
                f"f1_{role}": cc_scores[f"f1_{role}"],
            })
        iterations.append(iteration)

    br_scores, br_folds = classify(X, Y, skf, rf_clf, average="micro")
    cc_dataset = build_cc_data(iterations, br_scores)

    cc_general = cc_dataset[np.any([
        cc_dataset.metric == "Precision",
        cc_dataset.metric == "Recall",
        cc_dataset.metric == "F1",
        cc_dataset.metric == "AUC",
        cc_dataset.metric == "Jaccard",
        cc_dataset.metric == "Hamming Loss"
    ], axis=0)]

    cc_by_role = cc_dataset[np.any([
        cc_dataset.metric.str.contains("Backend"),
        cc_dataset.metric.str.contains("Frontend"),
        cc_dataset.metric.str.contains("Mobile"),
        cc_dataset.metric.str.contains("DevOps"),
        cc_dataset.metric.str.contains("DataScientist")
    ], axis=0)]

    p = (ggplot(cc_general, aes(x="index", y="value"))
         + geom_line()
         + geom_hline(yintercept=0, linetype="dashed")
         + facet_wrap("~ metric", ncol=2)
         + labs(x="Classifier Chains permutations", y="Metric value")
         + theme_bw())
    ggsave(plot=p, filename="ex3 wykres 1", path=PLOTS_OUT)

    p = (ggplot(cc_by_role, aes(x="index", y="value"))
         + geom_line()
         + geom_hline(yintercept=0, linetype="dashed")
         + facet_wrap("~ metric", ncol=3)
         + labs(x="Classifier Chains permutations", y="Metric value")
         + theme_bw())
    ggsave(plot=p, filename="ex3 wykres 2", path=PLOTS_OUT)


# RQ.4 How effectively can we identify full-stack developers?

def task4():
    run_classification(X_fs, Y_fs)

    fs_roles = ["Backend", "Frontend"]
    Y_fs.loc[Y_fs.FullStack == 1, fs_roles] = 1

    run_classification(X_fs, Y_fs)


def task5():
    # ******** Nearest Neighbors ********
    # Role           Precision Recall    F1        MCC       Support
    # Backend        0.33      0.13      0.18      0.07       362
    # Frontend       0.64      0.75      0.69      0.37       743
    # Mobile         0.53      0.27      0.36      0.25       388
    # DevOps         0.38      0.05      0.09      0.12       133
    # DataScientist  0.73      0.43      0.53      0.51       186
    #
    # Total:         0.59      0.44      0.50      0.26
    # AUC:           0.53
    # Jaccard:       0.34
    # Hamming Loss:  0.20
    # ******** Tuned(F1) Nearest Neighbors ********
    # Role           Precision Recall    F1        MCC       Support
    # Backend        0.34      0.05      0.09      0.06       362
    # Frontend       0.65      0.81      0.72      0.43       743
    # Mobile         0.64      0.23      0.33      0.28       388
    # DevOps         0.30      0.02      0.04      0.07       133
    # DataScientist  0.83      0.37      0.50      0.52       186
    #
    # Total:         0.65      0.43      0.52      0.27
    # AUC:           0.56
    # Jaccard:       0.35
    # Hamming Loss:  0.19
    # ******** Tuned(MCC) Nearest Neighbors ********
    # Role           Precision Recall    F1        MCC       Support
    # Backend        0.31      0.10      0.15      0.06       362
    # Frontend       0.65      0.78      0.71      0.41       743
    # Mobile         0.59      0.27      0.36      0.28       388
    # DevOps         0.45      0.04      0.07      0.12       133
    # DataScientist  0.79      0.43      0.54      0.54       186
    #
    # Total:         0.62      0.44      0.52      0.28
    # AUC:           0.55
    # Jaccard:       0.35
    # Hamming Loss:  0.19
    # ******** Neural Network ********
    # Role           Precision Recall    F1        MCC       Support
    # Backend        0.39      0.29      0.32      0.17       362
    # Frontend       0.74      0.78      0.75      0.53       743
    # Mobile         0.60      0.49      0.53      0.41       388
    # DevOps         0.40      0.25      0.30      0.27       133
    # DataScientist  0.76      0.64      0.68      0.65       186
    #
    # Total:         0.64      0.56      0.59      0.41
    # AUC:           0.56
    # Jaccard:       0.42
    # Hamming Loss:  0.18
    # ******** Tuned(F1) Neural Network ********
    # Role           Precision Recall    F1        MCC       Support
    # Backend        0.40      0.23      0.29      0.16       362
    # Frontend       0.74      0.75      0.74      0.51       743
    # Mobile         0.71      0.46      0.55      0.47       388
    # DevOps         0.54      0.21      0.28      0.28       133
    # DataScientist  0.81      0.64      0.72      0.69       186
    #
    # Total:         0.68      0.53      0.60      0.42
    # AUC:           0.64
    # Jaccard:       0.43
    # Hamming Loss:  0.17
    # ******** Tuned(MCC) Neural Network ********
    # Role           Precision Recall    F1        MCC       Support
    # Backend        0.43      0.30      0.35      0.20       362
    # Frontend       0.74      0.74      0.74      0.51       743
    # Mobile         0.61      0.50      0.54      0.42       388
    # DevOps         0.52      0.27      0.34      0.33       133
    # DataScientist  0.84      0.68      0.74      0.72       186
    #
    # Total:         0.66      0.56      0.60      0.44
    # AUC:           0.63
    # Jaccard:       0.43
    # Hamming Loss:  0.17
    # ******** Random Forest ********
    # Role           Precision Recall    F1        MCC       Support
    # Backend        0.53      0.08      0.13      0.13       362
    # Frontend       0.75      0.79      0.77      0.55       743
    # Mobile         0.85      0.38      0.53      0.49       388
    # DevOps         0.67      0.09      0.15      0.22       133
    # DataScientist  0.89      0.67      0.76      0.74       186
    #
    # Total:         0.77      0.49      0.60      0.43
    # AUC:           0.72
    # Jaccard:       0.43
    # Hamming Loss:  0.15
    # ******** Publication Random Forest ********
    # Role           Precision Recall    F1        MCC       Support
    # Backend        0.55      0.07      0.12      0.14       362
    # Frontend       0.74      0.81      0.77      0.56       743
    # Mobile         0.86      0.40      0.54      0.51       388
    # DevOps         0.55      0.09      0.14      0.20       133
    # DataScientist  0.90      0.67      0.76      0.75       186
    #
    # Total:         0.78      0.50      0.61      0.43
    # AUC:           0.73
    # Jaccard:       0.44
    # Hamming Loss:  0.15
    # ******** Tuned(F1) Random Forest ********
    # Role           Precision Recall    F1        MCC       Support
    # Backend        0.41      0.43      0.42      0.24       362
    # Frontend       0.76      0.81      0.78      0.58       743
    # Mobile         0.65      0.62      0.63      0.52       388
    # DevOps         0.52      0.46      0.47      0.44       133
    # DataScientist  0.73      0.80      0.76      0.73       186
    #
    # Total:         0.65      0.67      0.66      0.50
    # AUC:           0.73
    # Jaccard:       0.49
    # Hamming Loss:  0.16
    # ******** Tuned(MCC) Random Forest ********
    # Role           Precision Recall    F1        MCC       Support
    # Backend        0.50      0.33      0.39      0.27       362
    # Frontend       0.77      0.80      0.78      0.58       743
    # Mobile         0.72      0.58      0.64      0.55       388
    # DevOps         0.69      0.29      0.39      0.40       133
    # DataScientist  0.78      0.77      0.77      0.74       186
    #
    # Total:         0.71      0.62      0.66      0.51
    # AUC:           0.74
    # Jaccard:       0.50
    # Hamming Loss:  0.15
    

    names = [
        "Nearest Neighbors",
        "Tuned(F1) Nearest Neighbors",
        "Tuned(MCC) Nearest Neighbors",
        "Neural Network",
        "Tuned(F1) Neural Network",
        "Tuned(MCC) Neural Network",
        "Random Forest",
        "Publication Random Forest",
        "Tuned(F1) Random Forest",
        "Tuned(MCC) Random Forest",
    ]

    classifiers = [
        KNeighborsClassifier(n_jobs=-1),
        KNeighborsClassifier(p=2, n_neighbors=14, weights="distance", n_jobs=-1),
        KNeighborsClassifier(p=2, n_neighbors=8, weights="distance", leaf_size=20, n_jobs=-1),
        MLPClassifier(random_state=SEED),
        MLPClassifier(random_state=SEED, alpha=0.05, learning_rate_init=0.001, activation="logistic",
                      hidden_layer_sizes=(25,)),
        MLPClassifier(random_state=SEED, solver='adam', learning_rate='constant', hidden_layer_sizes=(30, 10),
                      alpha=1e-04, activation="logistic", max_iter=200, ),
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        RandomForestClassifier(random_state=SEED, n_jobs=-1, n_estimators=500),
        RandomForestClassifier(random_state=SEED, n_jobs=-1, bootstrap=False, class_weight='balanced',
                               criterion='entropy', max_depth=11, max_features=0.1, max_leaf_nodes=33,
                               min_samples_leaf=4, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=980,
                               oob_score=False, warm_start=False),
        RandomForestClassifier(random_state=SEED, n_jobs=-1, bootstrap=False, class_weight='balanced',
                               criterion='entropy', max_depth=None, max_features=0.1, max_leaf_nodes=110,
                               max_samples=None, min_samples_leaf=3, min_samples_split=2, min_weight_fraction_leaf=0.0,
                               n_estimators=1000, oob_score=False, warm_start=False)
    ]

    classifiers = map(lambda c: OneVsRestClassifier(c), classifiers)

    for name, clf in zip(names, classifiers):
        scores, _ = classify(X, Y, skf, clf, average="micro", name=name)

        print(f"******** {name} ********")
        classify_report(scores, Y.columns)


def tuning():
    # best score: 0.2784
    # best param: {'estimator__weights': 'distance', 'estimator__p': 2, 'estimator__n_neighbors': 8,
    #         'estimator__leaf_size': 20, 'estimator__algorithm': 'auto'}

    clf = OneVsRestClassifier(KNeighborsClassifier(n_jobs=-1))
    NN_grid = {
        "estimator__weights": ["distance"],
        "estimator__n_neighbors": [8],
        "estimator__algorithm": ["auto"],
        "estimator__p": [2],
        "estimator__leaf_size": [10, 20, 30, 40, 70]
    }
    #optimize_for_random("Nearest Neighbors", X, Y, clf, skf, NN_grid, mcc_ave_scorer(), 400, SEED)
    optimize_for_grid("Nearest Neighbors", X, Y, clf, skf, NN_grid, mcc_ave_scorer())

    # best score: 0.4435
    # best param: {'estimator__solver': 'adam', 'estimator__max_iter': 200, 'estimator__learning_rate': 'constant',
    #         'estimator__hidden_layer_sizes': (30,15), 'estimator__alpha': 0.0001, 'estimator__activation': 'logistic'}

    clf = OneVsRestClassifier(MLPClassifier(random_state=SEED))
    MLP_grid = {
        'estimator__hidden_layer_sizes': [(20,), (100,), (30, 15,)],
        'estimator__activation': ['identity', 'logistic', 'tanh', 'relu'],
        'estimator__solver': ['adam'],
        'estimator__alpha': [0.0001],
        'estimator__learning_rate': ['constant', 'adaptive', 'invscaling'],
        'estimator__max_iter': [150, 200, 250, 300],
    }
    optimize_for_random("Neural Network", X, Y, clf, skf, MLP_grid, mcc_ave_scorer(), 50, SEED)
    #optimize_for_grid("Neural Network", X, Y, clf, skf, MLP_grid, mcc_ave_scorer())

    # best score: 0.5083
    # best param: {'estimator__bootstrap': False, 'estimator__class_weight': 'balanced', 'estimator__criterion': 'entropy',
    #         'estimator__max_depth': None, 'estimator__max_features': 0.1, 'estimator__max_leaf_nodes': 110,
    #         'estimator__min_samples_leaf': 3, 'estimator__min_samples_split': 2,
    #         'estimator__min_weight_fraction_leaf': 0.0, 'estimator__n_estimators': 1000}

    clf = OneVsRestClassifier(RandomForestClassifier(random_state=SEED, n_jobs=-1))
    RF_grid = {
        'estimator__class_weight': ['balanced'],
        'estimator__criterion': ['entropy'],
        'estimator__bootstrap': [False],
        'estimator__max_features': ['auto', 0.1],
        'estimator__max_depth': [None, 9, 11, 13, 15],
        'estimator__max_leaf_nodes': [30, 110, 130],
        'estimator__min_samples_leaf': [2, 3, 4],
        'estimator__min_samples_split': [2],
        'estimator__min_weight_fraction_leaf': [0.0],
        'estimator__n_estimators': [500, 550, 950],
    }
    optimize_for_random("Random Forest", X, Y, clf, skf, RF_grid, mcc_ave_scorer(), 50, SEED)
    # optimize_for_grid("Random Forest", X, Y, clf, skf, RF_grid, mcc_ave_scorer())
