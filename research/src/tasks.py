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
    # Role           Precision Recall    F1        Support
    # Backend        0.33      0.13      0.18       362
    # Frontend       0.64      0.75      0.69       743
    # Mobile         0.53      0.27      0.36       388
    # DevOps         0.38      0.05      0.09       133
    # DataScientist  0.73      0.43      0.53       186
    #
    # Total:         0.59      0.44      0.50
    # AUC:           0.53
    # Jaccard:       0.34
    # Hamming Loss:  0.20
    # ******** Tuned(precise) Nearest Neighbors ********
    # Role           Precision Recall    F1        Support
    # Backend        0.00      0.00      0.00       362
    # Frontend       0.68      0.79      0.73       743
    # Mobile         0.67      0.05      0.08       388
    # DevOps         0.00      0.00      0.00       133
    # DataScientist  0.90      0.17      0.27       186
    #
    # Total:         0.69      0.35      0.46
    # AUC:           0.58
    # Jaccard:       0.30
    # Hamming Loss:  0.19
    # ******** Tuned(Rcall) Nearest Neighbors ********
    # Role           Precision Recall    F1        Support
    # Backend        0.28      0.26      0.27       362
    # Frontend       0.61      0.68      0.64       743
    # Mobile         0.36      0.35      0.35       388
    # DevOps         0.16      0.11      0.13       133
    # DataScientist  0.50      0.51      0.50       186
    #
    # Total:         0.46      0.46      0.46
    # AUC:           0.53
    # Jaccard:       0.30
    # Hamming Loss:  0.25
    # ******** Tuned(F1) Nearest Neighbors ********
    # Role           Precision Recall    F1        Support
    # Backend        0.34      0.05      0.09       362
    # Frontend       0.65      0.81      0.72       743
    # Mobile         0.64      0.23      0.33       388
    # DevOps         0.30      0.02      0.04       133
    # DataScientist  0.83      0.37      0.50       186
    #
    # Total:         0.65      0.43      0.52
    # AUC:           0.56
    # Jaccard:       0.35
    # Hamming Loss:  0.19
    # ******** Neural Net ********
    # Role           Precision Recall    F1        Support
    # Backend        0.39      0.29      0.32       362
    # Frontend       0.74      0.78      0.75       743
    # Mobile         0.60      0.49      0.53       388
    # DevOps         0.40      0.25      0.30       133
    # DataScientist  0.76      0.64      0.68       186
    #
    # Total:         0.64      0.56      0.59
    # AUC:           0.56
    # Jaccard:       0.42
    # Hamming Loss:  0.18
    # ******** Tuned(F1) Neural Net ********
    # Role           Precision Recall    F1        Support
    # Backend        0.41      0.22      0.28       362
    # Frontend       0.74      0.76      0.74       743
    # Mobile         0.66      0.47      0.54       388
    # DevOps         0.53      0.19      0.26       133
    # DataScientist  0.80      0.65      0.72       186
    #
    # Total:         0.67      0.53      0.60
    # AUC:           0.64
    # Jaccard:       0.42
    # Hamming Loss:  0.17
    # ******** Random Forest ********
    # Role           Precision Recall    F1        Support
    # Backend        0.53      0.08      0.13       362
    # Frontend       0.75      0.79      0.77       743
    # Mobile         0.85      0.38      0.53       388
    # DevOps         0.67      0.09      0.15       133
    # DataScientist  0.89      0.67      0.76       186
    #
    # Total:         0.77      0.49      0.60
    # AUC:           0.72
    # Jaccard:       0.43
    # Hamming Loss:  0.15
    # ******** Publication Random Forest ********
    # Role           Precision Recall    F1        Support
    # Backend        0.55      0.07      0.12       362
    # Frontend       0.74      0.81      0.77       743
    # Mobile         0.86      0.40      0.54       388
    # DevOps         0.55      0.09      0.14       133
    # DataScientist  0.90      0.67      0.76       186
    #
    # Total:         0.78      0.50      0.61
    # AUC:           0.73
    # Jaccard:       0.44
    # Hamming Loss:  0.15
    #     ******** Tuned(F1) Random Forest ********
    # Role           Precision Recall    F1        Support
    # Backend        0.41      0.43      0.42       362
    # Frontend       0.76      0.81      0.78       743
    # Mobile         0.65      0.62      0.63       388
    # DevOps         0.52      0.46      0.47       133
    # DataScientist  0.73      0.80      0.76       186
    #
    # Total:         0.65      0.67      0.66
    # AUC:           0.73
    # Jaccard:       0.49
    # Hamming Loss:  0.16

    names = [
        "Nearest Neighbors",
        "Tuned(precise) Nearest Neighbors",
        "Tuned(Rcall) Nearest Neighbors",
        "Tuned(F1) Nearest Neighbors",
        "Neural Net",
        "Tuned(F1) Neural Net",
        "Random Forest",
        "Publication Random Forest",
        "Tuned(F1) Random Forest",
    ]

    classifiers = [
        KNeighborsClassifier(n_jobs=-1),
        KNeighborsClassifier(p=1, n_neighbors=60, n_jobs=-1),
        KNeighborsClassifier(p=1, n_neighbors=1, weights="distance", n_jobs=-1),
        KNeighborsClassifier(p=2, n_neighbors=14, weights="distance", n_jobs=-1),
        MLPClassifier(random_state=SEED),
        MLPClassifier(random_state=SEED, alpha=0.05, learning_rate_init=0.001, activation="logistic",
                      hidden_layer_sizes=(25,)),
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        RandomForestClassifier(random_state=SEED, n_jobs=-1, n_estimators=500),
        RandomForestClassifier(random_state=SEED, n_jobs=-1, bootstrap=False, class_weight='balanced',
                               criterion='entropy', max_depth=11, max_features=0.1, max_leaf_nodes=33,
                               min_samples_leaf=4, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=980,
                               oob_score=False, warm_start=False)
    ]
    classifiers = map(lambda c: OneVsRestClassifier(c), classifiers)

    for name, clf in zip(names, classifiers):
        scores, _ = classify(X, Y, skf, clf, average="micro", name=name)

        print(f"******** {name} ********")
        classify_report(scores, Y.columns)
