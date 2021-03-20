from data import *

# RQ.1: How accurate are machine learning classifiers in identifying technical roles?


def task1():
    br_scores, br_folds = classify(X, Y, skf, rf_clf, average="micro")
    b_scores, b_folds = classify(X, Y, skf, baseline_clf, average="micro")
    nb_scores, nb_folds = classify(X, Y, skf, nb_clf, average="micro")

    print("******** Random Forest ********")
    classify_report(br_scores, Y.columns)
    print("\n******** Naive Bayes ********")
    classify_report(nb_scores, Y.columns)
    print("\n******** Baseline ********")
    classify_report(b_scores, Y.columns)



# RQ.2: What are the most relevant features to identify technical roles?


def task2():
    var_imp = feature_importances_rank(X, Y, clone(rf))

    var_imp["order"] = var_imp.groupby("role").rank(
        method="first", ascending=False)
    var_imp[var_imp.category == "Dependency"].groupby("role").tail(10)

    p = top_10_features(var_imp)
    ggsave(plot = p, filename = "ex2 wykres top 10", path = PLOTS_OUT)

    for r in Y.columns:
        features_df = build_histogram_data(X, Y, var_imp, r)
        p = plot_histogram_data(features_df, r)
        ggsave(plot = p, filename = "ex2 wykres - "+r, path = PLOTS_OUT)
    
    
    
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
    ggsave(plot = p, filename = "ex3 wykres 1", path = PLOTS_OUT)

    p = (ggplot(cc_by_role, aes(x="index", y="value"))
        + geom_line()
        + geom_hline(yintercept=0, linetype="dashed")
        + facet_wrap("~ metric", ncol=3)
        + labs(x="Classifier Chains permutations", y="Metric value")
        + theme_bw())
    ggsave(plot = p, filename = "ex3 wykres 2", path = PLOTS_OUT)


# RQ.4 How effectively can we identify full-stack developers?

def task4():
    fs_br_scores, _ = classify(X_fs, Y_fs, skf, rf_clf, average="micro")
    fs_b_scores, _ = classify(X_fs, Y_fs, skf, baseline_clf, average="micro")
    fs_nb_scores, _ = classify(X_fs, Y_fs, skf, nb_clf, average="micro")
    
    print("******** Random Forest ********")
    classify_report(fs_br_scores, Y_fs.columns)
    print("\n******** Naive Bayes ********")
    classify_report(fs_nb_scores, Y_fs.columns)
    print("\n******** Baseline ********")
    classify_report(fs_b_scores, Y_fs.columns)
    
    
    fs_roles = ["Backend", "Frontend"]
    Y_fs.loc[Y_fs.FullStack == 1, fs_roles] = 1

    fs_br_scores, _ = classify(X_fs, Y_fs, skf, rf_clf, average="micro")
    fs_b_scores, _ = classify(X_fs, Y_fs, skf, baseline_clf, average="micro")
    fs_nb_scores, _ = classify(X_fs, Y_fs, skf, nb_clf, average="micro")
    
    print("******** Random Forest ********")
    classify_report(fs_br_scores, Y_fs.columns, )
    print("\n******** Naive Bayes ********")
    classify_report(fs_nb_scores, Y_fs.columns)
    print("\n******** Baseline ********")
    classify_report(fs_b_scores, Y_fs.columns)


