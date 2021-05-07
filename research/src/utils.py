""" Utility functions to be used at the data analysis scripts. """

import re
from collections import OrderedDict

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from plotnine import *
from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (auc, f1_score, hamming_loss, jaccard_score,
                             precision_recall_curve,
                             precision_recall_fscore_support, precision_score,
                             recall_score)
from data import FOLDS, VERBOSE

def strip_html_tags(text):
    """Function to remove html tags.

    :param text: string that will be cleaned up
    :return: string without html tags
    """
    if text is np.nan:
        return text
    regex = re.compile(r"<.*?>")
    return re.sub(regex, "", text)


def strip_numbers(text):
    """Remove numbers from the text.

    :param text: text that will be cleaned up
    :return: string without numbers
    """
    if text is np.nan:
        return text
    regex = re.compile(r"-?\d+")
    return re.sub(regex, "", text)


def find_correlation(df, method="pearson", threshold=0.9):
    """Find features that are highly correlated.

    Based on a pandas.DataFrame object, this function returns a list of 
    features that present a correlation score higher than a given threshold.

    @param {pandas.DataFrame} df The DataFrame where correlation will be 
        calculated.
    @param {float} threshold Threshold value used to select correlated features

    @returns a list with features that have correlation score higher than the
        provided one.
    """
    corr_matrix = df.corr(method=method).abs()
    corr_means = {k: corr_matrix[k].mean()
                  for k in corr_matrix.columns.tolist()}
    corr_matrix.loc[:, :] = np.tril(corr_matrix, k=-1)

    correlated = {}
    for col in corr_matrix:
        corr_cols = corr_matrix[col][corr_matrix[col]
                                     >= threshold].index.tolist()
        corr_cols.append(col)

        if len(corr_cols) > 1:
            selected_cols = {k: corr_means[k] for k in corr_cols}

            selected_col = max(selected_cols, key=lambda k: selected_cols[k])
            correlated_col = corr_matrix.transpose()[selected_col].idxmax()
            correlated[selected_col] = (correlated_col,
                                        corr_matrix[correlated_col][selected_col])
    return correlated


def apply_bag_of_words(values, max_df=1.0, min_df=1):
    """Apply Bag of Words (BW) over a set o values.

    @param {values} the list of strings which BW will be applied to.
    @param {max_df} max document frequency value to set into BW.
    @param {min_df} min document frequency value to set into BW.

    @return a tuple with bag of words processed.
    """
    bw = TfidfVectorizer(stop_words="english", max_df=max_df, min_df=min_df)
    features = bw.fit_transform(values)
    return (bw.get_feature_names(), features)


def calculate_metrics(Y_true, Y_pred, Y_proba, average):
    """Calculates the desired metrics for this study.

    This function receives both predicted and real output to calculate the 
    metrics used in this study.

    @param {Y_true} Ground truth's output
    @param {Y_pred} Prediction's output
    @param {Y_proba} Prediction's output in probability format

    @return a list with the metrics results used in our study
    """
    if len(Y_true.shape) == 1:
        Y_proba = Y_proba[:, 1]
    p, r, f1, s = precision_recall_fscore_support(
        Y_true, Y_pred, zero_division=0)
    pr, rr, _ = precision_recall_curve(Y_true.ravel(), Y_proba.ravel())

    # for each label
    mcc = [matthews_corrcoef([s[i] for s in Y_true], [s[i] for s in Y_pred]) for i in range(0, Y_true.shape[1])]

    # all as one
    y_true1 = [s[i] * 2 ** i for i in range(0, len(Y_true[0])) for s in Y_true]
    y_pred1 = [s[i] * 2 ** i for i in range(0, len(Y_pred[0])) for s in Y_pred]
    mcc_all = matthews_corrcoef(y_true1, y_pred1)

    # average of results for single label
    mcc_ave = sum(mcc) / len(mcc)

    scores = (
        precision_score(Y_true, Y_pred, average=average, zero_division=0),
        recall_score(Y_true, Y_pred, average=average, zero_division=0),
        f1_score(Y_true, Y_pred, average=average, zero_division=0),
        auc(rr, pr),
        jaccard_score(Y_true, Y_pred, average=average),
        hamming_loss(Y_true, Y_pred),
        mcc_all,
        mcc_ave,
        mcc,
        p,
        r,
        f1,
        s
    )
    return scores


def classify(X, Y, skf, clf, round_threshold=0.5, average="macro", name=""):
    """ Classification function.

    This function performs a multi-label classification using the dataset and 
    classifier sent as parameter. 

    @param {X} Input data.
    @param {Y} Dependent variables.
    @param {clf} Classifier used in classification process.
    @param {round_threshold} threshold for classifying a prediciton as true.
    @param {average} Prediction type.

    @return a tuple with the overall scores obtained from the classification 
        and the score for each generated fold.
    """
    X = X.values
    if isinstance(Y, pd.Series):
        labels = ["{}_0".format(Y.name), "{}_1".format(Y.name)]
        Y = np.ravel(Y)
    else:
        Y, labels = Y.values, list(Y.columns)

    fold_results = []
    counter = 1
    for train, test in skf.split(X, Y):
        log(f"{name}: {counter}/{FOLDS}")
        counter += 1
        current_clf = clone(clf)
        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]

        current_clf.fit(X_train, Y_train)
        Y_prob = current_clf.predict_proba(X_test)
        Y_pred = current_clf.predict(X_test)

        (p, r, f1, auc, jac, hl, mcc_all,
         mcc_ave, mcc, p_c, r_c, f1_c, s_c) = calculate_metrics(Y_test, Y_pred, Y_prob, average)

        # calculate overall scores for current fold
        fold_scores = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "mcc_ave": mcc_ave,
            "mcc_all": mcc_all,
            "auc": auc,
            "jaccard": jac,
            "hamming_loss": hl,
        }

        for i in range(len(labels)):
            fold_scores["precision_{0}".format(labels[i])] = p_c[i]
            fold_scores["recall_{0}".format(labels[i])] = r_c[i]
            fold_scores["f1_{0}".format(labels[i])] = f1_c[i]
            fold_scores["support_{0}".format(labels[i])] = s_c[i]
            fold_scores["mcc_{0}".format(labels[i])] = mcc[i]

        fold_results.append({
            "scores": fold_scores,
            "y_pred": Y_pred,
            "y_prob": Y_prob,
            "y_test": Y_test
        })

    scores = {}
    for score in fold_results[0]["scores"].keys():
        values = [s["scores"][score] for s in fold_results]
        scores[score] = (np.sum(values) if score.startswith("support_")
                         else np.mean(values))

    return scores, fold_results


def classify_report(scores, columns):
    """Print classification results as a report.

    @param {scores} The scores results from the classification.
    @param {columns} The columns which were predicted in the classification.
    """
    if isinstance(columns, str):
        columns = ["{}_0".format(columns), "{}_1".format(columns)]

    print("{: <15}{: <10}{: <10}{: <10}{: <10}{}".format(
        "Role", "Precision", "Recall", "F1", "MCC", "Support"), flush=True)
    for role in columns:
        p = scores["precision_{}".format(role)]
        r = scores["recall_{}".format(role)]
        f1 = scores["f1_{}".format(role)]
        mcc = scores["mcc_{}".format(role)]
        s = scores["support_{}".format(role)]
        print("{: <15}{:.2f}{:10.2f}{:10.2f}{:10.2f}{:10}"
              .format(role, p, r, f1, mcc, s), flush=True)

    p, r, f1, mcc_ave = scores["precision"], scores["recall"], scores["f1"], scores["mcc_ave"]
    print("\n{: <15}{:.2f}{:10.2f}{:10.2f}{:10.2f}".format("Total:", p, r, f1, mcc_ave), flush=True)

    print("MCC(together): {:.2f}".format(scores["mcc_all"]), flush=True)
    print("AUC:           {:.2f}".format(scores["auc"]), flush=True)
    print("Jaccard:       {:.2f}".format(scores["jaccard"]), flush=True)
    print("Hamming Loss:  {:.2f}".format(scores["hamming_loss"]), flush=True)


def top_10_features(df):
    """Generates feature importance plot.

    This function generates features importance plots based on the dataframe
    sent as parameter to this function.

    @param {df} dataframe with feature importance values to be plotted
    @return the resulting plot
    """
    df = df.groupby("role").tail(10).reset_index(drop=True)
    df["i"] = df.index.tolist()
    categories = CategoricalDtype(categories=df["i"].tolist(), ordered=True)
    df["i"] = df["i"].astype(categories)

    def convert_label(labels):
        return OrderedDict([
            (df[df.i == l[0]].feature.tolist()[0], l[1])
            for l in list(labels.items())
        ])

    return (
            ggplot(df, aes("i", "value", group="category"))
            + geom_segment(
        aes(x="i", xend="i", y="min(value)",
            yend="max(value)"),
        linetype="dashed",
        size=1,
        color="grey"
    )
            + geom_point(aes(color="category", shape="category"), size=7)
            + scale_x_discrete(labels=convert_label)
            + scale_y_continuous(labels=lambda x: ["%d%%" % (v * 100) for v in x])
            + scale_color_brewer(type="qual", palette=7)
            + guides(
        color=guide_legend(title="Category"),
        shape=guide_legend(title="Category")
    )
            + labs(y="% Relevance", x="Features", color="category",
                   shape="category")
            + theme_matplotlib()
            + theme(strip_text=element_text(size=18),
                    axis_title=element_text(size=18),
                    axis_text=element_text(size=16),
                    axis_text_x=element_text(size=16),
                    legend_position="top",
                    legend_text=element_text(size=16),
                    legend_title=element_text(size=18, margin={"b": 10}),
                    legend_title_align="center",
                    aspect_ratio=1.4,
                    panel_spacing_y=0.5,
                    panel_spacing_x=2.8,
                    figure_size=(14, 9))
            + coord_flip()
            + facet_wrap("~ role", ncol=3, scales="free",
                         labeller=as_labeller({
                             "Backend": "Backend",
                             "Frontend": "Frontend",
                             "Mobile": "Mobile"
                         })
                         )
    )


def feature_importances_rank(X, Y, clf):
    """Generates a feature importance rank.

    @param {X} Classifier input.
    @param {Y} Dependent variable.
    @param {clf} Classifier model.

    @return pandas dataframe with the feature importance rank.
    """
    var_imp = []
    for role in list(Y):
        Y_role = Y[role]
        vi_clf = clf.fit(X, Y_role)

        role_vimp = []
        for i, f in enumerate(vi_clf.feature_importances_):
            f_name = X.columns[i]

            if f_name.endswith("(author)") or f_name.endswith("(total)") \
                    or f_name.endswith("(rate)"):
                f_type = "Language"
            elif f_name.endswith("(Bio)"):
                f_type = "Bio"
            elif f_name.endswith("(desc.)"):
                f_type = "Rep. Description"
            elif f_name.endswith("(topic)"):
                f_type = "Rep. Topic"
            elif f_name.endswith("(name)"):
                f_type = "Rep. Name"
            elif f_name.endswith("(dep.)"):
                f_type = "Dependency"
            else:
                f_type = "NO_TYPE"

            role_vimp.append({
                "feature": f_name,
                "value": f,
                "category": f_type,
                "role": role
            })
        role_vimp.sort(key=lambda x: x["value"])
        var_imp += role_vimp
    return pd.DataFrame(var_imp)


def build_histogram_data(X, Y, df, role):
    """Build a dataframe containing the histogram information for the most
    relevant features of a given role.

    @param {X} Input variables used in the model.
    @param {Y} Output variables used in the model.
    @param {df} Dataframing containing the variable importance ranking.
    @param {role} The role which dataframe will be generated.

    @return a dataframe synthesizing the histogram information.
    """
    df = df.groupby("role").tail(10).reset_index(drop=True)
    df["i"] = df.index.tolist()
    categories = CategoricalDtype(categories=df["i"].tolist(), ordered=True)
    df["i"] = df["i"].astype(categories)

    features = df.loc[df["role"] == role].feature.values
    devs = []
    for i, row in pd.concat([X, Y], axis=1).iterrows():
        for f in features:
            devs.append({
                "feature": f,
                "value": row[f],
                role: "Yes" if row[role] else "No"
            })
    dataframe = pd.DataFrame(devs, columns=["feature", "value", role])
    dataframe[role] = dataframe[role].astype("category")
    return dataframe


def plot_histogram_data(histogram_df, role):
    """Plot the feature relevance histogram for a given role.

    @param {histogram_df} Histogram dataframe, built at `build_histogram_data`
        method.
    @param {role} Base role used to generate histogram.

    @return The histogram to be printed
    """
    return (
            ggplot(histogram_df, aes(x="value", fill=role))
            + geom_histogram(alpha=0.5, position="fill", bins=5)
            + facet_wrap("~feature",
                         scales="free",
                         nrow=2,
                         labeller=["No", "Yes"]
                         )
            + theme_matplotlib()
            + labs(x="Percentile values (each bar represents 10%)", y="Frequency")
            + theme(
        figure_size=(9, 4),
        axis_text=element_blank(),
        axis_ticks=element_blank()
    )
    )


def build_cc_data(iterations, original_scores):
    """Build classifier chain dataset.

    Gather the classifier chain results from `iterations` into a single dataset
    to be analyzed later. If needed, the user can pass `original_scores` as
    parameter to obtain CC scores in relative terms.

    @param {iterations} a list with the Classifier Chain results.
    @param {original_scores} results obtained from the original classification
        (without CC).

    @return a pandas dataset with the results for each iteration of the CC.
    """
    cc_dataset = pd.DataFrame(iterations)
    for k in cc_dataset.loc[:, "precision":].columns:
        cc_dataset[k] -= original_scores[k]

    melt_vars = list(set(iterations[0].keys()) - set(original_scores.keys()))
    cc_dataset = cc_dataset.melt(id_vars=melt_vars,
                                 var_name="metric",
                                 value_name="value")

    cc_dataset.metric = cc_dataset.metric.str.replace(
        "hamming_loss", "Hamming Loss")
    cc_dataset.metric = cc_dataset.metric.str.replace(
        "precision_Backend", "Backend (Precision)")
    cc_dataset.metric = cc_dataset.metric.str.replace(
        "recall_Backend", "Backend (Recall)")
    cc_dataset.metric = cc_dataset.metric.str.replace(
        "f1_Backend", "Backend (F1)")
    cc_dataset.metric = cc_dataset.metric.str.replace(
        "precision_Frontend", "Frontend (Precision)")
    cc_dataset.metric = cc_dataset.metric.str.replace(
        "recall_Frontend", "Frontend (Recall)")
    cc_dataset.metric = cc_dataset.metric.str.replace(
        "f1_Frontend", "Frontend (F1)")
    cc_dataset.metric = cc_dataset.metric.str.replace(
        "precision_Mobile", "Mobile (Precision)")
    cc_dataset.metric = cc_dataset.metric.str.replace(
        "recall_Mobile", "Mobile (Recall)")
    cc_dataset.metric = cc_dataset.metric.str.replace(
        "f1_Mobile", "Mobile (F1)")
    cc_dataset.metric = cc_dataset.metric.str.replace(
        "precision_DevOps", "DevOps (Precision)")
    cc_dataset.metric = cc_dataset.metric.str.replace(
        "recall_DevOps", "DevOps (Recall)")
    cc_dataset.metric = cc_dataset.metric.str.replace(
        "f1_DevOps", "DevOps (F1)")
    cc_dataset.metric = cc_dataset.metric.str.replace(
        "precision_DataScientist", "DataScientist (Precision)")
    cc_dataset.metric = cc_dataset.metric.str.replace(
        "recall_DataScientist", "DataScientist (Recall)")
    cc_dataset.metric = cc_dataset.metric.str.replace(
        "f1_DataScientist", "DataScientist (F1)")
    cc_dataset.metric = cc_dataset.metric.str.replace("precision", "Precision")
    cc_dataset.metric = cc_dataset.metric.str.replace("recall", "Recall")
    cc_dataset.metric = cc_dataset.metric.str.replace("f1", "F1")
    cc_dataset.metric = cc_dataset.metric.str.replace("auc", "AUC")
    cc_dataset.metric = cc_dataset.metric.str.replace("jaccard", "Jaccard")

    cc_dataset.metric = cc_dataset.metric.astype(
        CategoricalDtype(cc_dataset.metric.unique(), ordered=True))

    return cc_dataset


###################################################################

def log(*args, **kwargs):
    """Print if in debug mode"""
    if VERBOSE:
        print(*args, **kwargs)


# if 1 I can be sure that positive are really positive, but I accept that many positive can be marked as negative
def precision_scorer():
    def score_func(y_true, y_pred):
        return precision_score(y_true, y_pred, average="micro", zero_division=0)

    return make_scorer(score_func, greater_is_better=True)


# if 1 I can be sure there is no positives marked as negative
def recall_scorer():
    def score_func(y_true, y_pred):
        return recall_score(y_true, y_pred, average="micro", zero_division=0)

    return make_scorer(score_func, greater_is_better=True)


# if 1 I can be sure positives are positive and negatives are negative
def f1_scorer():
    def score_func(y_true, y_pred):
        return f1_score(y_true, y_pred, average="micro", zero_division=0)

    return make_scorer(score_func, greater_is_better=True)


def mcc_ave_scorer():
    def score_func(y_true, y_pred):
        if isinstance(y_true, pd.Series):
            y_t = np.ravel(y_true)
        else:
            y_t = y_true.values
        mcc = [matthews_corrcoef([s[i] for s in y_t], [s[i] for s in y_pred]) for i in range(0, len(y_t[0]))]
        return sum(mcc) / len(mcc)

    return make_scorer(score_func, greater_is_better=True)


def mcc_all_scorer():
    def score_func(y_true, y_pred):
        if isinstance(y_true, pd.Series):
            y_t = np.ravel(y_true)
        else:
            y_t = y_true.values
        y_true1 = [s[i] * 2 ** i for i in range(0, len(y_t[0])) for s in y_t]
        y_pred1 = [s[i] * 2 ** i for i in range(0, len(y_pred[0])) for s in y_pred]
        result = matthews_corrcoef(y_true1, y_pred1)
        return result

    return make_scorer(score_func, greater_is_better=True)


def optimize_for_grid(name, x, y, clf, skf, grid, scorer):
    current_clf = clone(clf)
    grid_cv = GridSearchCV(current_clf, grid, scoring=scorer, n_jobs=-1, cv=skf)
    grid_cv.fit(x, y)
    print(f'******** {name} ********', flush=True)
    print(f'best score: {round(grid_cv.best_score_, 4)}')
    print(f'best param: {grid_cv.best_params_}')
    print(grid_cv.cv_results_)


def optimize_for_random(name, x, y, clf, skf, grid, scorer, iter, seed):
    from datetime import datetime
    import time

    start_time = time.time()
    current_clf = clone(clf)
    random_cv = RandomizedSearchCV(current_clf, grid, scoring=scorer, n_jobs=-1, cv=skf, n_iter=2, random_state=seed)
    random_cv.fit(x, y)
    elapsed_time = time.time() - start_time
    print("Current Time =", datetime.now().strftime("%H:%M:%S"))
    print("estimated time:", flush=True)
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time / 2 * iter)), flush=True)

    current_clf = clone(clf)
    random_cv = RandomizedSearchCV(current_clf, grid, scoring=scorer, n_jobs=-1, cv=skf, n_iter=iter, random_state=seed)
    random_cv.fit(x, y)

    print(f'******** {name} ********', flush=True)
    print(f'best score: {round(random_cv.best_score_, 4)}')
    print(f'best param: {random_cv.best_params_}')
    print(random_cv.cv_results_)
