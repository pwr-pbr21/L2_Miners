from sys import stderr

from sklearn.preprocessing import normalize

from common.utils import *
from utils import read_github_data, read_stack_data

SEED = 42
FOLDS = 10
CORR_THRESHOLD = 0.7
REPO_THRESHOLD = 5

GITHUB_PATH = "../data/step_2_GithubData.json"
STACK_PATH = "../data/step_1_2_StackUsersOut.csv"

PROCESSED_PATH_OUT = "../data/step_3_processed_ground_truth.csv"



BIO_MIN = 0.01
BIO_MAX = 0.2
DESC_MIN = 0.04
DESC_MAX = 0.15
NAMES_MIN = 0.03
NAMES_MAX = 0.25
TOPICS_MIN = 0.01
TOPICS_MAX = 0.25


def prepare_author_information(so_data, gh_data) -> pd.DataFrame:
    # Create dataframe
    data = []
    for gh_login, user_info in gh_data.get("users").items():
        stack_info = so_data.loc[so_data['GithubUrl'] == gh_login]
        if not stack_info.empty:
            gh_repos = len(user_info.get("repositories"))
            gh_bio = user_info.get("bio")
            gh_roles = stack_info.Role.values[0]
            data.append({
                'gh_login': gh_login,
                'gh_repos': gh_repos,
                'gh_bio': gh_bio,
                'gh_roles': gh_roles
            })
        else:
            print(f'Warning: \"{gh_login}\" not in Stack data!', file=stderr)
    df = pd.DataFrame(data)
    df["Backend"] = df["gh_roles"].apply(lambda x: 'Backend' in x)
    df["Frontend"] = df["gh_roles"].apply(lambda x: 'Frontend' in x)
    df["Mobile"] = df["gh_roles"].apply(lambda x: 'Mobile' in x)
    df["DevOps"] = df["gh_roles"].apply(lambda x: 'DevOps' in x)
    df["DataScientist"] = df["gh_roles"].apply(lambda x: 'DataScience' in x)
    df["FullStack"] = df["gh_roles"].apply(lambda x: 'FullStack' in x)

    df = df.drop("gh_roles", axis=1)
    df.gh_bio = df.gh_bio.fillna("")  # fill empty cells with empty string
    return df


def author_information(authors_ds) -> (pd.DataFrame, pd.DataFrame):
    authors_ds.gh_bio = authors_ds.gh_bio.apply(strip_html_tags).apply(strip_numbers)

    filtered_authors = authors_ds[authors_ds.gh_repos >= REPO_THRESHOLD].fillna("")
    filtered_authors.loc[:, "Backend":].sum()
    filtered_authors.drop(["gh_bio", "gh_repos"], axis=1) \
        .groupby(["Backend", "Frontend", "Mobile", "DevOps", "DataScientist", "FullStack"]) \
        .count() \
        .reset_index()

    bio_bw = apply_bag_of_words(filtered_authors.gh_bio.values.astype("U"), BIO_MAX, BIO_MIN)
    print(f"{len(bio_bw[0])} words were selected for developer bio after Bag of Words.")
    return pd.DataFrame(
        data=normalize(bio_bw[1].toarray()),
        columns=[b + " (Bio)" for b in bio_bw[0]],
        index=authors_ds.gh_login
    ), filtered_authors


def prepare_repos_information(gh_data) -> pd.DataFrame:
    data = []
    for gh_login, user_info in gh_data.get("users").items():
        for repo_name, repo_info in user_info.get("repositories").items():
            repo_desc = repo_info.get("description")
            repo_tags = " ".join(repo_info.get("topics"))
            data.append({
                'gh_login': gh_login,
                'repo_name': repo_name,
                'repo_desc': repo_desc,
                'repo_tags': repo_tags,
            })
    df = pd.DataFrame(data)
    df.repo_name = df.repo_name.fillna("")
    df.repo_desc = df.repo_desc.fillna("")
    df.repo_tags = df.repo_tags.fillna("")
    return df


def repos_information(repo_df, bio_df):
    repo_df.repo_name = repo_df.repo_name.apply(strip_html_tags).apply(strip_numbers)
    repo_df.repo_desc = repo_df.repo_desc.apply(strip_html_tags).apply(strip_numbers)
    repo_df.repo_tags = repo_df.repo_tags.apply(strip_html_tags).apply(strip_numbers)
    desc_df = repo_df.groupby("gh_login").agg(lambda c: " ".join(c))
    # right join with bio_ds to include developers without repositories
    desc_df = desc_df.join(bio_df, how="right").iloc[:, :3]

    repo_desc_bw = apply_bag_of_words(
        desc_df.repo_desc.values.astype("U"), DESC_MAX, DESC_MIN)
    repo_topics_bw = apply_bag_of_words(
        desc_df.repo_tags.values.astype("U"), DESC_MAX, DESC_MIN)
    repo_names_bw = apply_bag_of_words(
        desc_df.repo_name.values.astype("U"), DESC_MAX, DESC_MIN)

    rdesc_ds = pd.DataFrame(
        data=normalize(repo_desc_bw[1].toarray()),
        columns=[b + " (desc.)" for b in repo_desc_bw[0]],
        index=desc_df.index
    )
    rtopics_ds = pd.DataFrame(
        data=normalize(repo_topics_bw[1].toarray()),
        columns=[b + " (topic)" for b in repo_topics_bw[0]],
        index=desc_df.index
    )
    rnames_ds = pd.DataFrame(
        data=normalize(repo_names_bw[1].toarray()),
        columns=[b + " (name)" for b in repo_names_bw[0]],
        index=desc_df.index
    )
    return rdesc_ds, rtopics_ds, rnames_ds


def prepare_language_information(gh_data) -> pd.DataFrame:
    data = []
    for gh_login, user_info in gh_data.get("users").items():
        user_data = {'gh_login': gh_login}
        user_language = {}
        for repo_name, repo_info in user_info.get("repositories").items():
            language = repo_info.get("mainLanguage")
            if language is None:
                continue

            commits_author = repo_info.get("commits_authored")
            commits_total = repo_info.get("commits_total")
            commits_rate = commits_author / commits_total

            if language in user_language:
                user_data[language + "_author"] += commits_author
                user_data[language + "_total"] += commits_total
                user_data[language + "_rate"] += commits_rate
                user_language[language] += 1
            else:
                user_data[language + "_author"] = commits_author
                user_data[language + "_total"] = commits_total
                user_data[language + "_rate"] = commits_rate
                user_language[language] = 1
        for column_name, column_value in user_data.items():
            if column_name.endswith("_rate"):
                user_data[column_name] /= user_language[column_name[:-6]]
        data.append(user_data)
    return pd.DataFrame(data).fillna(0.0)


def language_information(lang_ds, bio_ds) -> pd.DataFrame:
    lang_rate = lang_ds.loc[:, lang_ds.columns.str.endswith("_rate")] \
        .assign(gh_login=lang_ds.gh_login) \
        .groupby(["gh_login"]) \
        .mean()

    lang_author = lang_ds.loc[:, lang_ds.columns.str.endswith("_author")] \
        .assign(gh_login=lang_ds.gh_login) \
        .groupby(["gh_login"]) \
        .sum()

    lang_total = lang_ds.loc[:, lang_ds.columns.str.endswith("_total")] \
        .assign(gh_login=lang_ds.gh_login) \
        .groupby(["gh_login"]) \
        .sum()

    lang_rate = lang_rate.join(bio_ds.iloc[:, :0], how="right").fillna(0)
    lang_author = lang_author.join(bio_ds.iloc[:, :0], how="right").fillna(0)
    lang_total = lang_total.join(bio_ds.iloc[:, :0], how="right").fillna(0)

    lang_ds = lang_rate.join([lang_author, lang_total])

    dropped_languages = find_correlation(lang_ds, "spearman", CORR_THRESHOLD)
    lang_ds = lang_ds.drop(dropped_languages.keys(), axis=1)

    unique_languages = lang_ds.nunique()[lang_ds.nunique() <= 1].index
    lang_ds = lang_ds.drop(unique_languages, axis=1)

    return lang_ds.rename(columns={
        **{k: k.replace("_author", " (author)") for k in lang_author.columns},
        **{k: k.replace("_rate", " (rate)") for k in lang_rate.columns},
        **{k: k.replace("_total", " (total)") for k in lang_total.columns},
    })


def prepare_dependencies_information(gh_data) -> pd.DataFrame:
    data = []
    dependencies = set()
    users = set()
    for gh_login, user_info in gh_data.get("users").items():
        for repo_name, repo_info in user_info.get("repositories").items():
            for dependency in repo_info.get("dependencies"):
                dependencies.add(dependency)
                data.append({
                    'gh_login': gh_login,
                    'dep_name': dependency
                })
        users.add(gh_login)

    return pd.DataFrame(data)


def dependencies_information(dep_df, bio_df) -> pd.DataFrame:
    deps_popularity = dep_df.groupby('dep_name')['dep_name'].count()
    deps_popularity = deps_popularity.sort_values(ascending=False)
    deps_popularity = deps_popularity.iloc[:1000]
    dep_df = dep_df[dep_df['dep_name'].isin(deps_popularity.index)]
    dep_df = pd.get_dummies(dep_df.set_index('gh_login')['dep_name'].astype(str)).max(level=0).sort_index()

    deps_ds = dep_df.join(bio_df.iloc[:, :0], how="right").fillna(0)

    dropped_dependencies = find_correlation(deps_ds, "spearman", CORR_THRESHOLD)
    deps_ds = deps_ds.drop(dropped_dependencies.keys(), axis=1)

    unique_dependencies = deps_ds.nunique()[deps_ds.nunique() <= 1].index
    deps_ds = deps_ds.drop(unique_dependencies, axis=1)

    deps_ds = deps_ds.rename(columns={k: k + " (dep)" for k in deps_ds.columns})
    deps_ds = deps_ds.astype(bool)
    return deps_ds


def prepare_dataset():
    print("1: Reading stack and github data")
    so_data, gh_data = read_stack_data(STACK_PATH), read_github_data(GITHUB_PATH)

    print("2: Preparing author information")
    author_df = prepare_author_information(so_data, gh_data)
    bio_ds, filtered_authors = author_information(author_df)

    print("3: Preparing repos information")
    repo_df = prepare_repos_information(gh_data)
    rdesc_ds, rtopics_ds, rnames_ds = repos_information(repo_df, bio_ds)

    print("4: Preparing language information")
    lang_ds = prepare_language_information(gh_data)
    lang_ds = language_information(lang_ds, bio_ds)

    print("5: Preparing dependencies information")
    deps_ds = prepare_dependencies_information(gh_data)
    deps_ds = dependencies_information(deps_ds, bio_ds)

    X = bio_ds.join([rdesc_ds, rtopics_ds, rnames_ds, lang_ds, deps_ds])
    Y = filtered_authors.loc[:, "Backend":]
    Y.index = X.index
    Y = Y.astype(int)
    Z = X.join(Y)
    return X, Y, Z


def prepare_and_save_dataset():
    _, _, Z = prepare_dataset()
    Z.to_csv(PROCESSED_PATH_OUT, index=False, sep=";")


if __name__ == '__main__':
    prepare_and_save_dataset()