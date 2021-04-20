from sys import stderr

from sklearn.preprocessing import normalize

from common.utils import *
from utils import read_github_data, read_stack_data

# Work in progress (!)

SEED = 42
FOLDS = 10
CORR_THRESHOLD = 0.7
REPO_THRESHOLD = 5

GITHUB_PATH = "../data/step_2_GithubData.json"
STACK_PATH = "../data/step_1_2_StackUsersOut.csv"

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
    df.gh_bio = df.gh_bio.fillna("")  # fill empty cells with empty string
    return df


def author_information(author_df: pd.DataFrame) -> pd.DataFrame:
    author_df.gh_bio = author_df.gh_bio.apply(strip_html_tags).apply(strip_numbers)
    bio_bw = apply_bag_of_words(author_df.gh_bio.values.astype("U"), BIO_MAX, BIO_MIN)
    print(f"{len(bio_bw[0])} words were selected for developer bio after Bag of Words.")
    return pd.DataFrame(
        data=normalize(bio_bw[1].toarray()),
        columns=[b + " (Bio)" for b in bio_bw[0]],
        index=author_df.gh_login
    )


def prepare_repos_information(gh_data) -> pd.DataFrame:
    data = []
    for gh_login, user_info in gh_data.get("users").items():
        for repo_name, repo_info in user_info.get("repositories").items():
            repo_desc = repo_info.get("description")
            repo_tags = ""  # To fix
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


def repos_information(repo_df: pd.DataFrame, bio_df):
    repo_df.repo_name = repo_df.repo_name.apply(strip_html_tags).apply(strip_numbers)
    repo_df.repo_desc = repo_df.repo_desc.apply(strip_html_tags).apply(strip_numbers)
    repo_df.repo_tags = repo_df.repo_tags.apply(strip_html_tags).apply(strip_numbers)
    desc_df = repo_df.groupby("gh_login").agg(lambda c: " ".join(c))
    # right join with bio_ds to include developers without repositories
    desc_df = desc_df.join(bio_df, how="right").iloc[:, :3]

    repo_desc_bw = apply_bag_of_words(
        desc_df.repo_desc.values.astype("U"), DESC_MAX, DESC_MIN)
    # repo_topics_bw = apply_bag_of_words(
    #     desc_df.repo_tags.values.astype("U"), DESC_MAX, DESC_MIN)
    repo_names_bw = apply_bag_of_words(
        desc_df.repo_name.values.astype("U"), DESC_MAX, DESC_MIN)

    rdesc_ds = pd.DataFrame(
        data=normalize(repo_desc_bw[1].toarray()),
        columns=[b + " (desc.)" for b in repo_desc_bw[0]],
        index=desc_df.index
    )
    # rtopics_ds = pd.DataFrame(
    #     data=normalize(repo_topics_bw[1].toarray()),
    #     columns=[b + " (topic)" for b in repo_topics_bw[0]],
    #     index=desc_df.index
    # )
    rnames_ds = pd.DataFrame(
        data=normalize(repo_names_bw[1].toarray()),
        columns=[b + " (name)" for b in repo_names_bw[0]],
        index=desc_df.index
    )
    return rdesc_ds, pd.DataFrame(), rnames_ds


def prepare_language_information() -> pd.DataFrame:
    raise NotImplementedError()


def language_information() -> pd.DataFrame:
    raise NotImplementedError()


def prepare_dependencies_information(gh_data: pd.DataFrame) -> pd.DataFrame:
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


def dependencies_information(dep_df: pd.DataFrame, bio_df: pd.DataFrame) -> pd.DataFrame:
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


def main():
    print("1: Reading stack and github data")
    so_data, gh_data = read_stack_data(STACK_PATH), read_github_data(GITHUB_PATH)

    print("2: Preparing author information")
    author_df = prepare_author_information(so_data, gh_data)
    bio_df = author_information(author_df)

    print("3: Preparing repos information")
    repo_df = prepare_repos_information(gh_data)
    rdesc_ds, rtopics_ds, rnames_ds = repos_information(repo_df, bio_df)

    print("4: Preparing language information")

    print("5: Preparing dependencies information")
    deps_ds = prepare_dependencies_information(gh_data)
    deps_ds = dependencies_information(deps_ds, bio_df)


if __name__ == '__main__':
    main()
