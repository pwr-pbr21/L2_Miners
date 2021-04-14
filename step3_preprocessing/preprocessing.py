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
            print(f'Error: \"{gh_login}\" not in Stack data!', file=stderr)
    df = pd.DataFrame(data)
    df.gh_bio = df.gh_bio.fillna("")  # fill empty cells with empty string
    return df


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


def author_information(df):
    df.gh_bio = df.gh_bio.apply(strip_html_tags).apply(strip_numbers)
    bio_bw = apply_bag_of_words(df.gh_bio.values.astype("U"), BIO_MAX, BIO_MIN)
    print(f"{len(bio_bw[0])} words were selected for developer bio after Bag of Words.")
    return pd.DataFrame(
        data=normalize(bio_bw[1].toarray()),
        columns=[b + " (Bio)" for b in bio_bw[0]],
        index=df.gh_login
    )


def main():
    print("1: Reading stack and github data")
    _so_data, _gh_data = read_stack_data(STACK_PATH), read_github_data(GITHUB_PATH)

    print("2: Preparing author information")
    author_ds = prepare_author_information(_so_data, _gh_data)
    bio_ds = author_information(author_ds)

    print("3: Preparing repos information")
    repo_ds = prepare_repos_information(_gh_data)


if __name__ == '__main__':
    main()
