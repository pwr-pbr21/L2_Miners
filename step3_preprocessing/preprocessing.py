import json
from ast import literal_eval
from sys import stderr

from sklearn.preprocessing import normalize

from common.utils import *

# Work in progress (!)

SEED = 42
FOLDS = 10
CORR_THRESHOLD = 0.7
REPO_THRESHOLD = 5

GITHUB_PATH = "../data/step_2_GithubData.json"
STACK_PATH = "../data/step_1_2_StackUsersOut.csv"

DEPENDENCIES_PATH = "/data/repo_dependencies.csv"
DESCRIPTIONS_PATH = "/data/repo_descriptions.csv"
LANGUAGES_PATH = "/data/repo_commits.csv"
DEVELOPERS_PATH = "../data/developers.csv"
DEVELOPERS_FS_PATH = "/data/developers-with-fullstack.csv"

BIO_MIN = 0.01
BIO_MAX = 0.2
DESC_MIN = 0.04
DESC_MAX = 0.15
NAMES_MIN = 0.03
NAMES_MAX = 0.25
TOPICS_MIN = 0.01
TOPICS_MAX = 0.25


def prepare_author_information() -> pd.DataFrame:
    # Read Github + Stack data
    with open(GITHUB_PATH) as github_file:
        gh_data = json.load(github_file)
    so_data = pd.read_csv(STACK_PATH)
    so_data.Role = so_data.Role.apply(literal_eval)
    # Create dataframe
    df = pd.DataFrame(columns=['gh_login', 'gh_repos', 'gh_bio', 'gh_roles'])
    for gh_login, user_info in gh_data.get("users").items():
        stack_info = so_data.loc[so_data['GithubUrl'] == gh_login]
        if not stack_info.empty:
            gh_repos = len(user_info.get("repositories"))
            gh_bio = user_info.get("bio")
            gh_roles = stack_info.Role.values[0]
            df = df.append({
                'gh_login': gh_login,
                'gh_repos': gh_repos,
                'gh_bio': gh_bio,
                'gh_roles': gh_roles
            }, ignore_index=True)
        else:
            print(f'Error: \"{gh_login}\" not in Stack data!', file=stderr)
    return df


def author_information(df):
    df.gh_bio = df.gh_bio.apply(strip_html_tags).apply(strip_numbers)
    bio_bw = apply_bag_of_words(df.gh_bio.values.astype("U"), BIO_MAX, BIO_MIN)
    print(f"{len(bio_bw[0])} words were selected for developer bio after Bag of Words.")
    bio_ds = pd.DataFrame(
        data=normalize(bio_bw[1].toarray()),
        columns=[b + " (Bio)" for b in bio_bw[0]],
        index=df.gh_login
    )
    return bio_ds


# author_information()
df1 = prepare_author_information()
df2 = author_information(df1)
breakpoint()
