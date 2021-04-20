import json
import os
from typing import Dict

import pandas as pd
import requests as rq

INPUT_FILENAME = '../data/step_1_2_StackUsersOut.csv'
OUTPUT_FILENAME = '../data/step_2_GithubData.json'
AUTHORIZATION_KEY_FILE = 'auth.key'

USERNAME_COLUMN_NAME = 'GithubUrl'
OUTPUT_ATTRIBUTE_NAME = {
    'users': 'users',
    'not_found': 'not_found_username',
    'with_errors': 'users_with_errors',
    'not_enough_repos': 'not_enough_repos',
    'user_name': 'username',
    'user_id': 'id',
    'user_bio': 'bio',
    'repos': 'repositories',
    'commits_total': 'commits_total',
    'commits_authored': 'commits_authored',
    'repo_name': 'name',
    'repo_topics': 'topics',
    'repo_main_language': 'mainLanguage',
    'repo_description': "description",
    'repo_deps': "dependencies",
}

with open(AUTHORIZATION_KEY_FILE) as AUTH_FILE:
    AUTH_TOKEN = AUTH_FILE.read()

GITHUB_GRAPHQL_ENDPOINT = 'https://api.github.com/graphql'
GITHUB_GRAPHQL_HEADERS = {
    'Accept': 'application/vnd.github.hawkgirl-preview+json',
    'Authorization': f'token {AUTH_TOKEN}'
}

SAVE_STEP = 4
REPOS_PER_PAGE = 7
MAX_REQUEST_ATTEMPTS = 2

USER_GRAPHQL_QUERY = 'query ($login: String!) { user(login: $login) { id bio } }'
REPOSITORY_GRAPHQL_QUERY = 'query ($login: String!, $userId: ID!, $limit: Int!, $repoCursor: String) {user(login: $login) {repositories(first: $limit, after: $repoCursor) {edges {cursor node {name isFork description dependencyGraphManifests(first: 100) {nodes {dependencies(first: 100) {nodes {packageName}}}} languages(first: 1, orderBy: {field: SIZE, direction: DESC}) {nodes {name}} topics: repositoryTopics(first: 100) {nodes {topic {name}}} defaultBranchRef {target {...on Commit {totalCommits: history {totalCount} userCommits: history(author: {id: $userId}) {totalCount}}}}}}}} }'

USER_NOT_FOUND_TYPE = 'NOT_FOUND'

REQUEST_COUNTER = 0


def load_data():
    if os.path.isfile(OUTPUT_FILENAME):
        with open(OUTPUT_FILENAME, "r") as outfile:
            data = json.load(outfile)
            users: dict = data[OUTPUT_ATTRIBUTE_NAME['users']]
            not_found: list = data[OUTPUT_ATTRIBUTE_NAME['not_found']]
            repos: list = data[OUTPUT_ATTRIBUTE_NAME['not_enough_repos']]
            with_errors: list = data[OUTPUT_ATTRIBUTE_NAME['with_errors']]
            return users, not_found, repos, with_errors
    else:
        return {}, [], [], []


def save_data(data):
    with open(OUTPUT_FILENAME, "w+") as outfile:
        json.dump(data, outfile, indent=2)


def update_data(add_users: dict, add_not_found: list, add_repos: list,
                add_with_errors: list):
    global users, not_found, repos, with_errors
    users = {**users, **add_users}
    repos = repos + add_repos
    not_found = not_found + add_not_found
    with_errors = with_errors + add_with_errors
    save_data({
        OUTPUT_ATTRIBUTE_NAME['users']: users,
        OUTPUT_ATTRIBUTE_NAME['not_found']: not_found,
        OUTPUT_ATTRIBUTE_NAME['not_enough_repos']: repos,
        OUTPUT_ATTRIBUTE_NAME['with_errors']: with_errors,
    })


def make_request(query: str, variables: Dict[str, str] = {}):
    global REQUEST_COUNTER
    REQUEST_COUNTER += 1
    print(f'REQUEST VARIABLES {variables}')

    body = {
        "query": query,
        "variables": variables
    }
    response = rq.post(GITHUB_GRAPHQL_ENDPOINT,
                       headers=GITHUB_GRAPHQL_HEADERS,
                       data=json.dumps(body))
    status = response.status_code
    response_json = response.json()
    errors = response_json.get('errors')
    data = response_json.get('data')
    if status != 200 or errors:
        print(f'Response status status: {status}, errors: {errors}')
        if errors and len(errors) == 1 and errors[0].get('type') == 'RATE_LIMITED':
            raise Exception("The limit of requests has been reached")
    return data, errors, status


def user_not_found_error(errors: list):
    return errors and len(errors) == 1 and errors[0].get('type') == USER_NOT_FOUND_TYPE

def make_request_for_user(query: str, variables: Dict[str, str] = {}):
    repeats = 0
    while repeats < MAX_REQUEST_ATTEMPTS:
        repeats += 1
        data, errors, status = make_request(query, variables)
        user_not_found = user_not_found_error(errors)
        if user_not_found:
            return None, True
        if status != 200 or errors:
            continue
        return data.get('user'), True
    return None, False


def make_request_for_repos(variables: Dict[str, any]):
    limit = REPOS_PER_PAGE
    while limit != 0:
        variables['limit'] = limit
        data, errors, status = make_request(REPOSITORY_GRAPHQL_QUERY, variables)
        if status != 200 or errors:
            limit = int(limit / 2)
            continue
        edges = data.get('user').get('repositories').get('edges')
        return edges, limit, True
    return None, None, False


def fetch_all_repos_data(username: str, user_id: str):
    variables = {"login": username, "userId": user_id}
    edges, limit, success = make_request_for_repos(variables)
    if not success:
        return None, False

    all_edges = edges
    while len(edges) == limit:
        variables['repoCursor'] = edges[-1].get('cursor')
        edges, limit, success = make_request_for_repos(variables)
        if not success:
            return None, False
        all_edges += edges

    return list(map(lambda e: e.get('node'), all_edges)), True


def extract_deps(repo: object):
    result = []
    for manifest in repo.get('dependencyGraphManifests').get('nodes'):
        deps = manifest.get('dependencies').get('nodes')
        result.extend(map(lambda e: e.get('packageName'), deps))
    return result


def fetch_repos(username: str, user_id: str):
    repos_list, success = fetch_all_repos_data(username, user_id)
    if not success:
        return None, False
    repos = {}
    for repo in repos_list:
        branchRef = repo.get('defaultBranchRef')
        # empty repos have no default branch
        if repo.get('isFork') or not branchRef:
            continue
        name = repo.get('name')
        deps = extract_deps(repo)
        description = repo.get('description')
        langs = repo.get('languages').get('nodes')
        branch = branchRef.get('target')
        user_commits = branch.get('userCommits').get('totalCount')
        total_commits = branch.get('totalCommits').get('totalCount')
        language = langs[0].get('name') if len(langs) == 1 else None
        topics = list(map(lambda e: e.get('topic').get('name'), repo.get('topics').get('nodes')))
        repos[name] = {
            OUTPUT_ATTRIBUTE_NAME['repo_name']: name,
            OUTPUT_ATTRIBUTE_NAME['repo_deps']: deps,
            OUTPUT_ATTRIBUTE_NAME['repo_topics']: topics,
            OUTPUT_ATTRIBUTE_NAME['repo_main_language']: language,
            OUTPUT_ATTRIBUTE_NAME['repo_description']: description,
            OUTPUT_ATTRIBUTE_NAME['commits_authored']: user_commits,
            OUTPUT_ATTRIBUTE_NAME['commits_total']: total_commits,
        }
    return repos, True


def fetch_data_for(usernames: pd.Series):
    not_found_users, not_enough_repos, users_with_errors = [], [], []
    collection = {}

    i = 0
    size = 0 if len(usernames.index) == 0 else usernames.index[-1]
    for idx, username in usernames.items():
        i += 1
        print(f'Progress {idx} of {size}, Total requests: {REQUEST_COUNTER}')

        variables = {"login": username}
        user, success = make_request_for_user(USER_GRAPHQL_QUERY, variables)
        if not success:
            users_with_errors.append(username)
        elif not user:
            not_found_users.append(username)
        else:
            user_id = user.get('id')
            repos, success = fetch_repos(username, user_id)
            if not success:
                users_with_errors.append(username)
            elif len(repos.keys()) < 5:
                not_enough_repos.append(username)
            else:
                collection[username] = {
                    OUTPUT_ATTRIBUTE_NAME['user_name']: username,
                    OUTPUT_ATTRIBUTE_NAME['user_bio']: user.get('bio'),
                    OUTPUT_ATTRIBUTE_NAME['user_id']: user_id,
                    OUTPUT_ATTRIBUTE_NAME['repos']: repos
                }
        if i % SAVE_STEP == 0:
            update_data(collection, not_found_users, not_enough_repos,
                        users_with_errors)
            not_found_users, not_enough_repos, users_with_errors = [], [], []
            collection = {}

    update_data(collection, not_found_users, not_enough_repos,
                users_with_errors)
    print(f'Progress {size} of {size}, Total requests: {REQUEST_COUNTER}')


def get_waiting_usernames():
    skip = repos + not_found + with_errors + list(users.keys())

    input_un = pd.read_csv(INPUT_FILENAME, delimiter=',')[USERNAME_COLUMN_NAME]
    distinct_un = input_un.drop_duplicates()
    return distinct_un[~distinct_un.isin(skip)]


users, not_found, repos, with_errors = load_data()

waiting_usernames = get_waiting_usernames()
fetch_data_for(waiting_usernames)

print(f'Users with errors: {len(with_errors)}')
print(f'Not existing usernames in GitHub: {len(not_found)}')
print(f'GitHub users without enough repos: {len(repos)}')
print(f'Number of valid users in result file: {len(users.keys())}')
