import json
import os
from typing import Dict

import pandas as pd
import requests as rq

INPUT_FILENAME = '../stack/QueryResultsOut.csv'
OUTPUT_FILENAME = 'output.json'
AUTHORIZATION_KEY_FILE = 'auth.key'

USERNAME_COLUMN_NAME = 'GithubUrl'
OUTPUT_ATTRIBUTE_NAME = {
    'users': 'users',
    'not_found': 'not_found_username',
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

SAVE_STEP = 3
MAX_ATTEMPTS_NUMBER = 3

REPOS_PER_PAGE = 100
MAX_REPOS_REQUEST_ATTEMPTS = 5

USER_GRAPHQL_QUERY = 'query ($login: String!) { user(login: $login) { id bio } }'
REPOSITORY_GRAPHQL_QUERY = 'query ($login: String!, $userId: ID!, $limit: Int!, $repoCursor: String) {user(login: $login) {repositories(first: $limit, after: $repoCursor) {edges {cursor node {name isFork description dependencyGraphManifests {nodes {dependencies {nodes {packageName}}}} languages(first: 1, orderBy: {field: SIZE, direction: DESC}) {nodes {name}} topics: repositoryTopics(first: 100) {nodes {topic {name}}} defaultBranchRef {target {...on Commit {totalCommits: history {totalCount} userCommits: history(author: {id: $userId}) {totalCount}}}}}}}} }'

REQUEST_COUNTER = 0
NOT_EXISTING_USERNAMES = 0
USERS_WITHOUT_ENOUGH_REPOS = 0


def load_data():
    if os.path.isfile(OUTPUT_FILENAME):
        with open(OUTPUT_FILENAME, "r") as outfile:
            data = json.load(outfile)
            users: dict = data[OUTPUT_ATTRIBUTE_NAME['users']]
            not_found: list = data[OUTPUT_ATTRIBUTE_NAME['not_found']]
            repos: list = data[OUTPUT_ATTRIBUTE_NAME['not_enough_repos']]
            return users, not_found, repos
    else:
        return {}, [], []


def save_data(data):
    with open(OUTPUT_FILENAME, "w+") as outfile:
        json.dump(data, outfile, indent=2)


def update_data(users: dict, not_found: list, lack_of_repos: list):
    l_users, l_not_found, l_repos = load_data()
    save_data({
        OUTPUT_ATTRIBUTE_NAME['users']: {**users, **l_users},
        OUTPUT_ATTRIBUTE_NAME['not_found']: not_found + l_not_found,
        OUTPUT_ATTRIBUTE_NAME['not_enough_repos']: lack_of_repos + l_repos
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
    if response.status_code != 200:
        print(response.json())
        print(f'Occurred unexpected status code: {response.status_code}')
    return response


def make_request_with_repeat(query: str, variables: Dict[str, str] = {}):
    repeats = 0
    while repeats < MAX_ATTEMPTS_NUMBER:
        repeats += 1
        response = make_request(query, variables)
        if response.status_code == 200:
            return response.json()
    raise Exception("Fetching data error")


def make_request_for_repos(query: str, variables: Dict[str, any]):
    limit = REPOS_PER_PAGE
    repeats = 0
    while repeats < MAX_REPOS_REQUEST_ATTEMPTS:
        repeats += 1
        variables['limit'] = limit
        resp = make_request(query, variables)
        if resp.status_code == 200:
            data = resp.json().get('data')
            edges = data.get('user').get('repositories').get('edges')
            return edges, limit
        limit = 1 if limit == 1 else int(limit / 2)
    raise Exception("Fetching data error")


def fetch_all_repos_data(username: str, user_id: str):
    variables = {"login": username, "userId": user_id}
    edges, limit = make_request_for_repos(REPOSITORY_GRAPHQL_QUERY, variables)
    all_edges = edges

    while len(edges) == limit:
        variables['repoCursor'] = edges[-1].get('cursor')
        edges, limit = make_request_for_repos(REPOSITORY_GRAPHQL_QUERY,
                                              variables)
        all_edges += edges
    return all_edges


def extract_deps(repo: object):
    result = []
    for manifest in repo.get('dependencyGraphManifests').get('nodes'):
        deps = manifest.get('dependencies').get('nodes')
        result.extend(map(lambda e: e.get('packageName'), deps))
    return result


def fetch_repos(username: str, user_id: str):
    edges_list = fetch_all_repos_data(username, user_id)
    repos = {}
    for edge in edges_list:
        repo = edge.get('node')
        branchRef = repo.get('defaultBranchRef')
        # empty repos have no default branch
        if repo.get('isFork') or not branchRef:
            continue
        name = repo.get('name')
        description = repo.get('description')
        branch = branchRef.get('target')
        topics = list(
            map(lambda e: e.get('name'), repo.get('topics').get('nodes')))
        langs = repo.get('languages').get('nodes')
        language = langs[0].get('name') if len(langs) == 1 else None
        deps = extract_deps(repo)
        repos[name] = {
            OUTPUT_ATTRIBUTE_NAME['repo_name']: name,
            OUTPUT_ATTRIBUTE_NAME['repo_topics']: topics,
            OUTPUT_ATTRIBUTE_NAME['repo_main_language']: language,
            OUTPUT_ATTRIBUTE_NAME['repo_description']: description,
            OUTPUT_ATTRIBUTE_NAME['commits_authored']: branch.get(
                'userCommits').get('totalCount'),
            OUTPUT_ATTRIBUTE_NAME['commits_total']: branch.get(
                'totalCommits').get('totalCount'),
            OUTPUT_ATTRIBUTE_NAME['repo_deps']: deps,
        }
    return repos


def fetch_user(username: str):
    variables = {"login": username}
    data = make_request_with_repeat(USER_GRAPHQL_QUERY, variables)
    return data.get('data').get('user')


def fetch_data_for(usernames: pd.Series):
    global NOT_EXISTING_USERNAMES, USERS_WITHOUT_ENOUGH_REPOS
    collection, not_found, not_enough_repos = {}, [], []

    i = 0
    size = usernames.index[-1]
    for idx, username in usernames.items():
        print(f'Progress {idx} of {size}, Total requests: {REQUEST_COUNTER}')
        i += 1
        if i % SAVE_STEP == 0:
            update_data(collection, not_found, not_enough_repos)
            collection, not_found, not_enough_repos = {}, [], []

        user = fetch_user(username)
        if not user:
            not_found.append(username)
            continue

        user_id = user.get('id')
        repos = fetch_repos(username, user_id)
        if len(repos.keys()) < 5:
            not_enough_repos.append(username)
            continue

        collection[username] = {
            OUTPUT_ATTRIBUTE_NAME['user_name']: username,
            OUTPUT_ATTRIBUTE_NAME['user_bio']: user.get('bio'),
            OUTPUT_ATTRIBUTE_NAME['user_id']: user_id,
            OUTPUT_ATTRIBUTE_NAME['repos']: repos
        }

    update_data(collection, not_found, not_enough_repos)
    print(f'Progress {size} of {size}, Total requests: {REQUEST_COUNTER}')
    print(f'Not existing usernames in GitHub: {NOT_EXISTING_USERNAMES}')
    print(
        f'GitHub users without enough repos: {USERS_WITHOUT_ENOUGH_REPOS}')
    print(f'Number of users in result file: {USERS_WITHOUT_ENOUGH_REPOS}')


def get_waiting_usernames():
    users, not_found, repos = load_data()
    skip = repos + not_found + list(users.keys())

    input_un = pd.read_csv(INPUT_FILENAME, delimiter=',')[USERNAME_COLUMN_NAME]
    distinct_un = input_un.drop_duplicates()
    return distinct_un[~distinct_un.isin(skip)]


waiting_usernames = get_waiting_usernames()
fetch_data_for(waiting_usernames)
