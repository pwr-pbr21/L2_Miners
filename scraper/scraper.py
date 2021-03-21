import json
import os
from typing import Dict

import pandas as pd
import requests as rq

INPUT_FILENAME = 'input.csv'
OUTPUT_FILENAME = 'output.json'
AUTHORIZATION_KEY_FILE = 'auth.key'

USERNAME_COLUMN_NAME = 'GithubUsername'
OUTPUT_ATTRIBUTE_NAME = {
    'user_name': 'username',
    'user_id': 'id',
    'user_bio': 'bio',
    'repos': 'repositories',
    'commits_total': 'commits_total',
    'commits_authored': 'commits_authored',
    'repo_name': 'name',
    'repo_topics': 'topics',
    'repo_commits': "commits",
    'repo_languages': 'languages',
    'repo_description': "description",
}

with open(AUTHORIZATION_KEY_FILE) as AUTH_FILE:
    AUTH_TOKEN = AUTH_FILE.read()

GITHUB_GRAPHQL_ENDPOINT = 'https://api.github.com/graphql'
GITHUB_GRAPHQL_HEADERS = {
    'Accept': 'application/vnd.github.hawkgirl-preview+json',
    'Authorization': f'token {AUTH_TOKEN}'
}

SAVE_STEP = 10
REPOS_PER_PAGE = 100
MAX_ATTEMPTS_NUMBER = 3

USER_GRAPHQL_QUERY = 'query ($login: String!) { user(login: $login) { id bio }}'
REPOSITORY_GRAPHQL_QUERY = 'query ($login: String!, $userId: ID!, $repoCursor: String) {user(login: $login) {repositories(first: ' + str(
    REPOS_PER_PAGE) + ', after: $repoCursor) {edges {cursor node {name description dependencyGraphManifests {nodes {dependencies {nodes {packageName}}}} languages(first: 100, orderBy: {field: SIZE, direction: DESC}) {nodes {name}} topics: repositoryTopics(first: 100) {nodes {topic {name}}} defaultBranchRef {target {...on Commit {totalCommits: history {totalCount} userCommits: history(author: {id: $userId}) {totalCount}}}}}}}} }'

REQUEST_COUNTER = 0


def load_data():
    if os.path.isfile(OUTPUT_FILENAME):
        with open(OUTPUT_FILENAME, "r") as outfile:
            return json.load(outfile)
    else:
        return {}


def save_data(data):
    with open(OUTPUT_FILENAME, "w+") as outfile:
        json.dump(data, outfile, indent=2)


def update_data(additional_data):
    loaded = load_data()
    merged = {**loaded, **additional_data}
    save_data(merged)


def make_request(query: str, variables: Dict[str, str] = {}):
    global REQUEST_COUNTER
    status_code = 0
    repeats = 0
    while status_code != 200:
        repeats = repeats + 1
        REQUEST_COUNTER = REQUEST_COUNTER + 1
        print(f'REQUEST VARIABLES {variables}')

        body = {
            "query": query,
            "variables": variables
        }
        response = rq.post(GITHUB_GRAPHQL_ENDPOINT, headers=GITHUB_GRAPHQL_HEADERS,
                           data=json.dumps(body))
        status_code = response.status_code
        if status_code == 200:
            break
        print(response.json())
        print(response.request.body)
        print(f'Occurred unexpected status code: {response.status_code}')
        if repeats >= MAX_ATTEMPTS_NUMBER:
            exit(1)
    return response.json().get('data')


def fetch_all_repos_data(username: str, user_id: str):
    variables = {"login": username, "userId": user_id}
    data = make_request(REPOSITORY_GRAPHQL_QUERY, variables)
    edges = data.get('user').get('repositories').get('edges')
    all_edges = edges

    while len(edges) == REPOS_PER_PAGE:
        cursor = edges[-1].get('cursor')
        variables['repoCursor'] = cursor
        data = make_request(REPOSITORY_GRAPHQL_QUERY, variables)
        edges = data.get('user').get('repositories').get('edges')
        all_edges += edges
    return all_edges


def fetch_repos(username: str, user_id: str):
    edges_list = fetch_all_repos_data(username, user_id)
    repos = {}
    for edge in edges_list:
        repo = edge.get('node')
        name = repo.get('name')
        description = repo.get('description')
        branch = repo.get('defaultBranchRef').get('target')
        topics = list(map(lambda e: e.get('name'), repo.get('topics').get('nodes')))
        languages = list(map(lambda e: e.get('name'), repo.get('languages').get('nodes')))
        commits = {
            OUTPUT_ATTRIBUTE_NAME['commits_authored']: branch.get('userCommits').get('totalCount'),
            OUTPUT_ATTRIBUTE_NAME['commits_total']: branch.get('totalCommits').get('totalCount'),
        }
        repos[name] = {
            OUTPUT_ATTRIBUTE_NAME['repo_name']: name,
            OUTPUT_ATTRIBUTE_NAME['repo_topics']: topics,
            OUTPUT_ATTRIBUTE_NAME['repo_commits']: commits,
            OUTPUT_ATTRIBUTE_NAME['repo_languages']: languages,
            OUTPUT_ATTRIBUTE_NAME['repo_description']: description,
        }
    return repos


def fetch_user(username: str):
    variables = {"login": username}
    data = make_request(USER_GRAPHQL_QUERY, variables)
    user = data.get('user')
    return {
        OUTPUT_ATTRIBUTE_NAME['user_name']: username,
        OUTPUT_ATTRIBUTE_NAME['user_bio']: user.get('bio'),
        OUTPUT_ATTRIBUTE_NAME['user_id']: user.get('id')
    }


def fetch_data_for(usernames: pd.Series):
    collection = {}
    size = usernames.size
    for idx, username in usernames.items():
        print(f'Progress {idx} of {size}, Total requests: {REQUEST_COUNTER}')
        user = fetch_user(username)
        user_id = user[OUTPUT_ATTRIBUTE_NAME['user_id']]
        repos = fetch_repos(username, user_id)
        user[OUTPUT_ATTRIBUTE_NAME['repos']] = repos
        collection[username] = user
        if idx % SAVE_STEP == 0:
            update_data(collection)
            collection = {}

    print(f'Progress {size} of {size}, Total requests: {REQUEST_COUNTER}')
    update_data(collection)


already_saved = load_data()
input_usernames = pd.read_csv(INPUT_FILENAME, delimiter=',')[USERNAME_COLUMN_NAME]

distinct_usernames = input_usernames.drop_duplicates()
distinct_usernames.isin(already_saved.keys())
waiting_usernames = distinct_usernames[~distinct_usernames.isin(already_saved.keys())]

fetch_data_for(waiting_usernames)
