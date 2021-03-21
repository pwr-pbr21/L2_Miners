import json
import os
from typing import Dict

import pandas as pd
import requests as rq

AUTHORIZATION_KEY_FILE = 'auth.key'

INPUT_FILENAME = 'input.csv'
BIO_COLUMN_NAME = 'Bio'
USERNAME_COLUMN_NAME = 'GithubUsername'

OUTPUT_FILENAME = 'output.json'
OUTPUT_ATTRIBUTE_NAME = {
    'author_name': 'author_name',
    'author_login': 'author_login',
    'username': 'username',
    'bio': 'bio',
    'commits_total': 'commits_total',
    'commits_authored': 'commits_authored',
    'repos': 'repositories',
    'repo_name': 'name',
    'repo_topics': 'topics',
    'repo_commits': "commits",
    'repo_languages': 'languages',
    'repo_description': "description",
}

SAVE_STEP = 1
PER_PAGE = 100

REPOS_URL = 'https://api.github.com/users/<username>/repos'
TOPICS_URL = 'https://api.github.com/repos/<username>/<repo_name>/topics'
COMMITS_URL = 'https://api.github.com/repos/<username>/<repo_name>/commits'
LANGUAGES_URL = 'https://api.github.com/repos/<username>/<repo_name>/languages'

AUTHORIZATION = (lambda: open(AUTHORIZATION_KEY_FILE).read())()
REQUEST_COUNTER = 0


def load_data():
    if os.path.isfile(OUTPUT_FILENAME):
        with open(OUTPUT_FILENAME, "r") as outfile:
            return json.load(outfile)
    else:
        return {}


def save_data(data):
    with open(OUTPUT_FILENAME, "w+") as outfile:
        return json.dump(data, outfile)


def update_data(additional_data):
    loaded = load_data()
    merged = {**loaded, **additional_data}
    save_data(merged)


def build_url(template: str, path_params: Dict[str, str]):
    url = template
    for key, value in path_params.items():
        url = url.replace(f'<{key}>', value)
    return url


def make_request(url: str, headers: Dict[str, str] = {}, query_params: Dict[str, str] = {}):
    global REQUEST_COUNTER
    REQUEST_COUNTER = REQUEST_COUNTER + 1
    print(f'GET {url}  path: {query_params}')
    response = rq.get(url, headers=headers, params=query_params)
    # Git repository is empty
    if response.status_code == 409:
        return []
    elif response.status_code == 200:
        return response.json()
    else:
        print(response.json())
        print(f'Occurred unexpected status code: {response.status_code}')
        exit(1)


def fetch_data(url_template: str, path_params: Dict[str, str],
               query_params: Dict[str, str] = {}, headers: Dict[str, str] = {}):
    url = build_url(url_template, path_params)
    headers['Authorization'] = f'token {AUTHORIZATION}'
    return make_request(url, headers=headers, query_params=query_params)


def fetch_all_data(url_template: str, path_params: Dict[str, str], query_params: Dict[str, str] = {}):
    url = build_url(url_template, path_params)
    headers = {'Authorization': f'token {AUTHORIZATION}'}
    query_params['per_page'] = PER_PAGE

    page = 1
    query_params['page'] = page
    data = make_request(url, headers, query_params)
    all_data = data

    while len(data) == PER_PAGE:
        page += 1
        query_params['page'] = page
        data = make_request(url, headers, query_params)
        all_data += data
    return all_data


def fetch_topics(username: str, repo_name: str):
    path_params = {'username': username, 'repo_name': repo_name}
    headers = {'accept': 'application/vnd.github.mercy-preview+json'}
    data = fetch_data(TOPICS_URL, path_params, headers=headers)
    names = data.get('names')
    if names:
        return names
    else:
        return []


def fetch_languages(username: str, repo_name: str):
    path_params = {'username': username, 'repo_name': repo_name}
    return fetch_data(LANGUAGES_URL, path_params)


def extract_data_from_commit(commit: dict):
    name = commit['name']
    topics = commit['topics']
    return {
        OUTPUT_ATTRIBUTE_NAME['repo_name']: name,
        OUTPUT_ATTRIBUTE_NAME['repo_topics']: topics
    }


def extract_data_from_commit(commitObj: dict):
    author_name, author_login = '', ''
    commit = commitObj.get('commit')
    if commit:
        author = commit.get('author')
        if author:
            author_name = author.get('name')
    author = commitObj.get('author')
    if author:
        author_login = author.get('login')
    return {
        OUTPUT_ATTRIBUTE_NAME['author_name']: author_name,
        OUTPUT_ATTRIBUTE_NAME['author_login']: author_login
    }


def fetch_commits(username: str, repo_name: str):
    path_params = {'username': username, 'repo_name': repo_name}
    data = fetch_all_data(COMMITS_URL, path_params)
    extracted = list(map(extract_data_from_commit, data))
    authored = list(filter(lambda c: c.get(OUTPUT_ATTRIBUTE_NAME['author_name']) == username or
                                     c.get(OUTPUT_ATTRIBUTE_NAME['author_login']) == username, extracted))
    return {
        OUTPUT_ATTRIBUTE_NAME['commits_total']: len(data),
        OUTPUT_ATTRIBUTE_NAME['commits_authored']: len(authored)
    }


def fetch_repos(username: str):
    path_params = {'username': username}
    data = fetch_all_data(REPOS_URL, path_params)

    repos = {}
    for repo in data:
        repo_name = repo['name']
        commits = fetch_commits(username, repo_name)
        languages = fetch_languages(username, repo_name)
        topics = fetch_topics(username, repo_name)
        repos[repo_name] = {
            OUTPUT_ATTRIBUTE_NAME['repo_topics']: topics,
            OUTPUT_ATTRIBUTE_NAME['repo_name']: repo_name,
            OUTPUT_ATTRIBUTE_NAME['repo_commits']: commits,
            OUTPUT_ATTRIBUTE_NAME['repo_languages']: languages,
            OUTPUT_ATTRIBUTE_NAME['repo_description']: repo['description'],
        }
    return repos


def fetch_data_for(names_and_bio: pd.DataFrame):
    collection = {}
    size = names_and_bio.shape[0]
    for idx, row in names_and_bio.iterrows():
        print(f'Progress {idx} of {size}, Total requests: {REQUEST_COUNTER}')
        username = row[USERNAME_COLUMN_NAME]
        bio = row[BIO_COLUMN_NAME]
        repos = fetch_repos(username)
        collection[username] = {
            OUTPUT_ATTRIBUTE_NAME['username']: username,
            OUTPUT_ATTRIBUTE_NAME['bio']: bio,
            OUTPUT_ATTRIBUTE_NAME['repos']: repos
        }
        if idx % SAVE_STEP == 0:
            update_data(collection)
            collection = {}

    print(f'Progress {size} of {size}, Total requests: {REQUEST_COUNTER}')
    update_data(collection)


input_data = pd.read_csv(INPUT_FILENAME, delimiter=',')
distinct = input_data.drop_duplicates(USERNAME_COLUMN_NAME)
names_and_bio = input_data.loc[:, [USERNAME_COLUMN_NAME, BIO_COLUMN_NAME]]

loaded = load_data()
skip_keys = loaded.keys()
waiting = names_and_bio.loc[~names_and_bio[USERNAME_COLUMN_NAME].isin(skip_keys)]
fetch_data_for(waiting)
