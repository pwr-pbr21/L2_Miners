import re
import sys

import pandas as pd

gh_username_patterns = [
    r'https?:\/\/www\.([^\/\.]+)\.github\.io.*',
    r'https?:\/\/([^\/\.]+)\.github\.io.*',
    r'https?:\/\/github\.com\/([^\/\.]+)',
    r'https?:\/\/www\.github\.com\/([^\/\.]+)',
]

role_patterns = [
    {
        'name': 'Frontend',
        'pattern': '.*front.{0,1}end.*'
    },
    {
        'name': 'Backend',
        'pattern': '.*back.{0,1}end.*'
    },
    {
        'name': 'DevOps',
        'pattern': '.*dev.{0,1}ops.*'
    },
    {
        'name': 'DataScience',
        'pattern': '.*data.{0,1}scientist.*'
    },
    {
        'name': 'Mobile',
        'pattern': '.*mobile.*'
    },
    {
        'name': 'FullStack',
        'pattern': '.*full.{0,1}stack.*'
    },
]


def extract_user_name(patterns, url):
    for p in patterns:
        res = re.match(p, url)
        if res is not None:
            return res.group(1)
    return None


def extract_roles(_role_patterns, description, is_fullstack_considered):
    if not is_fullstack_considered:
        _role_patterns = list(filter(lambda x : x['name'] != 'FullStack', role_patterns))
    detected_roles = []
    for r in _role_patterns:
        res = re.match(r['pattern'], description, flags=re.IGNORECASE)
        if res is not None:
            detected_roles.append(r['name'])
    return detected_roles


def run_stack(is_fullstack_considered, data_in, data_out):
    stack_data = pd.read_csv(data_in)
    stack_data = stack_data[~stack_data.AboutMe.isnull()]

    stack_data['GithubUrl'] = stack_data.apply(
        lambda row: extract_user_name(gh_username_patterns, row['WebsiteUrl']),
        axis=1)
    stack_data = stack_data[~stack_data.GithubUrl.isnull()]

    stack_data['Role'] = stack_data.apply(
        lambda row: extract_roles(role_patterns, row['AboutMe'], is_fullstack_considered), axis=1)
    stack_data = stack_data[stack_data['Role'].apply(lambda x: len(x)) > 0]

    stack_data.to_csv(data_out, columns=['Id', 'GithubUrl', 'Role'], index=False)


if __name__ == "__main__":
    assert len(sys.argv) == 4
    assert sys.argv[1] == "0" or sys.argv[1] == "1"
    _is_fullstack_considered = False if sys.argv[1] == "0" else True
    _data_in = sys.argv[2]
    _data_out = sys.argv[3]
    run_stack(_is_fullstack_considered, _data_in, _data_out)
