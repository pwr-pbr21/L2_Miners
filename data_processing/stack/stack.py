import pandas as pd
import re

gh_username_patterns = [
    r'https?:\/\/([^\/]+)\.github\.io.*',
    r'https?:\/\/github.com\/([^\/]+)',
    r'https?:\/\/www.github.com\/([^\/]+)',
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
]

# re.match({"name": "DevOps", "pattern": r"\bdev.{0,1}ops\b"}["pattern"], "i am a devops someone", flags=re.IGNORECASE)


def extract_user_name(patterns, url):
    for p in patterns:
        res = re.match(p, url)
        if res is not None:
            return res.group(1)
    return None


def extract_roles(roles, description):
    detected_roles = []
    for r in roles:
        res = re.match(r['pattern'], description, flags=re.IGNORECASE)
        if res is not None:
            detected_roles.append(r['name'])
    return detected_roles


stack_data = pd.read_csv('./stack/data/QueryResults.csv')
stack_data = stack_data[~stack_data.AboutMe.isnull()]

stack_data['GithubUrl'] = stack_data.apply(
    lambda row: extract_user_name(gh_username_patterns, row['WebsiteUrl']),
    axis=1)
stack_data = stack_data[~stack_data.GithubUrl.isnull()]

stack_data['Role'] = stack_data.apply(
    lambda row: extract_roles(role_patterns, row['AboutMe']), axis=1)
stack_data = stack_data[stack_data['Role'].apply(lambda x: len(x)) > 0]
