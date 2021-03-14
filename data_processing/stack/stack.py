import pandas as pd
import re
import sys

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
]


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


def main(argv):
    stack_data = pd.read_csv(argv[0])
    stack_data = stack_data[~stack_data.AboutMe.isnull()]

    stack_data['GithubUrl'] = stack_data.apply(
        lambda row: extract_user_name(gh_username_patterns, row['WebsiteUrl']),
        axis=1)
    stack_data = stack_data[~stack_data.GithubUrl.isnull()]

    stack_data['Role'] = stack_data.apply(
        lambda row: extract_roles(role_patterns, row['AboutMe']), axis=1)
    stack_data = stack_data[stack_data['Role'].apply(lambda x: len(x)) > 0]

    stack_data.to_csv(argv[1], columns=['Id', 'GithubUrl', 'Role'], index=False)


if __name__ == "__main__":
    assert len(sys.argv) == 3
    main(sys.argv[1:])
