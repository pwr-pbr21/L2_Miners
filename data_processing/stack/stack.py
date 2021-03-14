import pandas as pd
import re

gh_username_patterns = [
    "https?:\/\/([^\/]+)\.github\.io.*",
    "https?:\/\/github.com\/([^\/]+)",
    "https?:\/\/www.github.com\/([^\/]+)",
]


def extract_user_name(patterns, url):
    for p in patterns:
        res = re.match(p, url)
        if res is not None:
            return res.group(1)
    return None


stack_data = pd.read_csv('./stack/data/QueryResults.csv')
stack_data = stack_data[~stack_data.AboutMe.isnull()]
stack_data['GithubUrl'] = stack_data.apply(
    lambda row: extract_user_name(gh_username_patterns, row['WebsiteUrl']), axis=1)
stack_data = stack_data[~stack_data.GithubUrl.isnull()]
print()
