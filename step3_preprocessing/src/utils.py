import json
from ast import literal_eval

import pandas as pd


def read_stack_data(stack_path) -> pd.DataFrame:
    so_data = pd.read_csv(stack_path)
    so_data.Role = so_data.Role.apply(literal_eval)
    return so_data


def read_github_data(github_path):
    with open(github_path) as github_file:
        gh_data = json.load(github_file)
    return gh_data
