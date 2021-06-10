# Data preprocessing

## How to use?
```shell
python prepocessing.py <mode> <stack_data_in> <github_data_in> <data_out>
```
where:
- mode = 0, without FullStack role
- mode = 1, with FullStack role

## Example
```shell
python preprocessing.py 1 ../data/step_1_3_stack_users_out_full_stack.csv ../data/step_2_github_data_full_stack.json ../data/step_3_processed_ground_truth_fs.csv
```