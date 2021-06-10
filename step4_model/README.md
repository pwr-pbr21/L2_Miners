# Learning model

## How to use?
```shell
python start.py <task>
```
where:
- task: optional parameter for specifying task number, if not supplied will run all tasks.
    - 1 - Classifier accuracy
    - 2 - Feature relevance
    - 3 - Correlation between technical roles
    - 4 - Classifier accuracy for Fullstack developers
    - 5 - Hyper parameter tuning

**Make sure to specify paths for input/output data and plots in `data.py`**

## Example
```shell
python start.py 1
```