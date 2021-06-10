# L2 Miners

Mining the Technical Roles of GitHub Users 

## Directory structure
- common - common utils for all steps
- data - data used in research
- data_original - data used in base research
- step1_stack - Stack data early preprocessing
- step2_github - GitHub data downloading via GraphQL API
- step3_preprocesing - data preprocessing
- step4_model - machine learning model and tools

Each folder has a README file with usage description.

## Environment
To correctly run script in this repository you need Python3.8 with packages in 
`requirements.txt`. The recommended approach is to create a Python virtual environment 
and install libraries via `pip install -r requirements.txt`