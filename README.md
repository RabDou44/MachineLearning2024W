# Work Plan

## Datasets:
- Finding 4th datasets: 30 min

## Code:
- Data Preprocessing: 2h
    - Normalization
    - Handling missing values
    - Data exploration
- Building pipeline [general structure]: 3h
    - Write the stage for Holdout / CV
- Decision Tree: 1h
- SVM (SVC): 1h
- KNeighbours: 1h
- Evaluation => contingency table / plotting (measures/statistics): 3h
For binary classification:
"accuracy","f1-score","precision","recall"
For multi-classification:
"accuracy","f1-weighted","precision-weighted","recall-weighted"

## Report:
- Methodology
- Data Exploration
- Results of classification
- Conclusion

# Requirements:
python 3.12.0
pip-24.3.1

## Set up environment

WINDOWS CMD:
<!-- build venv -->
python -m venv venv\ 
<!-- activate env -->
venv\Scripts\activate.bat
<!-- loading requirements(packages needed for project) -->
python -m pip install -r requirements.txt
<!-- update requirements -->
python -m pip freeze > requirements.txt 
<!-- deactive -->
venv\Scripts\deactivate.bat

**Remember:**
Remember to load venv for jupyter

[How to set up environment for other OS's] (https://realpython.com/python-virtual-environments-a-primer/)
