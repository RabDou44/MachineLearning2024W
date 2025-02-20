# Work plan

## Goal: Implement annealing algorithm for automated  selection/configuration of ml algorithm
- prepare datasets 4x (classification) 
- selection 5 methods:
    - Decision Tree {"depth":range(1, 100), "criterion": ["gini", "entropy"]} 
    - SVM (SVC) {"kernel": [linear, rbf, poly]}
    - KNeighbours {"":[]}
    - Random Forest [] 
    - Logistic []
  
- metrics:
  -  accuracy

- [ ] main algorithm implementation (classification - python class):
  - [x] add wrapper for sklearn methods
  - [x] building search space/ grid of hyperparameters:
      - graph representation of grid
      - binary encoding of parameters 
  - [x] annealing algorithm
      - introduction of temperature parameter
      - [ ] provide time stop criteria
- [x] Big daddy wrapper
  - [x] fix time limit (1h for dataset)
  - [x] split for 5 classifiers
  - [x] time estimator to +- find the required level
- [ ] Evaluation of state-of-the-art (Accuracy and selected pipeline) across all datasets
  - [ ] split the dataset 80% training and 20% for evaluation.

## Evaluation:
- [ ] comparison with TPOT
- [ ] auto-sklearn

## Presentation:
- [ ] slides
