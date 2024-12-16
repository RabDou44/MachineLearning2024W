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
  - [ ] add wrapper for sklearn methods
  - [ ] building search space/ grid of hyperparameters 
  - [ ] annealing algorithm

## Evaluation:
- [ ] comparison with TPOT
- [ ] auto-sklearn

## Presentation:
- [ ] slides