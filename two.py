import numpy as np
from sklearn.datasets import make_classification
x,y=make_classification(n_samples=1000,n_features=20,n_informative=20,n_classes=2,random_state=42)
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
param_dist={
    "max_depth":[3,None],
    "max_features":randint(4,9),
    "min_samples_leaf":randint(1,6),
    "criterion":['gine','entropy']

}
tree=DecisionTreeClassifier()
tree_cv=RandomizedSearchCV(tree,param_dist,cv=5)
tree_cv.fit(x,y)
print("dicision:{}",format(tree_cv.best_params_))
print("best score:{}",format(tree_cv.best_score_))