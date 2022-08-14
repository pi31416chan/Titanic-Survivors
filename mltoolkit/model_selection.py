# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score,precision_score,recall_score,\
                            f1_score,ConfusionMatrixDisplay



# Classes



# Functions
def test_water(estimators:list,X:np.ndarray,y:list|np.ndarray,cv:int=4,
               scoring:str|list='accuracy',n_jobs:int=2,
               return_train_score=False) -> pd.DataFrame:
    '''
    Doing cross-validation on the passed list of estimators. Compile and output
    the test results as a pandas.DataFrame

    Parameters:
    ----------
    estimators: list or array of estimator
        The list of estimators to be evaluated.
    X: 2D-ndarray
        Refers to documentations on sklearn.model_selection.cross_validate.
    y: 1D-ndarray
        Refers to documentations on sklearn.model_selection.cross_validate.
    cv: int
        Refers to documentations on sklearn.model_selection.cross_validate.
    scoring: str
        Refers to documentations on sklearn.model_selection.cross_validate.
    n_jobs: int
        Refers to documentations on sklearn.model_selection.cross_validate.
    return_train_score: bool
        Refers to documentations on sklearn.model_selection.cross_validate.
    
    Returns:
    ----------
    DataFrame: pandas.DataFrame
    '''
    results = None
    for estimator in estimators:
        model = estimator()
        name = str(model)[:str(model).find('(')]
        score_dict = cross_validate(model,X,y,cv=cv,scoring=scoring,n_jobs=n_jobs,
                                    return_train_score=return_train_score)
        
        if results is None:
            results = pd.concat((pd.DataFrame([name]*cv,columns=['estimator']),
                                 pd.DataFrame(score_dict)),axis=1,join='inner')
        else:
            results = pd.concat((results,
                                 pd.concat((pd.DataFrame([name]*cv,columns=['estimator']),
                                            pd.DataFrame(score_dict)),axis=1,join='inner')),
                                 axis=0)
    else:
        return results.reset_index().rename({'index':'run'},axis=1)

def measure_classifier(estimator,trainortest:str,y_true,y_pred) -> dict:
    '''
    Quickly measure the performance of an estimator using common scoring method
    for classification.
    
    Parameters:
    ----------
    estimator: Estimator object
        Mainly for the use of extracting the name and labeling the output dict.
    trainortest: str
        To label the results as train or test measurement.
    y_true: 1D-ndarray
        True labels.
    y_pred: 1D-ndarray
        Predicted labels.
    
    Returns:
    ----------
    dict: Dictionary
    '''
    acc = accuracy_score(y_true,y_pred)
    prec = precision_score(y_true,y_pred)
    rec = recall_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred)
    
    name = str(estimator)
    # [:str(estimator).find('(')]
    print("Estimator:",name)
    print("Train/Test:",trainortest.title())
    print("Accuracy:",acc)
    print("Precision:",prec)
    print("Recall:",rec)
    print("F1:",f1)
    f = plt.figure(figsize=(6,6))
    ax = f.gca()
    cmdisp = ConfusionMatrixDisplay.from_predictions(y_true,y_pred,
                                                     display_labels=['Dead','Survived'],
                                                     cmap='Greys',ax=ax)
    plt.show()
    results = {
        "Estimator":name,
        "Train/Test":trainortest.title(),
        "Accuracy":acc,
        "Precision":prec,
        "Recall":rec,
        "F1":f1,
        "Confusion Matrix":cmdisp
    }
    return results