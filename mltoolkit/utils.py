# Imports
import os
import joblib


# Classes
def dump_model(estimator,path = "Trained Models\\",filename=None,yhat=None,scores=None,compress=5):
    '''
    Dump the objects passed as arguments into .pkl file.
    '''
    try:
        os.mkdir(path)
    except:
        pass
    # Dump estimator
    if not filename: filename = str(estimator)[:str(estimator).find('(')]
    joblib.dump(estimator,path+filename+".pkl",compress=compress)
    # Dump yhat
    if yhat is not None:
        joblib.dump(yhat,path+filename+"_yhat"+".pkl",compress=compress)
    # Dump cv scores
    if scores is not None:
        joblib.dump(scores,path+filename+"_scores"+".pkl",compress=compress)



# Functions
