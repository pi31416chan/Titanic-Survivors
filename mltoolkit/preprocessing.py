# Imports
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.base import _OneToOneFeatureMixin,TransformerMixin,BaseEstimator
from sklearn.preprocessing import StandardScaler,MinMaxScaler


# Classes
class MyScaler(TransformerMixin,BaseEstimator):
    def __init__(self,scalertype:str=None):
        '''
        Optional scaling method in one class. Allowing user to pass argument
        to decide which scaling method to use, or to skip it.

        Parameters:
        ----------
        scalertype: str | {'std','minmax','skip'}, Default: None
        specifying the scaler to transform the data
        '''
        if scalertype: 
            if scalertype in ['std','minmax','skip']:
                self.scalertype = scalertype
            else:
                raise ValueError('Specify scaler correctly. Options:{"std","minmax","skip"}')
        else: self.scalertype = None
    def fit(self,X,y=None,scalertype:str=None):
        '''
        scalertype: str | {'std','minmax','skip'}, Default: None
        specifying the scaler to transform the data
        '''
        if scalertype:
            if self.scalertype != scalertype:
                self.scalertype = scalertype
        if self.scalertype == 'std':
            self.scaler = StandardScaler()
        elif self.scalertype == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scalertype == 'skip':
            return self
        else:
            raise ValueError('Specify scaler correctly. Options:{"std","minmax","skip"}')
        return self
    def transform(self,X,scalertype:str=None):
        '''
        scalertype: str | {'std','minmax'}, Default: None
        specifying the scaler to transform the data
        '''
        if scalertype:
            if self.scalertype != scalertype:
                self.scalertype = scalertype
        if self.scalertype == 'std':
            self.scaler = StandardScaler()
        elif self.scalertype == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scalertype == 'skip':
            if type(X) == pd.DataFrame:
                return X.values
            elif type(X) == np.ndarray:
                return X
        else:
            raise ValueError('Specify scaler correctly. Options:{"std","minmax","skip"}')
        return self.scaler.fit_transform(X)

class ImputeFromOtherColumn(BaseEstimator,TransformerMixin):
    def __init__(self,impute_on:str,on_col:str,strategy:str='mean'):
        '''
        Replace missing values using a descriptive statistic (e.g. mean, median)
        along the target column based on the category that this row instance
        falls on another column.
        
        This imputer accepts pandas.DataFrame only.
        
        Parameters:
        ----------
        impute_on: str
            The target column.
        on_col: str
            The categorical column where the imputer will read the category of 
            the missing value in this column and assign the descriptive 
            statistical value (e.g. mean, median) to replace the missing value.
        strategy: str | {'mean','median'}
        '''
        self.impute_on = impute_on
        self.on_col = on_col
        if strategy not in ['mean','median']:
            raise ValueError("Value entered not in accepted strategies: 'mean' or 'median'")
        self.strategy = strategy
        self.groupby_results = None
        return None
    def fit(self,X,y=None):
        if type(X) is not pd.DataFrame:
            raise ValueError("Please fit pandas.DataFrame only.")
        
        if self.strategy == 'mean':
            self.groupby_results = X.groupby(self.on_col).mean()[self.impute_on]
        elif self.strategy == 'median':
            self.groupby_results = X.groupby(self.on_col).median()[self.impute_on]
        return self
    def transform(self,X,y=None):
        if self.groupby_results is None:
            raise ValueError("Please fit this transformer with data before transforming.")
        if type(X) is not pd.DataFrame:
            raise ValueError("Please transform pandas.DataFrame only.")
        X_copy = X.copy()
        X_null = X_copy[X_copy[self.impute_on].isnull()]
        
        for ind in X_null.index.values:
            val_at_key = X_null.loc[ind,self.on_col]
            X_copy.loc[ind,self.impute_on] = self.groupby_results[val_at_key]
        return X_copy

class ImageAugmentor(BaseEstimator,TransformerMixin):
    def __init__(self,rows,columns):
        '''
        Data Augmentation for image type of training data. This tool takes in an
        image data in 2D-numpy.ndarray or pandas.DataFrame and create 4 additional
        images with 1 pixel shifting to each direction.

        Note:
        ----------
            The output 2D-array is 5 times larger than the original input data.
            Beware of memory consumption.

        Parameters:
        ----------
        rows: int
            The row number of the image data, or the number of pixel in height.
        columns: int
            The column number of the image data, or the number of pixel in width.
        '''
        self.r = rows
        self.c = columns
        self.l = rows * columns
        self.r_zero = np.zeros((1,self.c))
        self.c_zero = np.zeros((self.r,1))
        self._augmented_images = []
        self._augmented_labels = []
        self._X = None
        self._X_fitted = False
        self._y = None
        self._y_fitted = False
        return None
    def fit(self,X,y=None):
        if type(X) == pd.DataFrame:
            self._X = X.values
            self._X_fitted = True
        elif type(X) == np.ndarray:
            self._X = X
            self._X_fitted = True
        else:
            raise ValueError('Input X must be in DataFrame or 2D ndarray.')
        
        if type(y) == pd.Series:
            self._y = y.values
            self._y_fitted = True
        elif type(y) == np.ndarray:
            self._y = y
            self._y_fitted = True
        elif not y:
            pass
        else:
            raise ValueError('Input y must be in Series or 1D ndarray.')
        
        return self
    def transform(self,X,y=None):
        if self._X_fitted == False:
            raise ValueError('Please fit the data before transforming.')
        self._augmented_images = []
        self._augmented_labels = []
        
        for i in range(self._X.shape[0]):
            x = self._X[i].reshape((self.r,self.c))
            if self._y_fitted: label = self._y[i]
            
            self._augmented_images.append(np.concatenate((x[:,1:],self.c_zero),axis=1).flatten())
            self._augmented_images.append(np.concatenate((self.c_zero,x[:,:-1]),axis=1).flatten())
            self._augmented_images.append(np.concatenate((x[1:,:],self.r_zero),axis=0).flatten())
            self._augmented_images.append(np.concatenate((self.r_zero,x[:-1,:]),axis=0).flatten())
            if self._y_fitted: [self._augmented_labels.append(label) for i in range(4)]
        if not self._y_fitted:
            return np.concatenate((self._X,np.asarray(self._augmented_images)),axis=0)
        elif self._y_fitted:
            return np.concatenate((self._X,np.asarray(self._augmented_images)),axis=0),np.concatenate((self._y,np.asarray(self._augmented_labels)))

class ExtractFromString(BaseEstimator,TransformerMixin):
    def __init__(self,pattern):
        '''
        Extracts text from 1D-pandas.DataFrame or pandas.Series according the 
        given regex pattern.

        If a 2D-pandas.DataFrame is passed, only the first column will be 
        operated on.

        Parameters:
        ----------
        pattern: str
            String in regex syntax, must be enclosed in parentheses.

        Examples:
        ----------
            "([0-9]+$)": Extracts numbers from text ending with numbers
            "(^[A-Z]+)": Extracts text from text ended starting with capital letters
            "(^[A-Z]{2})": Extracts first two capital letters from text

            Just encapsulate the regex pattern in parentheses.
        '''
        self.pattern = pattern
        return None
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        if type(X) is pd.DataFrame:
            X = X.iloc[:,0]
        elif type(X) is np.ndarray:
            X = pd.Series(X.flatten())
        else:
            raise ValueError("Only pd.DataFrame or np.ndarray are accepted for transformation")
        X = X.str.extract(self.pattern).values
        return X

class DropNAX(BaseEstimator,TransformerMixin):
    def __init__(self,subset=None,out_df=False,path="System\\"):
        '''
        Drop NA for specified columns.
        
        Parameters:
        ----------
        subset: int, str or list of int/str, Default None
            When 'subset' = None, it will drop NA where NA is found for all rows.
        out_df: bool, Default: False
            When out_df = True, the output is a pandas DataFrame, this enables
            this transformer to be used in chaining with ColumnTransformer.
        path: str, Default: "System\\"
            The path to save the indexer for y data indexing in the transformer
            'DropNAy'.
        '''
        self.subset = subset
        self.out_df = out_df
        self.path = path
        return None
    def fit(self,X,y=None,subset=None):
        if subset is not None:
            self.subset = subset
        return self
    def transform(self,X,y=None):
        if type(X) is np.ndarray:
            X = pd.DataFrame(X)
        X = X.dropna(subset=self.subset)
        
        os.mkdir(self.path)
        joblib.dump(X.index,self.path+"y_indexer",compress=3)
        
        if self.out_df: return X
        else: return X.values

class DropNAy(BaseEstimator,TransformerMixin):
    def __init__(self,out_sr=False,path="System\\"):
        '''
        Drop NA for y using the saved indexer from a previous Drop NA action
        on X by 'DropNAX'.
        
        Parameters:
        ----------
        out_sr: bool, Default: False
            When out_sr = True, the output is a pandas Series.
        path: str, Default: "System\\"
            The path to load the indexer saved by the transformer 'DropNAX'
            for y data indexing.
        '''
        self.out_sr = out_sr
        self.path = path
        self.indexer = None
        return None
    def fit(self,y):
        self.indexer = joblib.load(self.path+"y_indexer")
        return self
    def transform(self,y):
        if type(y) is np.ndarray:
            y = pd.Series(y)
        y = y[self.indexer]
        if self.out_sr: return y
        else: return y.values



# Functions
