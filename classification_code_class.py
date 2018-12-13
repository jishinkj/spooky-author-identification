#importing libraries
import pandas as pd
import numpy as np

from imblearn.over_sampling import *
from imblearn.combine import *
from imblearn.ensemble import *
from collections import Counter
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import * 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
import pickle
#%%

class Classification():
    def __init__(self):
        pass
    
    def _sampling(self):
        dxdy = pd.concat([self.X,self.y],axis=1)
        print('Original dataset shape {}'.format(Counter(dxdy.iloc[:,-1])))
        sm = self.sampling(random_state=42)
        df_x, df_y = sm.fit_sample(dxdy.iloc[:,:-1], dxdy.iloc[:,-1])
        print('Resampled dataset shape {}'.format(Counter(df_y)))
        
        return df_x, df_y
    
    
    def _prediction_with_sampling(self):
        dfx, dfy = self._sampling()
        X_train, X_test, y_train, y_test = self._train_test(dfx, dfy)
        self.classifier = self.model
        self.classifier.fit(X_train, y_train)
        
        y_pred = self.classifier.predict(X_test)
        return ((round(((accuracy_score(y_test,y_pred))*100))),y_test,y_pred)


    def _train_test(self, IDV, DV):
        '''TODO - Change the values of IDV and DV'''
        
        return train_test_split(IDV, DV, test_size=0.20, random_state=0)


    def _predict(self, test_df):
        y_pred_submission = self.classifier.predict_proba(test_df)
        return y_pred_submission
    
    def grid_search(self, param_grid):
        Xtrain, _, Ytrain, _ = self._train_test(self.X, self.y)
        CV_rfc = GridSearchCV(estimator = self.model, param_grid = param_grid, cv = 5)
        CV_rfc.fit(Xtrain, Ytrain)
#        print("Best parameters are: ", CV_rfc.best_params_)
        return CV_rfc.best_params_

    def pickle_classifier(self):
        with open("del_model_new.pickle", "wb") as f:
            pickle.dump(self.classifier, f, protocol = 2)
        print("Name of the pickle file is 'del_model_new.pickle'")
        # TODO - print the location where the pickle file is saved

    def analysis(self, X, y, sampling, model):
        self.X = X
        self.y = y
        self.sampling = sampling
        self.model = model
        accuracy, y_test, y_pred = self._prediction_with_sampling()
        #_,y=runall()

        # TODO - End analysis method here. 
        val = pd.DataFrame(self.y.value_counts(normalize=True).index)
        valr = val.index.tolist()
        self.n_classes = len (val.index.tolist())
        #binarize for ploting roc curve
        y_test = label_binarize(y_test, classes=valr)
        y_pred = label_binarize(y_pred, classes=valr)
        # Compute ROC curve and ROC area for each class
        self.fpr = dict()
        self.tpr = dict()
        self.roc_auc = dict()
        
        for i in range(self.n_classes):
            self.fpr[i], self.tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
            self.roc_auc[i] = auc(self.fpr[i], self.tpr[i])
        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([self.fpr[i] for i in range(self.n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += interp(all_fpr, self.fpr[i], self.tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= self.n_classes
        # Plot all ROC curves
        # self.plot_roc()
        return (accuracy)

    def plot_roc(self):
        # Plot all ROC curves
        lw=2
        plt.figure()
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(self.n_classes), colors):
            plt.plot(self.fpr[i], self.tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, self.roc_auc[i]))
            
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Roc Curve')
        plt.legend(loc="lower right")
        plt.show()