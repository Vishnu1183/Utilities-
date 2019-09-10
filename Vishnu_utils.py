import pandas as pd
import numpy as np

# 1- getting feature importance of variables with name for XGB model
def xgb_feature_imp(model):
    """
    Gives the feature importance along with variable names for an XGB Model.
    """
    return(pd.DataFrame({'features' : model.get_booster().feature_names, 'importance' : model.feature_importances_}).\
           sort_values('importance',ascending = False))

#feat_imp = xgb_feature_imp(model = your_model_name)



# 2- getting ngrams from string
def ngrams(s,n):
    """
    generates the ngrams of a string.
    """
    ngram_s = []
    for i in range(len(s)-(n-1)):
        temp_s= s[i:i+n]
        ngram_s.append(temp_s)
    return ngram_s



# 3- getting jaccard distance from 2 strings 
def jaccard_distance(a,b,q=2):
    """
    Equivalent of jaccard distance from stringdist in R.
    Calculates jaccard distance with ngram = q (default q=2) for 2 strings.  
    a , b the strings for which jaccard distance is to be calculcated.
    q = n of n-gram.
    """
    if a == b:
        return 1 
    def ngrams(s,n):
        ngram_s = []
        for i in range(len(s)-(n-1)):
            temp_s= s[i:i+n]
            ngram_s.append(temp_s)
        return ngram_s
         
    set_a= set(ngrams(s = a,n = q))
    set_b= set(ngrams(s = b,n = q))
    return 1- (len(set_a.intersection(set_b)) /len(set_a.union(set_b)))


# 4 - getting intercept and coefficients of logistic regression model

def logreg_coef(model,data):    
    """
    Gives the intercepts and coefficients along with variable names for logistic regression.
    model : name of the logistic model
    data :  data on which logistic model was fit, column order must be same as that of the
            data on which model was fit
    """
    intercept = pd.DataFrame({'variable' : 'intercept', 'coefficient' : model.intercept_})
    coefficient = pd.DataFrame({'variable' : data.columns, 'coefficient' : model.coef_.transpose().flatten()})
    coefficient = coefficient.reindex(coefficient.coefficient.abs().sort_values(ascending = False).index)
    return(pd.concat([intercept,coefficient], axis = 0).reset_index(drop = True))

#lr_coefs = logreg_coef(model = your_model_name, data = X_train[features])
