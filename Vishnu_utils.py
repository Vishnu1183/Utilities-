import pandas as pd
import numpy as np

# 1- getting feature importance of variables with name for XGB model
def feature_importance(model):
    return(pd.DataFrame({'features' : model.get_booster().feature_names, 'importance' : model.feature_importances_}).sort_values('importance',ascending = False))

feat_imp = feature_importance(model = model_name)


# 2- getting jaccard distance from 2 strings 
def jaccard_distance(a,b,q=2):
    """
    Equivalent of jaccard distance from stringdist in R.
    Calculates jaccard distance with ngram = 2 for 2 strings  
    a , b the strings for which jaccard distance is to be calculcated
    q = n of n-gram
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
    #intersection = set_a.intersection(set_b)
    #union = set_b.union(set_a)
    return 1- (len(set_a.intersection(set_b)) /len(set_a.union(set_b)))


