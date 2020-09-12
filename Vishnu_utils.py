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
        return 0 
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


# 5 - DIct Vectorizer on train and transforming it on test
def dict_vec(train_set,cols,is_test, test_set):
    """
    returns dict vectorizer on train set or train & test set for chosen columns
    train_set: Dataset on which DV is to be fit
    cols: List of columns of train_set which are to be considered for DV
    is_test: Boolean, If DV is to be transformed on test too
    test_set: Test set on which DV is to be transformed
    """
    from sklearn.feature_extraction import DictVectorizer
    import pandas as pd
    dvec = DictVectorizer(sparse=False)
    if not is_test:
        train_dvec = dvec.fit_transform(train_set[cols].transpose().to_dict().values())
        train_dvec = pd.DataFrame(train_dvec, index = train_set.index, columns = dvec.get_feature_names())
        train_df = pd.concat([train_set.drop(cols, axis = 1),train_dvec], axis = 1)
        return train_df,pd.DataFrame(),dvec
    else:
        train_dvec = dvec.fit_transform(train_set[cols].transpose().to_dict().values())
        train_dvec = pd.DataFrame(train_dvec, index = train_set.index, columns = dvec.get_feature_names())
        train_df = pd.concat([train_set.drop(cols, axis = 1),train_dvec], axis = 1)
        test_dvec = dvec.transform(test_set[cols].transpose().to_dict().values())
        test_dvec = pd.DataFrame(test_dvec, index = test_set.index, columns = dvec.get_feature_names())
        test_df = pd.concat([test_set.drop(cols, axis = 1),test_dvec], axis = 1)
        return train_df, test_df,dvec

df_train,df_test,dv = dict_vec(train, ['Sex','SibSp', 'Parch', 'Pclass', 'Embarked'],is_test = True,test_set =test)


def ks_gini_metrics(base, probability="Probability_of_event", event_name="Event", total_name="Total",
                   ascending=False):
    """
    By : Kanishk Dogar
    Get the KS/Gini coefficient from a pivot table with 3 specific columns - probability/score band, event count and total count
    Parameters
    ----------
    base: pd.DataFrame
        A pandas dataframe created using a group by operation with a probability or score band.
        The pivot should be created with margins=False
    probability: str
        column name of the column that contains the band
    event_name: str
        column name of the column that contains the event count for every band
    total_name: str
        column name of the column that contains the total count for every band
    ascending: bool
        Order of the probability or score band in the final table
    """
    base = base.loc[:, [probability, event_name, total_name]]
    base = base[base.loc[:, total_name].notnull()]
    base = base.append(pd.DataFrame(data={event_name: np.sum(base.loc[:, event_name]),
                                          total_name: np.sum(base.loc[:, total_name]),
                                          probability: "All"},index=["All"]), ignore_index=True, sort=True)

    base = base[base.loc[:, probability] != "All"]. \
    sort_values(by=probability, ascending=ascending). \
    append(base[base.loc[:, probability] == "All"], sort=True).loc[:, [probability, total_name, event_name]]

    base["Non_"+event_name] = base.loc[:, total_name] - base.loc[:, event_name]
    base["Cumulative_Non_"+event_name] = base.loc[:, "Non_"+event_name].cumsum()
    base.loc[base[base.loc[:, probability] == "All"].index, "Cumulative_Non_"+event_name] = \
    base.loc[base[base.loc[:, probability] == "All"].index, "Non_"+event_name]
    base["Cumulative_"+event_name] = base.loc[:, event_name].cumsum()
    base.loc[base[base.loc[:, probability] == "All"].index, "Cumulative_Event"] = \
    base.loc[base[base.loc[:, probability] == "All"].index, "Event"]
    base["Population_%"] = base.loc[:, total_name]/base[base.loc[:, probability] == "All"].loc[:, total_name].values
    base["Cumulative_Non_"+event_name+"_%"] = \
    base.loc[:, "Cumulative_Non_"+event_name]/base[base.loc[:, probability] == "All"].loc[:, "Cumulative_Non_"+event_name].values
    base["Cumulative_"+event_name+"_%"] = \
    base.loc[:, "Cumulative_"+event_name]/base[base.loc[:, probability] == "All"].loc[:, "Cumulative_"+event_name].values
    base["Difference"] = base["Cumulative_"+event_name+"_%"] - base["Cumulative_Non_"+event_name+"_%"]
    base[event_name+"_rate"] = base.loc[:, event_name]/base.loc[:, total_name]

    base["Gini"] = ((base["Cumulative_"+event_name+"_%"]+base["Cumulative_"+event_name+"_%"].shift(1).fillna(0))/2) \
    *(base["Cumulative_Non_"+event_name+"_%"]-base["Cumulative_Non_"+event_name+"_%"].shift(1).fillna(0))

    base.loc[base[base.loc[:, probability] == "All"].index, "Gini"] = np.nan
    model_KS = np.max(base[base.loc[:, probability] != "All"].Difference)*100
    model_Gini = (2*(np.sum(base[base.loc[:, probability] != "All"].Gini))-1)*100
    return base, model_KS, model_Gini


def proba_score_performance(actual, prediction, test_set=False, train_bins=0, event=1, target="target",
                              probability="Probability_of_event", event_name="Event", total_name="Total",
                             ascending=False, bins=10):
    """
    By : Kanishk Dogar
    Get the KS/Gini coefficient and the table to create the lorenz curve with 10 bins
    Parameters
    ----------
    actual: pd.Series
        A pandas Series with the target values
    prediction: np.array
    A numpy array with the predicted probabilities or score. 1 D array with the same length as actual
    test_set: bool
        Set to False if the prediction needs to be binned using quantiles. True if training set bins are present
        train_bins = a list of cut points if this is True
    train_bins: list
        list of cutpoints that bin the training set into 10 parts
    event: integer
        The target value the gini table needs to be created for
   target: str
        The name of the target column in `actual`. If the name does not match, it will be changed to the user input
    probability: str
        column name of the column that contains the band
    event_name: str
        column name of the column that contains the event count for every band
    total_name: str
        column name of the column that contains the total count for every band
    ascending: bool
        Order of the probability or score band in the final table
    bins: integer
        no. of quantile bins to create
    """
    actual.name = target
    performance = pd.concat([pd.DataFrame(prediction, columns=[probability], index=actual.index), pd.DataFrame(actual)], axis=1)
    performance.loc[:, target] = np.where(performance.loc[:, target] == event, 1, 0)

    if test_set:
        performance[probability] = pd.cut(performance.loc[:, probability], bins=train_bins, include_lowest=True)
    else:
        _, train_bins = pd.qcut(performance.loc[:, probability].round(12), bins, retbins=True, duplicates="drop")
        train_bins[0] = np.min([0.0, performance.loc[:, probability].min()])
        train_bins[train_bins.shape[0]-1] = np.max([1.0, performance.loc[:, probability].max()])
        performance[probability] = pd.cut(performance.loc[:, probability], bins=train_bins, include_lowest=True)   

    performance = pd.concat([performance.groupby(by=probability)[target].sum(),
                     performance.groupby(by=probability)[target].count()], axis=1)
    performance[probability] = performance.index
    performance.columns = [event_name, total_name, probability]

    performance, model_KS, model_Gini = ks_gini_metrics(performance, probability=probability, event_name=event_name,
                                                       total_name=total_name, ascending=ascending)

    if test_set:
        return performance, model_KS, model_Gini
    else:
        return performance, model_KS, model_Gini, train_bins

    

