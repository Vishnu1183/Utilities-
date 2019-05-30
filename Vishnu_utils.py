import pandas as pd
import numpy as np

# getting feature importance of variables with name for XGB model
def feature_importance(model):
    return(pd.DataFrame({'features' : model.get_booster().feature_names, 'importance' : model.feature_importances_}).sort_values('importance',ascending = False))

feat_imp = feature_importance(model = model_name)
