import numpy as np
import pandas as pd
import xgboost as xgb
from time import time
from sklearn.pipeline import Pipeline
from sklearn import model_selection, preprocessing

from sklearn.decomposition import TruncatedSVD
from xgboost.sklearn import XGBRegressor
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV

file_train = '../data/raw/train.csv'
file_test = '../data/raw/test.csv'
df_train = pd.read_csv(file_train)
df_test = pd.read_csv(file_test)

y = 'price_doc'
exclusions = ["id", "timestamp", "price_doc"]
X = [x for x in df_train.columns if x not in exclusions]


def encode_cat_vars(df):
    for c in X:
        if df[c].dtype == 'object':
            label = preprocessing.LabelEncoder()
            label.fit(list(df[c].values))
            df[c] = label.transform(list(df[c].values))


encode_cat_vars(df_train)

# -----------------------------------------------------
#
#  XGBoost Randomized Grid Search
#
# -----------------------------------------------------



pipeline = Pipeline([
    ('reg', XGBRegressor(nthread = 1, objective = 'reg:linear'))
])

# https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
# Note: Parameters of pipelines can be set using '__' separated parameter names:
parameters = {
    'reg__n_estimators': st.randint(100, 1000),
    'reg__max_depth': st.randint(4, 15),
    'reg__learning_rate': st.uniform(0.05, 0.4),
    'reg__colsample_bytree': st.beta(10, 1),
    'reg__subsample': st.beta(10, 1),
    'reg__gamma': st.uniform(0, 10),
    'reg__reg_alpha': st.expon(0, 50),
    'reg__min_child_weight': st.expon(3, 50)
    }

random_search = RandomizedSearchCV(pipeline, parameters, n_jobs=2, verbose=1, refit = False)

t0 = time()
# random_search.fit(df_train.loc[0:1000,X], df_train.loc[0:1000,y])
random_search.fit(df_train[X], df_train[y])

print("done in %0.3fs \n" % (time() - t0))
print("Best score: %0.3f" % random_search.best_score_) # Best score: 0.29

print("Best parameters set:")
best_parameters = random_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))





# Train final model
parameters = {
    'reg__n_estimators': 600,
    'reg__max_depth': 4,
    'reg__learning_rate': 0.33,
    'reg__colsample_bytree': 0.9,
    'reg__subsample': 0.95,
    'reg__gamma': 0.84,
    'reg__reg_alpha': 27,
    'reg__min_child_weight': 10.4
}

t0 = time()
pipeline.set_params(**parameters).fit(df_train[X], df_train[y])
print("Done in %0.3fs \n" % (time() - t0))

# Return predictions from final model
prediction = pipeline.predict(df_train[X])

# Plot variable importance
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(8, 13))
#xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

xgb.plot_importance(pipeline, max_num_features=20, height=0.5, ax=ax)

plot_importance(pipeline)
fig.show()


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(df_train[y], prediction))




# Create kaggle submission file

encode_cat_vars(df_test)
        
y_predict = pipeline.predict(df_test[X])
output = pd.DataFrame({'id': df_test['id'], 'price_doc': y_predict})
output.head()

output.to_csv('../output/xgb_submission.csv', index=False)