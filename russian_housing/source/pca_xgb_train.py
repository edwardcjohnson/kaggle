import numpy as np
import pandas as pd
from time import time
from sklearn.pipeline import Pipeline
from sklearn import model_selection, preprocessing

from sklearn.decomposition import PCA
from xgboost.sklearn import XGBRegressor
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV

processed_dir = '../data/processed/'
df_train = pd.read_pickle(processed_dir + 'macro_train.pkl')
df_test = pd.read_pickle(processed_dir + 'macro_test.pkl')

y = 'price_doc'
exclusions = ["id", "timestamp", "price_doc"]
X = [x for x in df_train.columns if x not in exclusions]


def encode_cat_vars(df):
    for c in X:
        if df[c].dtype == 'object':
            label = preprocessing.LabelEncoder()
            label.fit(list(df[c].values))
            df[c] = label.transform(list(df[c].values))

df_test = df_test.dropna(thresh=len(df_test), axis=1)

encode_cat_vars(df_train)

encode_cat_vars(pd.concat([df_train, df_test]))

# -----------------------------------------------------
#
#  XGBoost Randomized Grid Search
#
# -----------------------------------------------------



pipeline = Pipeline([
    ("imputer", preprocessing.Imputer(strategy="median", axis=0)),
    ('pca', PCA(n_components = 50)),#'mle', svd_solver = 'full')),
    ('reg', XGBRegressor(nthread = 1))#, objective = 'reg:linear'))
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

random_search = RandomizedSearchCV(pipeline, parameters, n_jobs=2, verbose=1)

t0 = time()
#random_search.fit(df_train.loc[0:1000,X], df_train.loc[0:1000,y])
random_search.fit(df_train[X], df_train[y])

print("done in %0.3fs \n" % (time() - t0))
print("Best score: %0.3f" % random_search.best_score_) # Best score: 0.29

print("Best parameters set:")
best_parameters = random_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))





# Train final model
parameters = {
    'reg__n_estimators': 746,
    'reg__max_depth': 10,
    'reg__learning_rate': 0.06,
    'reg__colsample_bytree': 0.89,
    'reg__subsample': 0.84,
    'reg__gamma': 5.3,
    'reg__reg_alpha': 27.3,
    'reg__min_child_weight': 117.9
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
print(rms)




# Create kaggle submission file
X = [x for x in df_test.columns if x not in exclusions]
encode_cat_vars(df_test)
        
y_predict = pipeline.predict(df_test[X])
output = pd.DataFrame({'id': df_test['id'], 'price_doc': y_predict})
output.head()

output.to_csv('../output/xgb_submission.csv', index=False)