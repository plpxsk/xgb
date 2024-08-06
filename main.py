from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import make_pipeline

from xgboost import XGBClassifier

# train and evaluate with nested cross-validation
inner_cv = 5
outer_cv = 3
inner_metric = 'roc_auc'
outer_metrics = ['roc_auc', 'average_precision']

random_state = 99
n_jobs_1 = 2
n_jobs_2 = 3


X, y = make_classification(random_state=random_state)

params = dict(
    xgbclassifier__n_estimators=[100, 500],
    xgbclassifier__max_depth=[1, 3, 5]
)

clf = make_pipeline(
    StandardScaler(),
    XGBClassifier(random_state=random_state, n_jobs=n_jobs_1)
)

# inner cv
gcv = GridSearchCV(clf, params, scoring=inner_metric, cv=inner_cv, n_jobs=1,
                   return_train_score=False)

# outer cv
results = cross_validate(gcv, X, y, scoring=outer_metrics, cv=outer_cv,
                         n_jobs=n_jobs_2, return_train_score=False)

# show estimates of generalization performance of our classifier
print(results['test_roc_auc'].mean(), "ROC AUC")
print(results['test_average_precision'].mean(), "PR AUC")

# final model that can be used in production
# gcv.fit(X, y)
# gcv.best_params_

# predict in production
# gcv.predict(X_new)
# or more explicitly
# gcv.best_estimator_.predict(X_new)
