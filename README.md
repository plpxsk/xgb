# XGBoost with nested cross validation

In python, with `scikit-learn` and `xgboost`

# Usage

Install into virtual environment:

	pip install -r requirements.txt
	
Run script:

	python main.py
	
Example result:

```
(.venv) xgb# python main.py
0.9013119953863898 ROC AUC
0.895233774919653 PR AUC
```

These are estimates of generalization performance from the outer
cross-validation loop via sklearn's `cross_validate()`.

A separate inner loop of cross-validation is done with sklearn's `GridSearchCV()`

Predict on new data:

	gcv.best_estimator_.predict(X_new)

# Learn more

See blog posts:

  * [Simple example of XGBoost with nested cross validation, in python](https://plpxsk.github.io/2024/08/06/xgboost-nested-cv.html) (2024)
  * [What is nested cross-validation (for) and why you should use it](https://plpxsk.github.io/2017/12/02/nested-cross-validation.html) (2017)
