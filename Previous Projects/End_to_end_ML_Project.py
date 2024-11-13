import pandas as pd
import numpy as np
import os
import tarfile
from six.moves import urllib
import matplotlib.pyplot as plt

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_path=HOUSING_PATH, housing_url=HOUSING_URL):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housihg.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path = HOUSING_PATH):
    housing_csv = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(housing_csv)
housing = load_housing_data()

%matplotlib inline
housing.hist(bins=50, figsize=(20,15))

housing = load_housing_data()

#Creating a test set, putting it aside and never look at it


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


#The median income is a very important feature, so I want to make sure that the test dataset is a represantative of it

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels=[1,2,3,4,5])

#Now let's do Stratified Sampling based on Income category
housing["income_cat"].hist()

from sklearn.model_selection import StratifiedShuffleSplit

splits = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)


for train_index, test_index in splits.split(housing, housing["income_cat"]):
    stra_train_set = housing.loc[train_index]
    stra_test_set = housing.loc[test_index]
    

(stra_train_set["income_cat"].value_counts()/len(stra_train_set)).sort_index()

#comparsion

def income_cat_props(data):
    return data["income_cat"].value_counts()/len(data["income_cat"])

compare_props = pd.DataFrame({"overall": income_cat_props(housing),
                              "Stratified": income_cat_props(stra_train_set),
                              "Random": income_cat_props(train_set)}).sort_index()


compare_props["%Rand_error"] = 100 * compare_props["Random"] / compare_props["overall"] - 100
compare_props["%Strat_error"] = 100 * compare_props["Stratified"] / compare_props["overall"] - 100

for set_ in (stra_train_set, train_set):
    set_.drop("income_cat", axis=1, inplace=True)
    
    
housing = stra_train_set.copy()

#Visualizing the data
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="Population", 
             c="median_house_value", cmap = plt.get_cmap("jet"), colorbar=True)


corr_matrix = housing.drop("ocean_proximity", axis=1).corr()

corr_matrix["median_house_value"].sort_values(ascending=True)

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

#Attributre Combination

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.drop("ocean_proximity", axis=1).corr()

corr_matrix["median_house_value"].sort_values(ascending=False)

housing = stra_train_set.drop("median_house_value", axis=1)
housing_labels = stra_train_set["median_house_value"]

#Transforming the Numerical Attributes

from sklearn.impute import SimpleImputer

impute = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)
housing_cat = housing[["ocean_proximity"]].copy()
impute.fit(housing_num)
impute.statistics_

X = impute.transform(housing_num)

#Handling Text and Categorical Data

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

#Custome Transformers

from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
        

    
#Transformation Pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("attribs_adder", CombinedAttributesAdder()),
                         ("std_scaler", StandardScaler())])

#housing_num_tr = num_pipeline.fit_transform(housing_num)

#Column Transformer

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipelie = ColumnTransformer([("num", num_pipeline, num_attribs),
                                  ("cat", OneHotEncoder(), cat_attribs)])


housing_prepared = full_pipelie.fit_transform(housing)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#Let's try it out on some of the training set

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipelie.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

#Measuring RMSE on the whole train set

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_rse = mean_squared_error(housing_predictions, housing_labels)
lin_rmse = np.sqrt(lin_rse)
print(lin_rmse)

#let's train a more powerful model to avoid underfitting. Let's choose Decision Trees Regressor

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

#Let's evaluate it on the training set

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_predictions, housing_labels)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, cv=10,
                         scoring="neg_mean_squared_error")

tree_rmse = np.sqrt(-scores)

def display_scores(scores):
    print("Scores", scores)
    print("Mean", scores.mean())
    print("Standard Deviation", scores.std())
    
    
#doing the same with Linear Regression

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, cv=10, scoring="neg_mean_squared_error")

lin_scores_rmse = np.sqrt(-lin_scores)

display_scores(lin_scores_rmse)


##RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, cv=10, scoring="neg_mean_squared_error")
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

#fune_tune your model
from sklearn.model_selection import GridSearchCV

param_grid = [{"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
              {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2,3,4]}]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
grid_search.best_estimator_
grid_search.best_params_
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
pd.DataFrame(cvres)


from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint

params_distribs = {"n_estimators": randint(low=1, high=200),
                   "max_features": randint(low=1, high=8)}


foest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=params_distribs, cv=10, n_iter=10, scoring="neg_mean_squared_error", random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

rnd_search_results = rnd_search.cv_results_

for mean_score, params in zip(rnd_search_results["mean_test_score"], rnd_search_results["params"]):
    print(np.sqrt(-mean_score), params)
    
    
final_model = grid_search.best_estimator_

x_test = stra_train_set.drop("median_house_value", axis=1)
y_test = stra_train_set["median_house_value"].copy()

x_test_prepared = full_pipelie.fit_transform(x_test)
final_predictions = final_model.predict(x_test_prepared)

final_mse = mean_squared_error(final_predictions, y_test)
final_rmse = np.sqrt(final_mse)
final_rmse