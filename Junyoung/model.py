# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 20:58:49 2021

@author: USER
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as LGB
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso,ElasticNet,Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

#%% 데이터 나누기
path = "D:/project"

dfbase = pd.read_csv(path + "/new_base.csv")

dfbase
dfbase.info()
dfbase.isna().sum()


new_base = dfbase.set_index(["base_date", "emd_nm"])


imputer = SimpleImputer(strategy = 'mean')
new_dfbase = pd.DataFrame(imputer.fit_transform(new_base))


temp = list(new_base.columns)

new_dfbase.columns = temp


sc = StandardScaler()



x = new_dfbase.drop("waste_em_g", axis=1)
y = new_dfbase["waste_em_g"]

sc = StandardScaler()

sc_x = sc.fit_transform(x)
sc_y = sc.fit_transform(y)


train_x, test_x, train_y, test_y = train_test_split(sc_x, sc_y, test_size = 0.3, random_state = 100)

#%% 모델학습

models = []
models.append(['Ridge', Ridge()])
models.append(['Lasso', Lasso()])
models.append(['ElasticNet', ElasticNet()])
models.append(['SVR', SVR()])
models.append(['Random Forest', RandomForestRegressor()])
models.append(['XGBoost', XGBRegressor()])
models.append(['LinearRegression', LinearRegression()])
models.append(['CatBoostRegressor', CatBoostRegressor(logging_level=("Silent"))])
models.append(['PLSRegression', PLSRegression()])
models.append(['Lightgbm', LGB.LGBMRegressor()])

list_1 = []

for m in range(len(models)):
    print(models[m])
    model = models[m][1]
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    scores = mean_squared_error(test_y, y_pred)**0.5
    list_1.append(scores)


df_1 = pd.DataFrame(models)    
df = pd.DataFrame(list_1)
df.index = df_1.iloc[:,0]



df['mean'] = df.mean(axis=1) 
