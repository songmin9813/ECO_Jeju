# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 22:20:00 2021

@author: USER
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as LGB
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso,ElasticNet,Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


path = "D:/git_project/ECO_Jeju/Sungmin/new_datas/2nd_edition"

# waste = pd.read_csv(path + "/01_음식물쓰레기_FOOD_WASTE_210811_update.CSV", encoding = "CP949", low_memory=False)
# korean = pd.read_csv(path + "/02-1_내국인유동인구_KOREAN.CSV", encoding = "CP949")
# long = pd.read_csv(path + "/02-2_장기체류 외국인 유동인구_LONG_TERM_FRGN.CSV", encoding = "CP949")
# short = pd.read_csv(path + "/02-3_단기체류 외국인 유동인구_SHORT_TERM_FRGN.CSV", encoding = "CP949")
# res = pd.read_csv(path + "/03_거주인구_RESIDENT_POP.CSV", encoding = "CP949")
# card = pd.read_csv(path + "/04_음식관련 카드소비_CARD_SPENDING.CSV", encoding = "CP949")

waste = pd.read_csv(path + "/waste_group.csv")
korean = pd.read_csv(path + "/korean_group.csv")
long = pd.read_csv(path + "/long_group.csv")
short = pd.read_csv(path + "/short_group.csv")
res = pd.read_csv(path + "/resident_group.csv")
card = pd.read_csv(path + "/card_group.csv")


le = LabelEncoder()
sc = StandardScaler()
#%% waste 전처리
waste
waste.info()
new_waste = waste


enm = waste["emd_nm"].value_counts()


temp=list(waste.groupby(['emd_nm']))
new_list=[]
for value in temp:
    new_list.append(value[0])
new_list.sort()
dict_={}
for i,value in enumerate(new_list):
    dict_[value]=i

le.fit(waste["emd_nm"])
emd_nm = le.transform(waste["emd_nm"])
new_waste["emd_nm"] = emd_nm


new_waste = new_waste.set_index(["base_date", "emd_nm"])



x = new_waste.drop("waste_em_g", axis=1)
y = np.array(new_waste["waste_em_g"])

sc_x = sc.fit_transform(x)
sc_y = sc.fit_transform(y.reshape(-1,1))


train_x, test_x, train_y, test_y = train_test_split(sc_x, sc_y, test_size = 0.3, random_state = 100)

models = []
models.append(['Ridge', Ridge()])
models.append(['Lasso', Lasso()])
models.append(['ElasticNet', ElasticNet()])
# models.append(['SVR', SVR()])
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


df

#%% korean 전처리

korean
korean.info()

new_korean = korean


temp=list(korean.groupby(['emd_nm']))
new_list=[]
for value in temp:
    new_list.append(value[0])
new_list.sort()
dict_={}
for i,value in enumerate(new_list):
    dict_[value]=i


le.fit(korean["emd_nm"])
emd_nm = le.transform(korean["emd_nm"])
new_korean["emd_nm"] = emd_nm

new_korean = new_korean.set_index(["base_date", "emd_nm"])
new_korean.columns
new_korean = new_korean.drop(['korean_sex_남성', 'korean_sex_여성', 'korean_age_0', 'korean_age_10',
       'korean_age_20', 'korean_age_30', 'korean_age_40', 'korean_age_50',
       'korean_age_60', 'korean_age_70', 'korean_age_80'], axis = 1)


#%% concat waste korean

waste_korean = pd.concat([new_waste, new_korean], axis = 1)
col = waste_korean.columns



imputer = SimpleImputer(strategy = 'mean')
new_dfbase = pd.DataFrame(imputer.fit_transform(waste_korean))
new_dfbase.columns = col


df_corr = new_dfbase.corr()


x = new_dfbase.drop("waste_em_g", axis=1)
y = np.array(new_dfbase["waste_em_g"])

sc = StandardScaler()

sc_x = sc.fit_transform(x)
sc_y = sc.fit_transform(y.reshape(-1,1))


train_x, test_x, train_y, test_y = train_test_split(sc_x, sc_y, test_size = 0.3, random_state = 100)

#%% 모델학습

models = []
models.append(['Ridge', Ridge()])
models.append(['Lasso', Lasso()])
models.append(['ElasticNet', ElasticNet()])
# models.append(['SVR', SVR()])
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


df
y_pred


#%% long 전처리

long
long.info()

long.isna().sum()
new_long = long

le.fit(long["nationality"])
nationality = le.transform(long["nationality"])
new_long["nationality"] = nationality


new_na = new_long["nationality"]

le.fit(long["city"])
city = le.transform(long["city"])
new_long["long"] = city

le.fit(long["emd_nm"])
emd_nm = le.transform(long["emd_nm"])
new_long["emd_nm"] = emd_nm


lcd = long["emd_cd"].value_counts()
lnm = long["emd_nm"].value_counts()


#%% short 전처리

short
short.info()
short.isna().sum()

scd = short["emd_cd"].value_counts()
snm = short["emd_nm"].value_counts()



#%% resdient 전처리

res
res.info()
res.isna().sum()

rcd = res["emd_cd"].value_counts()
rnm = res["emd_nm"].value_counts()


#%% card 전처리

card
card.info()
card.isna().sum()

ccd = card["emd_cd"].value_counts()
cnm = card["emd_nm"].value_counts()


cmct_ccd = card["mct_cat_cd"].value_counts()
cmct_nm = card["mct_cat_nm"].value_counts()
