# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 22:20:00 2021

@author: USER
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder ## 라벨인코딩을 위해 사용
from sklearn.preprocessing import StandardScaler ## StandardScaler를 위해 사용
from sklearn.model_selection import train_test_split ## train_test셋을 나누기 위해 사용
from sklearn.ensemble import RandomForestRegressor ## 랜덤포레스트 회귀모델
from sklearn.linear_model import LinearRegression ## 선형 회귀모델
from xgboost import XGBRegressor ## XGBoost 회귀모델
from catboost import CatBoostRegressor ## CatBoost 회귀모델
import lightgbm as LGB ## LGBM 모델
from sklearn.cross_decomposition import PLSRegression ## PLS 회귀모델
from sklearn.linear_model import Lasso,ElasticNet,Ridge ## Lasso, Ridge, ElasticNet 회귀모델
from sklearn.svm import SVR ## SVM 회귀모델
from sklearn.metrics import mean_squared_error ## 평가지표로 사용할 MSE
from sklearn.impute import SimpleImputer ## 결측값 처리 패키지
import statsmodels.api as sm ## 단순선형회귀분석에 사용


# path = "D:/git_project/ECO_Jeju/Sungmin/new_datas/2nd_edition"
path = "C:/Users/mitha/OneDrive/바탕 화면/dev/ECO_Jeju/Sungmin/new_datas/2nd_edition/"

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


## 라벨인코더는 범주형 변수를 명목형 변수(수치형)으로 변환 할때 사용
## 피처들의 단위가 모두 다르므로 StandardScaler를 사용 -> 이상치가 있나 없나 탐지해봐야함
le = LabelEncoder()
sc = StandardScaler()
#%% waste 전처리
waste
waste.info()
new_waste = waste


temp=list(waste.groupby(['emd_nm']))
new_list=[]
for value in temp:
    new_list.append(value[0])
new_list.sort()
dict_={}
for i,value in enumerate(new_list):
    dict_[value]=i


## le.fit을 이용해 명목형변수로 변경
le.fit(waste["emd_nm"])
emd_nm = le.transform(waste["emd_nm"])
new_waste["emd_nm"] = emd_nm
new_waste = new_waste.set_index(["base_date", "emd_nm"])





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


new_korean = new_korean.drop(['korean_sex_남성', 'korean_sex_여성', 'korean_age_0', 'korean_age_10',
       'korean_age_20', 'korean_age_30', 'korean_age_40', 'korean_age_50',
       'korean_age_60', 'korean_age_70', 'korean_age_80'], axis = 1)


new_korean = new_korean.set_index(["base_date", "emd_nm"])




#%% long 전처리

long
long.info()

long.isna().sum()
new_long = long

new_long.columns

new_long = new_long.drop(['long_nationality_AUS', 'long_nationality_BGD',
       'long_nationality_CAN', 'long_nationality_CHN', 'long_nationality_DEU',
       'long_nationality_EGY', 'long_nationality_ETC', 'long_nationality_FRA',
       'long_nationality_GBR', 'long_nationality_IDN', 'long_nationality_IND',
       'long_nationality_JPN', 'long_nationality_KAZ', 'long_nationality_KGZ',
       'long_nationality_KHM', 'long_nationality_LKA', 'long_nationality_MGL',
       'long_nationality_MMR', 'long_nationality_MYS', 'long_nationality_NGR',
       'long_nationality_NPL', 'long_nationality_NZL', 'long_nationality_PAK',
       'long_nationality_PHL', 'long_nationality_RUS', 'long_nationality_THA',
       'long_nationality_TWN', 'long_nationality_UKR', 'long_nationality_USA',
       'long_nationality_UZB', 'long_nationality_VNM'], axis = 1)



le.fit(long["emd_nm"])
emd_nm = le.transform(long["emd_nm"])
new_long["emd_nm"] = emd_nm

new_long = new_long.set_index(["base_date", "emd_nm"])





#%% short 전처리

short
short.info()
short.columns

new_short = short
new_short = new_short.drop(['short_nationality_CHN', 'short_nationality_ETC',
       'short_nationality_HKG', 'short_nationality_IDN',
       'short_nationality_JPN', 'short_nationality_MYS',
       'short_nationality_SGP', 'short_nationality_THA',
       'short_nationality_USA', 'short_nationality_VNM'], axis = 1)


le.fit(short["emd_nm"])
emd_nm = le.transform(short["emd_nm"])
new_short["emd_nm"] = emd_nm

new_short = new_short.set_index(["base_date", "emd_nm"])

#%% resdient 전처리 년도와 월을합쳐서 하나의 피처로 만들어서 진행해야 할 꺼같다
# -> resdient 데이터를 사용할꺼면 다른 데이터들도 월단위로 바꿔야할 꺼 같다

res
res.info()
res.isna().sum()

new_res = res

le.fit(res["emd_nm"])
emd_nm = le.transform(res["emd_nm"])
new_res["emd_nm"] = emd_nm
res.columns




#%% card 전처리

card
card.info()
card.isna().sum()

new_card = card

le.fit(card["emd_nm"])
emd_nm = le.transform(card["emd_nm"])
new_card["emd_nm"] = emd_nm

new_card = new_card.set_index(["base_date", "emd_nm"])

#%% 데이터 합치기, train test 셋 분리
data = pd.concat([new_waste, new_korean, new_long, new_short, new_card], axis = 1)
col = data.columns
data_index = data.index


## 결측값을 각 피처들의 평균값으로 대체하기 위해 SimpleImputer 사용
imputer = SimpleImputer(strategy = 'mean')
imp_data = pd.DataFrame(imputer.fit_transform(data))
imp_data.columns = col
imp_data = imp_data.set_index(data_index)



## 이 부분에서 변수를 삭제해서 더 진행할 수 있다.
x = imp_data.drop(["waste_em_g", "korean_resd_pop_cnt"], axis=1)
x.columns
y = np.array(imp_data["waste_em_g"])

# StandardScaler를 통해 값을 정규화 시킨다
sc_x = sc.fit_transform(x)
sc_y = sc.fit_transform(y.reshape(-1,1))


# 모델링을 하기 위해 train과 test셋의 비율을 7:3으로 나눈다. 
train_x, test_x, train_y, test_y = train_test_split(sc_x, sc_y, test_size = 0.3, random_state = 100)

#%% 단순선형회귀분석  

model1 = sm.OLS(train_y, train_x) 
fitted_model = model1.fit()
fitted_model.summary() 
tempx=x.corr()
# 1.korean_resd_pop_cnt 변수의 p-value 값이 0.05보다 높아서 삭제
## 전부 0.05안에 들어 유의미한 변수이며 R결정계수값이 0.979로 높으편이다


#%% 모델링
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


df
