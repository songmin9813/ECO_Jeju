# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 22:20:00 2021

@author: USER
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder ## 라벨인코딩을 위해 사용
from sklearn.preprocessing import RobustScaler # 단순선형회귀분석을 위해 사용
from sklearn.ensemble import RandomForestRegressor ## 랜덤포레스트 회귀모델
from xgboost import XGBRegressor ## XGBoost 회귀모델
from catboost import CatBoostRegressor ## CatBoost 회귀모델
import lightgbm as LGB ## LGBM 모델
from sklearn.metrics import mean_squared_error ## 평가지표로 사용할 MSE
from sklearn.impute import SimpleImputer ## 결측값 처리 패키지
import statsmodels.api as sm ## 단순선형회귀분석에 사용
from sklearn.model_selection import GridSearchCV
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")

path = "D:/git_project/ECO_Jeju/Sungmin/new_datas/final_edition" # waste, korean, long, short, res, card 경로
path2 = "D:/git_project/ECO_Jeju/Sungmin/new_datas/external" # rfid 경로
path3 = "D:/빅콘테스트/데이터/02_평가데이터_update(210806)"


waste = pd.read_csv(path + "/waste_group.csv")
korean = pd.read_csv(path + "/korean_group.csv")
long = pd.read_csv(path + "/long_group.csv")
short = pd.read_csv(path + "/short_group.csv")
res = pd.read_csv(path + "/resident_group.csv")
card = pd.read_csv(path + "/card_group.csv")
rfid = pd.read_csv(path + "/rfid_group.csv")

## 라벨인코더는 범주형 변수를 명목형 변수(수치형)으로 변환 할때 사용
le = LabelEncoder()
rs = RobustScaler()
#%% waste 전처리
waste
waste.info()
new_waste = waste

sample_index = waste["emd_nm"].unique()

emd_nm = new_waste["emd_nm"]
## le.fit을 이용해 명목형변수로 변경
le.fit(waste["emd_nm"])
emd_nm = le.transform(waste["emd_nm"])
new_waste["emd_nm"] = emd_nm
new_waste = new_waste.set_index(["base_date", "emd_nm"])





#%% korean 전처리

korean.drop(korean.loc[korean["emd_nm"] == "추자면"].index, inplace = True)
korean.drop(korean.loc[korean["emd_nm"] == "우도면"].index, inplace = True)
korean.info()

new_korean = korean

korean["emd_nm"].unique()

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

long.drop(long.loc[long["emd_nm"] == "추자면"].index, inplace = True)
long.drop(long.loc[long["emd_nm"] == "우도면"].index, inplace = True)


new_long = long

new_long.columns
long["emd_nm"].value_counts()  ## 추자면, 우도면 평가데이터에 없음

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
short.drop(short.loc[short["emd_nm"] == "추자면"].index, inplace = True)
short.drop(short.loc[short["emd_nm"] == "우도면"].index, inplace = True)

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

#%% resdient 

res
res.info()
res.isna().sum()
res.drop(res.loc[res["emd_nm"] == "추자면"].index, inplace = True)
res.drop(res.loc[res["emd_nm"] == "우도면"].index, inplace = True)

new_res = res

le.fit(res["emd_nm"])
emd_nm = le.transform(res["emd_nm"])
new_res["emd_nm"] = emd_nm
res.columns

new_res = new_res.set_index(["base_date", "emd_nm"])


#%% card 전처리

card
card.info()
card.isna().sum()

card["emd_nm"].value_counts()
card.drop(card.loc[card["emd_nm"] == "추자면"].index, inplace = True)
card.drop(card.loc[card["emd_nm"] == "우도면"].index, inplace = True)

new_card = card

le.fit(card["emd_nm"])
emd_nm = le.transform(card["emd_nm"])
new_card["emd_nm"] = emd_nm

new_card = new_card.set_index(["base_date", "emd_nm"])


#%% rifd 전처리

rfid.info()
rfid.isna().sum()

new_rfid = rfid

le.fit(rfid["emd_nm"])
emd_nm = le.transform(rfid["emd_nm"])
new_rfid["emd_nm"] = emd_nm

new_rfid = new_rfid.set_index(["base_date", "emd_nm"])

#%% 데이터 합치기, train test 셋 분리
data = pd.concat([new_waste, new_korean, new_long, new_short, new_res, new_card, new_rfid], axis = 1)
col = data.columns
data_index = data.index


# 7월 평균 데이터 생성
sum_seven = data.loc['2018-07']+data.loc['2019-07']+data.loc['2020-07']
avg_seven = sum_seven/3
# 8월 평균데이터 생성
sum_eight = data.loc['2018-08']+data.loc['2018-08']+data.loc['2018-08']
avg_eight = sum_eight/3

# 멀티인덱스 생성
index_2 = [['2021-07','2021-08'],['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21',
                                  '22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41']]


multi_index_2 = pd.MultiIndex.from_product(index_2)
new_index = data_index.append(multi_index_2)


new_data = pd.concat([data, avg_seven, avg_eight])

## 결측값을 각 피처들의 평균값으로 대체하기 위해 SimpleImputer 사용
imputer = SimpleImputer(strategy = 'mean')
imp_data = pd.DataFrame(imputer.fit_transform(new_data))
imp_data.columns = col
imp_data = imp_data.set_index(new_index)



x = imp_data.drop(["waste_em_g", "long_visit_pop_cnt", "korean_resd_pop_cnt", "korean_visit_pop_cnt","korean_work_pop_cnt","disCount","long_resd_pop_cnt","waste_pay_amt","long_work_pop_cnt","mct_cat_nm_6","mct_cat_nm_7","mct_cat_nm_4","mct_cat_nm_1","mct_cat_nm_0","mct_cat_nm_8","mct_cat_nm_9","use_cnt"], axis=1)
x.columns

# 변수삭제 반복 후 상관계수 비교
x_corr = x.corr()
y = np.array(imp_data["waste_em_g"])
# 모델링을 하기 위해 train과 test셋을 2018-01부터 2021-04까지로 나눈다 704 
train_x = x.iloc[0:1764]
test_x = x.iloc[1764:]
train_y = y[0:1764]
test_y = y[1764:]


#%% 단순선형회귀분석  
# 단순선형귀분석에 이용하기 위해 robust scaling을 진행
train_rs_x = rs.fit_transform(train_x)
model1 = sm.OLS(train_y, train_rs_x) 
fitted_model = model1.fit()
fitted_model.summary() 
y_pred = fitted_model.predict(test_x)
scores = mean_squared_error(test_y, y_pred)**0.5


#%% VIF 확인
features = "waste_em_cnt+short_visit_pop_cnt+resident_resid_reg_pop+resident_foreign_pop+resident_total_pop+mct_cat_nm_2+mct_cat_nm_3+mct_cat_nm_5+mct_cat_nm_10+disQuantity"
w, z = dmatrices("waste_em_g ~" + features, data=imp_data, return_type = "dataframe")
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(z.values, i) for i in range(z.shape[1])]
vif["features"] = z.columns

vif
#%% 모델링
models = []
models.append(['Random Forest', RandomForestRegressor()])
models.append(['XGBoost', XGBRegressor()])
models.append(['CatBoostRegressor', CatBoostRegressor(logging_level=("Silent"))])
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


## Lightgbm 모델의 rmse가 가장 낮음
df

#%% LGBM 하이퍼 파라미터 튜닝
gridParams = { 
    'learning_rate': [0.005],
    'n_estimators': [40],
    'num_leaves': [16,32, 64], 
    'random_state' : [501],
    'num_boost_round' : [3000],
    'colsample_bytree' : [0.65, 0.66],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1,1.2],
    'reg_lambda' : [1,1.2,1.4], 
    }

lgbm  = LGB.LGBMRegressor(n_estimators=100)

gridcv = GridSearchCV(lgbm, param_grid = gridParams, cv = 3)
gridcv.fit(train_x, train_y, eval_metric = 'mse')


print('Optimized hyperparameters', gridcv.best_params_)



#%% LGBM 4663982.861972165
model = LGB.LGBMRegressor(colsample_bytree = 0.65,
                           learning_rate = 0.005,
                           n_estimators = 40,
                           num_iterations=3000,
                           feature_fraction=0.7,
                           num_leaves = 32,
                           random_state = 501,
                           reg_alpha = 1,
                           reg_lambda = 1, 
                           subsample = 0.7)

model.fit(train_x, train_y)
y_pred = model.predict(test_x)

scores = mean_squared_error(test_y, y_pred)**0.5

y_pred
scores


#%% CatBoost Regressor 2947498.404632697
model2 = CatBoostRegressor(logging_level = ("Silent"))

model2.fit(train_x, train_y)
y_pred2 = model2.predict(test_x)

scores2 = mean_squared_error(test_y, y_pred2) ** 0.5

y_pred2
scores2


#%% 랜덤포레스트 하이퍼파라미터 튜닝

gridParams2 = {'n_estimators' : [10,100],
          'max_depth' : [6,8,10,12],
          'min_samples_leaf' : [8, 12, 18],
          'min_samples_split' : [8, 16, 20]
          }
rf  = RandomForestRegressor(n_estimators=100)

gridcv = GridSearchCV(rf, param_grid = gridParams2, cv = 3)
gridcv.fit(train_x, train_y)


print('Optimized hyperparameters', gridcv.best_params_) 

#%% 랜덤포레스트 모델선정 5437888.144746658
model3 = RandomForestRegressor(max_depth = 10,
                           min_samples_leaf = 8,
                           n_estimators = 100,
                           min_samples_split = 16,
                           )

model3.fit(train_x, train_y)
y_pred3 = model3.predict(test_x)

scores3 = mean_squared_error(test_y, y_pred3)**0.5

y_pred3
scores3


#%% Xgboost regressor 하이퍼 파라미터튜닝
gridParams3 = {'n_estimators' : [40,400,4000],
          'max_depth' : [6,8,10,12],
          'colsample_bytree' : [0.2, 0.9, 0.1],
          'eta' : [0.01, 0.05, 0.1, 0.5]
          }
reg  = XGBRegressor()

gridcv = GridSearchCV(reg, param_grid = gridParams3, cv = 3)
gridcv.fit(train_x, train_y, eval_metric = "rmse")


print('Optimized hyperparameters', gridcv.best_params_) 

#%% Xgb regressor 모델 선정

model4 = XGBRegressor(n_estimators = 4000,
                      max_depth = 6,
                      colsample_bytree = 0.9,
                      eta = 0.1,)

model4.fit(train_x, train_y)
y_pred4 = model4.predict(test_x)

scores4 = mean_squared_error(test_y, y_pred4) ** 0.5

y_pred4
scores4
#%% sample submission #csv로 작성 후 엑셀에서 붙여넣기로 진행

seven = np.around(y_pred2[0:42],1)
eight = np.around(y_pred2[42:],1)

DF = pd.DataFrame({'행정동명': sample_index, '7월 배출량(g)': seven, '8월 배출량(g)': eight})

submission = pd.read_csv(path3 + "/sample_submission2.csv")
submission.set_index(submission["NO"], inplace = True)
submission.drop(["NO"], axis = 1, inplace=True)

submission = pd.merge(submission, DF, on = '행정동명', how = 'inner')
submission.drop(["7월 배출량(g)_x", "8월 배출량(g)_x"], axis = 1, inplace = True)
submission.columns = ['행정동명', '7월 배출량(g)', '8월 배출량(g)']
submission.to_csv('D:/빅콘테스트/데이터/02_평가데이터_update(210806)/result.csv')
