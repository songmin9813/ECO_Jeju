# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 22:20:00 2021

@author: USER
"""

import numpy as np
import pandas as pd
import datetime

path = "D:/project"

waste = pd.read_csv(path + "/01_음식물쓰레기_FOOD_WASTE_210811_update.CSV", encoding = "CP949")
korean = pd.read_csv(path + "/02-1_내국인유동인구_KOREAN.CSV", encoding = "CP949")
long = pd.read_csv(path + "/02-2_장기체류 외국인 유동인구_LONG_TERM_FRGN.CSV", encoding = "CP949")
short = pd.read_csv(path + "/02-3_단기체류 외국인 유동인구_SHORT_TERM_FRGN.CSV", encoding = "CP949")
res = pd.read_csv(path + "/03_거주인구_RESIDENT_POP.CSV", encoding = "CP949")
card = pd.read_csv(path + "/04_음식관련 카드소비_CARD_SPENDING.CSV", encoding = "CP949")


#%% waste 전처리
waste
waste.info()

ecd = waste["emd_cd"].value_counts()
enm = waste["emd_nm"].value_counts()


#%% korean 전처리

korean
korean.info()

kcd = korean["emd_cd"].value_counts()
knm = korean["emd_nm"].value_counts()


#%% long 전처리

long
long.info()

lcd = long["emd_cd"].value_counts()
lnm = long["emd_nm"].value_counts()


#%% short 전처리

short
short.info()

scd = short["emd_cd"].value_counts()
snm = short["emd_nm"].value_counts()



#%% resdient 전처리

res
res.info()

rcd = res["emd_cd"].value_counts()
rnm = res["emd_nm"].value_counts()


#%% card 전처리

card
card.info()

ccd = card["emd_cd"].value_counts()
cnm = card["emd_nm"].value_counts()


cmct_ccd = card["mct_cat_cd"].value_counts()
cmct_nm = card["mct_cat_nm"].value_counts()
