library(ggplot2)
library(tidyverse)
library(data.table)
setwd("D:/git_project/ECO_Jeju/Sungmin/new_datas/3rd_edition")
waste <- read.csv("waste_group1.csv", encoding = 'CP949')
  

# 평균버리는 쓰레기량
emg_g_day_sum <- waste %>% group_by(emd_nm) %>% summarise(waste_em_g = sum(as.numeric(waste_em_g)/as.numeric(waste_em_cnt)))
windows()
ggplot(emg_g_day_sum, aes(x=reorder(emd_nm,waste_em_g), y=waste_em_g)) + geom_bar(stat = "identity") +
  theme(axis.text.x=element_text(angle = 45)) +
  theme(axis.text.x=element_text(vjust=0.5)) +
  labs(title = "읍면동별 평균 쓰레기배출량") +
  labs(x='읍면동', y='평균쓰레기배출량') 


emg_g_sum <- waste %>% group_by(emd_nm) %>% summarise(waste_em_g = sum(as.numeric(waste_em_g)))
ggplot(emg_g_sum, aes(x=reorder(emd_nm,waste_em_g), y=waste_em_g)) + geom_bar(stat = "identity") +
  theme(axis.text.x=element_text(angle = 45)) +
  theme(axis.text.x=element_text(vjust=0.5)) +
  labs(title = "읍면동별 총 쓰레기배출량") +
  labs(x='읍면동', y='총쓰레기배출량') 



windows()
emg_g_mean <- waste %>% group_by(emd_nm) %>% summarise(waste_em_g = mean(as.numeric(waste_em_g)))
ggplot(emg_g_mean, aes(x=reorder(emd_nm,waste_em_g), y=waste_em_g)) + geom_bar(stat = "identity") +
  theme(axis.text.x=element_text(angle = 45)) +
  theme(axis.text.x=element_text(vjust=0.5)) +
  labs(title = "읍면동별 쓰레기 총량") +
  labs(x='읍면동', y='총합쓰레기량') 


  



# 어느 월에 쓰레기가 가장 많이 나왔는지도 봐야함
emg_g_month <- waste %>% group_by(base_date, emd_nm) %>% summarise(sum_em_g = sum(as.numeric(waste_em_g)))
emg_g_month

windows()
ggplot(emg_g_month, aes(x=emd_nm, y=sum_em_g, group=emd_nm)) +
  geom_line(aes(color = emd_nm)) +
  facet_wrap(~base_date)



# 년도별 쓰레기량
windows()
ggplot(waste, aes(x=base_date, y=waste_em_g, group=emd_nm)) + 
  geom_line(aes(color=emd_nm)) +
  theme_bw() +
  theme(axis.text.x=element_text(angle = 45)) +
  theme(axis.text.x=element_text(hjust=1.2)) +
  labs(x='년월', y='읍면동별 쓰레기량')




library(ggmap)

names <- c("건입동","남원읍","노형동",
           "대륜동","대정읍","대천동",
           "도두동","동홍동","봉개동","삼도1동",
           "삼도2동","삼양동","서홍동","성산읍","송산동",
           "아라동","안덕면","애월읍",
           "연동","영천동","예래동","오라동","외도동",
           "용담1동","용담2동","이도1동","이도2동","이호동",
           "일도1동","일도2동","정방동","중문동","중앙동",
           "천지동","표선면","화북동","효돈동","구좌읍",
           "조천읍","한경면","한림읍")
addr <- c("제주시 건입동","서귀포시 남원읍","제주시 노형동",
          "서귀포시 대륜동","서귀포시 대정읍","서귀포시 대천동", "제주시 도두동", "서귀포시 동홍동", "제주시 봉개동",
          "제주시 삼도1동","제주시 삼도2동","제주시 삼양동",
          "서귀포시 서홍동","서귀포시 성산읍","서귀포시 송산동", "제주시 아라동", "서귀포시 안덕면","제주시 애월읍","제주시 연동", "서귀포시 영천동", "서귀포시 예래동", "제주시 오라동", "제주시 외도동","제주시 용담1동","제주시 용담2동", "제주시 이도1동", "제주시 이도2동", "제주시 이호동", "제주시 일도1동", "제주시 일도2동", "서귀포시 정방동","서귀포시 중문동", "서귀포시 중앙동", "서귀포시 천지동", "서귀포시 표선면", "제주시 화북동", "서귀포시 효돈동","제주시 구좌읍","제주시 조천읍","제주시 한경면", "제주시 한림읍")
          

gc <- geocode(enc2utf8(addr))

gc
# A tibble: 6 x 2
#     lon   lat
#   <dbl> <dbl>
# 1  127.  33.5
# 2  127.  33.5
# 3  127.  33.3
# 4  126.  33.3
# 5  126.  33.4
# 6  126.  33.3


df <- data.frame(name=names,
                 lon=gc$lon,
                 lat=gc$lat)
cen <- c(mean(df$lon),mean(df$lat))
map <- get_googlemap(center=cen,
                     maptype="roadmap",
                     zoom=10,
                     size=c(640,640),
                     marker=gc)
ggmap(map) + geom_text(gc, mapping=aes(x=as.numeric(lon), y=as.numeric(lat)+0.06),label=names, size=5)

