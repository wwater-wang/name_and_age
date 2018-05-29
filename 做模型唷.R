library(tidyverse)
library(stringr)
library(ROSE)
library(sigmoid)
library(xgboost)

abt <- read_csv("name_abt.csv")

#clean_data-------------
abt <- abt %>%
  mutate(birth_yr = str_sub(birthday, 6, 9) %>% as.integer(),
         name = str_replace_all(final_name, "[A-Za-z0-9?？﹌ㄨㄤㄧ　<@;/ -,.-]{1,}", ""),
         len = nchar(name)) %>%
  filter(len == 3,
         birth_yr >= 1940, birth_yr <= 1998) %>% #從分布圖上這段區間稍較合理
  filter(!str_detect(name, "先生|小姐")) %>%
  select(name, birth_yr)

#拆解姓名
abt <- abt %>%
  mutate(name = str_sub(name,2,3),
         name_1st = str_sub(name,1,1),
         name_2nd = str_sub(name,2,2))

#出生年分組 #先測試40歲的分界線
abt <- abt %>%
  mutate(birth_1980_below = ifelse(birth_yr < 1980, 1, 0))

abt$birth_1980_below %>% table() %>% prop.table()

#測試組
set.seed(203)
test.index <- sample(1:length(abt$name), length(abt$name) * 0.3)
abt.train <- abt[-test.index,]
abt.test <- abt[test.index,]

#Ann的規則---------------------
pred <- abt.train %>%
  group_by(name) %>%
  summarise(pred = mean(birth_1980_below)) %>%
  ungroup()

#start_predict
tmp <- left_join(abt.test, pred) %>%
  #filter(!is.na(pred)) %>%
  mutate(pred = ifelse(is.na(pred), runif(1), pred))

roc.curve(tmp$birth_1980_below, tmp$pred, plotit = T)
pred.dt2 = ifelse(tmp$pred>=0.5,1,0)
table(pred.dt2, tmp$birth_1980_below) %>% prop.table() ##confusion matrix
sum(pred.dt2==tmp$birth_1980_below)/nrow(tmp)## ACC
sum(pred.dt2==1 & tmp$birth_1980_below==1)/(sum(pred.dt2==1)) ## 1980+ precision
sum(pred.dt2==0 & tmp$birth_1980_below==0)/(sum(pred.dt2==0)) ## ~1980 precision
sum(pred.dt2==1 & tmp$birth_1980_below==1)/(sum(pred.dt2==1 & tmp$birth_1980_below==1) + sum(pred.dt2=="0" & tmp$birth_1980_below=="1")) ##Recall

#拆字新規則---------------------
a <- abt.train %>%
  group_by(name) %>%
  summarise(pred_tot = mean(birth_1980_below),
            cnt_tot = n()) %>%
  ungroup()

b <- abt.train %>%
  group_by(name_1st) %>%
  summarise(pred_1st = mean(birth_1980_below),
            cnt_1st = n()) %>%
  ungroup()

c <- abt.train %>%
  group_by(name_2nd) %>%
  summarise(pred_2nd = mean(birth_1980_below),
            cnt_2nd = n()) %>%
  ungroup()

#start_predict
tmp <- left_join(abt.test, a) %>%
  left_join(b) %>%
  left_join(c)

tmp[,c(8,10)] <- tmp[,c(8,10)] %>%
  sapply(function(x) ifelse(is.na(x), 0.5, x))

tmp[,c(7,9,11)] <- tmp[,c(7,9,11)] %>%
  sapply(function(x) ifelse(is.na(x), 0, x))

tmp <- tmp %>%
  mutate(pred = pred_tot,
         pred = ifelse(is.na(pred), sigmoid(pred_1st + pred_2nd - 1), pred),
         #pred = ifelse(is.na(pred), (pred_1st*cnt_1st + pred_2nd*cnt_2nd)/(cnt_1st+cnt_2nd), pred),
         pred = ifelse(pred < (sigmoid(pred_1st + pred_2nd - 1)), (sigmoid(pred_1st + pred_2nd - 1)), pred),
         #pred = ifelse(pred < (pred_1st*cnt_1st + pred_2nd*cnt_2nd)/(cnt_1st+cnt_2nd), (pred_1st*cnt_1st + pred_2nd*cnt_2nd)/(cnt_1st+cnt_2nd), pred),
         pred = ifelse(is.na(pred), runif(1), pred))


tmp
#mutate(pred = ifelse(is.na(pred), runif(1), pred))
roc.curve(tmp$birth_1980_below, tmp$pred, plotit = T)

pred.dt2 = ifelse(tmp$pred>=0.5,1,0)
table(pred.dt2, tmp$birth_1980_below) %>% prop.table() ##confusion matrix
sum(pred.dt2==tmp$birth_1980_below)/nrow(tmp)## ACC
sum(pred.dt2==1 & tmp$birth_1980_below==1)/(sum(pred.dt2==1)) ## 1980+ precision
sum(pred.dt2==0 & tmp$birth_1980_below==0)/(sum(pred.dt2==0)) ## ~1980 precision
sum(pred.dt2==1 & tmp$birth_1980_below==1)/(sum(pred.dt2==1 & tmp$birth_1980_below==1) + sum(pred.dt2=="0" & tmp$birth_1980_below=="1")) ##Recall

tmp %>%
  mutate(pred = ifelse(pred >= 0.5, 1, 0)) %>%
  filter(birth_1980_below != pred)

#拆字新規則，進模型---------------------
a <- abt.train %>%
  group_by(name) %>%
  summarise(pred_tot = mean(birth_1980_below),
            cnt_tot = n()) %>%
  ungroup()

b <- abt.train %>%
  group_by(name_1st) %>%
  summarise(pred_1st = mean(birth_1980_below),
            cnt_1st = n()) %>%
  ungroup()

c <- abt.train %>%
  group_by(name_2nd) %>%
  summarise(pred_2nd = mean(birth_1980_below),
            cnt_2nd = n()) %>%
  ungroup()

mdl <- left_join(abt.train, a) %>%
  left_join(b) %>%
  left_join(c)

mdl[,c(8,10)] <- mdl[,c(8,10)] %>%
  sapply(function(x) ifelse(is.na(x), 0.5, x))

mdl[,c(7,9,11)] <- mdl[,c(7,9,11)] %>%
  sapply(function(x) ifelse(is.na(x), 0, x))

mdl <- mdl %>%
  mutate(pred = pred_tot,
         pred = ifelse(is.na(pred), sigmoid(pred_1st + pred_2nd - 1), pred),
         pred = ifelse(pred < (sigmoid(pred_1st + pred_2nd - 1)), (sigmoid(pred_1st + pred_2nd - 1)), pred),
         pred = ifelse(is.na(pred), runif(1), pred))

mdl$birth_1980_below <- mdl$birth_1980_below %>% as.factor()
sparse_matrix <- Matrix::sparse.model.matrix(birth_1980_below ~ .-1, data =  mdl[, c(5:12)])


Y = as.integer(mdl$birth_1980_below) - 1
param = list("objective" = "binary:logistic",
             "eval_metric" = "auc",
             "num_class" = 1)
xg.mdl = xgboost(param=param, data=sparse_matrix, label=Y, nrounds=20) 

#start_predict
tmp <- left_join(abt.test, a) %>%
  left_join(b) %>%
  left_join(c)

tmp[,c(8,10)] <- tmp[,c(8,10)] %>%
  sapply(function(x) ifelse(is.na(x), 0.5, x))

tmp[,c(7,9,11)] <- tmp[,c(7,9,11)] %>%
  sapply(function(x) ifelse(is.na(x), 0, x))

tmp <- tmp %>%
  mutate(pred = pred_tot,
         pred = ifelse(is.na(pred), sigmoid(pred_1st + pred_2nd - 1), pred),
         pred = ifelse(pred < (sigmoid(pred_1st + pred_2nd - 1)), (sigmoid(pred_1st + pred_2nd - 1)), pred),
         pred = ifelse(is.na(pred), runif(1), pred),
         pred_tot = ifelse(is.na(pred_tot), 0.5, pred))


tmp
test.mtx <- Matrix::sparse.model.matrix(birth_1980_below ~ .-1, data =  tmp[, c(5:12)])
tmp$pred <- predict(xg.mdl, test.mtx)
#mutate(pred = ifelse(is.na(pred), runif(1), pred))
roc.curve(tmp$birth_1980_below, tmp$pred, plotit = T)

pred.dt2 = ifelse(tmp$pred>=0.5,1,0)
table(pred.dt2, tmp$birth_1980_below) %>% prop.table() ##confusion matrix
sum(pred.dt2==tmp$birth_1980_below)/nrow(tmp)## ACC
sum(pred.dt2==1 & tmp$birth_1980_below==1)/(sum(pred.dt2==1)) ## 1980+ precision
sum(pred.dt2==0 & tmp$birth_1980_below==0)/(sum(pred.dt2==0)) ## ~1980 precision
sum(pred.dt2==1 & tmp$birth_1980_below==1)/(sum(pred.dt2==1 & tmp$birth_1980_below==1) + sum(pred.dt2=="0" & tmp$birth_1980_below=="1")) ##Recall

tmp %>%
  mutate(pred = ifelse(pred >= 0.5, 1, 0)) %>%
  filter(birth_1980_below != pred)