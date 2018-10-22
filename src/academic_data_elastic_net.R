if (!require(data.table)) {
  install.packages('data.table')
  suppressPackageStartupMessages(library(data.table))
}

if (!require(glmnet)) {
  install.packages('glmnet')
  suppressPackageStartupMessages(library(glmnet))
}

set.seed(2018)
datapath = '~/github-repos/2016-summer-jordan-coral-reef-regression/data/Wagner_et_al_test_data.csv'
data = data.table(read.csv(datapath))
train_rows = sample(seq(nrow(data)), size=0.8*nrow(data))
data_train = data[train_rows,]
data_test = data[-train_rows,]

y1 = data[, FishAbund]
y2 = data[, FishRich]
x = as.matrix(data[, .(PatchVol, CaveProp, 
                       CoralProp, LdimAbund)])

y1_train = data_train[, FishAbund]
y1_test = data_test[, FishAbund]
y2_train = data_train[, FishRich]
y2_test = data_test[, FishRich]
x_train = as.matrix(data_train[, .(PatchVol, CoralProp, LdimAbund)])
x_test = as.matrix(data_test[, .(PatchVol, CoralProp, LdimAbund)])

# Tune elastic net model
alpha_perf = data.table()
for (i in 1:10) {
  fit1 = cv.glmnet(x_train, y1_train, type.measure='mse', 
                  alpha=i/10, family='gaussian')
  fit2 = cv.glmnet(x_train, y2_train, type.measure='mse', 
                   alpha=i/10, family='gaussian')
  yhat1 = predict(fit1, s=fit1$lambda.min, newx=x_test)
  yhat2 = predict(fit2, s=fit2$lambda.min, newx=x_test)
  mse1 = mean((y1_test - yhat1)^2)
  mse2 = mean((y2_test - yhat2)^2)
  mpe1 = mean(abs((y1_test - yhat1) / y1_test))
  mpe2 = mean(abs((y2_test - yhat2) / y2_test))
  #rsq1 = (1-sum((y1_test-yhat1))
  perf = data.table(alpha=i/10, mse1=mse1, mse2=mse2,
                    mpe1=mpe1, mpe2=mpe2)
  alpha_perf = rbind(alpha_perf, perf)
}

# Regular linear model
fit1 = lm(FishAbund ~ PatchVol + CoralProp + LdimAbund, 
          data=data_train)
fit2 = lm(FishRich ~ PatchVol + CoralProp + LdimAbund, 
          data=data_train)
yhat1 = predict(fit1, newdata=data_test)
yhat2 = predict(fit2, newdata=data_test)
mse1 = mean((y1_test - yhat1)^2)
mse2 = mean((y2_test - yhat2)^2)
mpe1 = mean(abs((y1_test - yhat1) / y1_test))
mpe2 = mean(abs((y2_test - yhat2) / y2_test))
results = data.table(mse1=mse1, mse2=mse2,
                     mpe1=mpe1, mpe2=mpe2)
