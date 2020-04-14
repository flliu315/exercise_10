#==========================================
##buidling a model with caret
##including following steps: load data, preProcess data,
##select features, split dataset, build and assess the model
rm(list = ls())

##==========================================
#install and load caret package
install.packages("caret", dependencies = TRUE,
                 INSTALL_opts = '--no-lock')
library(caret)

##==========================================
# Load a dataset
data <- read.csv("data/npcl11.csv")
str(data)
head(data)

##==========================================
#preProcess data
library(Hmisc, quietly=TRUE)
contents(data)
summary(data)

library(fBasics, quietly=TRUE)
skewness(data, na.rm=TRUE)

#impute NA in the dataset
library(skimr) # for basic statistics
skimmed <- skim_to_wide(data)
skimmed[, 2:12]

preProcess_missingdata_model <- preProcess(data[,2:12], #build a model
                                           method='knnImpute')
preProcess_missingdata_model

library(RANN) # imputing NA algorithm
data_NA <- predict(preProcess_missingdata_model, 
                   newdata = data)
anyNA(data_NA)

#for one-hot code
dummies_model <- dummyVars(loss_rate ~ ., 
                           data= data_NA)# build a model
data_NA_dum_mat <- predict(dummies_model, 
                           newdata = data_NA)
data_NA_dum <- data.frame(data_NA_dum_mat)#rebuild a dataframe including target
loss_rate <- data_NA$loss_rate
data_clean <- cbind(loss_rate,data_NA_dum)
head(data_clean)

##=========================================
# transforming data
library(tidyverse)
data_class <- data [,-1] %>% 
  mutate(loss_rate = 
           case_when(loss_rate >= 0.4 ~ 'serious',
                     loss_rate < 0.4 ~ 'normal')) %>% 
  rename(loss_degree=loss_rate)
head(data_class)
write.csv(data_class, file = "data/data_class.csv")

##========================================
##select features
#visualize feature importance
x = as.matrix(data_class[, 1:11])
y = as.factor(data_class$loss_degree)

featurePlot(x, y, plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation = "free"), 
                          y = list(relation="free")))

featurePlot(x, y, plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))

##estimate feature importance using one of three methods (caret)
#automatically selecting a subset of the most predictive features
options(warn=-1)
set.seed(1234)

subsets <- c(1:5, 8, 11)
ctrl <- rfeControl(functions = rfFuncs, #random forest algorithem
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)
ImProfile <- rfe(x, y, 
                 sizes=subsets, 
                 rfeControl=ctrl)# run the RFE algorithm
print(ImProfile)
predictors(ImProfile)# list the chosen features
plot(ImProfile, type=c("g", "o"))# plot the results

# searching for and removing redundant features
corr_Matrix <- cor(data_class[,1:11])
print(corr_Matrix)
highlyCorr <- findCorrelation(corr_Matrix, cutoff=0.5)
print(highlyCorr)

#ranking features by importance
control <- trainControl(method="repeatedcv", 
                        number=10, repeats=3)# cross-validation
model <- train(loss_degree~., 
               data=data_class, 
               method="rf", 
               preProcess="scale", 
               trControl=control)# train the model
importance <- varImp(model, scale=FALSE)
print(importance)# summarize importance
plot(importance)# plot importance

##===============================================
##train and tune models 
#split the dataset
set.seed(1234)
train_idx <- createDataPartition(data_class$loss_degree, p=0.75, list=FALSE)
training <- data_class[train_idx,]
test <- data_class[-train_idx,]

##build rf model and evaluate its performance
#build rf model
set.seed(1234)
rf_fit <- train(as.factor(loss_degree) ~ IA + PA + CA + Q + G, 
                data = training, 
                method = "rf")
rf_fit
plot(rf_fit)

#evaluate rf performance
rf_pred <- predict(rf_fit, test)
rf_pred
confusionMatrix(reference = as.factor(test$loss_degree), 
                data = rf_pred,
                mode = "everything")

#set uneLength or tuneGrid for better model performance

ctrl <- trainControl(
  method = 'cv',                  
  number = 5,                     
  savePredictions = 'final',
  classProbs = T,                  
  summaryFunction=twoClassSummary) 

rf_fit <- train(as.factor(loss_degree) ~., #optimize mtry with tuneLength
                data = training, 
                method = "rf", 
                tuneLength = 5,
                trControl = ctrl,
                verbose = FALSE
)

#evaluate rf performance
rf_pred <- predict(rf_fit, test)
rf_pred
confusionMatrix(reference = as.factor(test$loss_degree), 
                data = rf_pred,
                mode = "everything")

library(MLeval)
x <- evalm(rf_fit)
x$roc
x$stdres



##======================================================
##build and compare models 
# Set up training control
ctrl <- trainControl(method = "repeatedcv",   
                     number = 5,	
                     # Use AUC to pick the best model
                     summaryFunction=twoClassSummary,	
                     classProbs=TRUE,
                     allowParallel = TRUE)

# selects each tuning parameter using expand.grid

##==================================================
##training multiple models
set.seed(1234)  
rpart_fit = train(as.factor(loss_degree) ~.,
                  data=training, 
                  method='rpart', 
                  tuneLength=15, 
                  trControl = ctrl)

svm_fit = train(as.factor(loss_degree) ~ .,
                data=training, 
                method='svmRadial', 
                tuneLength=15, 
                trControl = ctrl)

models_compare <- resamples(list(rpart = rpart_fit, randomForest = rf_fit, SVM= svm_fit))
summary(models_compare)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales)


##=================================================
library(caretEnsemble)

# Stacking Algorithms - Run multiple algos in one call.
ctrl <- trainControl(method="repeatedcv", 
                     number=10, 
                     repeats=3,
                     savePredictions=TRUE, 
                     classProbs=TRUE)

algorithmList <- c('rf', 'rpart', 'svmRadial')

set.seed(1234)
models <- caretList(as.factor(loss_degree) ~ .,
                    data=training, 
                    trControl=ctrl, 
                    methodList=algorithmList) 
results <- resamples(models)
summary(results)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

