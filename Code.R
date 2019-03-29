library("car") # vif()
library(DAAG) # library for CV
library("caret")

path = "F:/Company/TresVista/Test 1.csv"
customer <- read.csv(path,header = T)
View(customer)

colnames(customer)

ncol(customer)
nrow(customer)

str(customer)


#check for NA, and Blanks
col_name = colnames(customer) [apply(customer, 2, function(n) any(is.na(n)))]
if(length(col_name) > 0)
{
  print("Blank present in columns : ")
  print(col_name)
} else 
  print("No NA")

col_name = colnames(customer) [apply(customer, 2, function(n) any(n==""))]
if(length(col_name) > 0)
{
  print("Blank present in columns : ")
  print(col_name)
} else 
  print("No Blanks")

col_name = colnames(customer) [apply(customer, 2, function(n) any(n==0))]
if(length(col_name) > 0)
{
  print("Zeroes present in columns : ")
  print(col_name)
} else 
  print("No Zeroes")


col_name = colnames(customer) [apply(customer, 2, function(n) any(n=='?'))]
if(length(col_name) > 0)
{
  print("? present in columns : ")
  print(col_name)
} else 
  print("No ?")

levels(customer$demographic_slice)
table(customer$card_offer)
#________________ Delete Specific column _________________
customer$customer_id <- NULL

100*prop.table(table(customer$card_offer))
100*prop.table(table(customer$ad_exp))

# Data Visualization
library(ggplot2)
plot(customer$est_income)
boxplot(customer$est_income)
hist(customer$est_income)
hist(customer$imp_cscore)

boxplot(customer$imp_cscore)
boxplot(customer$RiskScore)
hist(customer$axio_score)
boxplot(customer$imp_crediteval)

ncol(customer)

levels(factor(customer$card_offer))
#rename
customer$card_offer[customer$card_offer == "FALSE"] = 0
customer$card_offer[customer$card_offer == "TRUE"] = 1
View(customer)

table(customer$card_offer)
# randomly shuffle the dataset
grp = runif(nrow(customer))
customer = customer[order(grp),]


View(customer)


# split data into training and test
sample_size = floor(0.7*nrow(customer))
sample_ind = sample(seq_len(nrow(customer)), sample_size)
train = customer[sample_ind,]
test = customer[-sample_ind,]

nrow(train)
nrow(test)
ncol(train)

table(train$card_offer)


###################### Model Building #####################

#-----------------------------------------

#############################################
# build the logistic regression model
# GLM - generalised linear model
glm1 = glm(card_offer ~ ., family = "binomial", data=train)

# summarise model
summary(glm1)

# exp(coefficients) will give the odds
glm1$coefficients[]
glm1$coefficients[3]


###################################################
# cross validation technique
library(e1071)

train_Control = trainControl(method="cv", number=10)
cvmodel1 = train(card_offer~., data=train, trControl=train_Control,
                 method="glm", family=binomial())
summary(cvmodel1)

cvmodel1$results
cvmodel1$finalModel
pdct_cv = predict(glm1,train,type="response")
pdct_cv = ifelse(pdct_cv <= 0.45, 0, 1)

# build the confusion matrix
table(predicted = pdct_cv, actual = train$card_offer)


# predict the Y-values
predict_y = predict(glm1,test,type="response") #response means prob.
head(predict_y)
predict_y = ifelse(predict_y <= 0.45, 0, 1)

# build the confusion matrix for Testing Dataset
table(predicted = predict_y, actual = test$card_offer)

#######################################################
table(customer$demographic_slice)
#2 model in logistic Regression
#----------------------
cust1 <- customer[,c("demographic_slice","country_reg","est_income","hold_bal",
                  "pref_cust_prob" ,"imp_cscore","card_offer")]
View(cust1)


# split data into training and test
sample_size = floor(0.7*nrow(cust1))
sample_ind = sample(seq_len(nrow(cust1)), sample_size)
train1 = cust1[sample_ind,]
test1 = cust1[-sample_ind,]

glm2 = glm(card_offer ~ ., family = "binomial", data=train1)

# summarise model
summary(glm2)

# exp(coefficients) will give the odds
glm1$coefficients[]
glm1$coefficients[3]

View(train1)

###################################################
# cross validation technique

train_Control = trainControl(method="cv", number=10)
cvmodel2 = train(card_offer~., data=train1, trControl=train_Control,
                 method="glm", family=binomial())
summary(cvmodel2)

cvmodel2$results
cvmodel2$finalModel
pdct_cv = predict(glm2,train1,type="response")
pdct_cv = ifelse(pdct_cv <= 0.45, 0, 1)

# build the confusion matrix
table(predicted = pdct_cv, actual = train1$card_offer)


# predict the Y-values
predict_y1 = predict(glm2,test1,type="response") #response means prob.
head(predict_y1)

table(train1$card_offer)

predictions = ifelse(predict_y1 <=0.5, 0,1) 

# build the confusion matrix
table(predicted = predictions,actual = test1$card_offer)

#check if values are right or not using the formulas in pdf model-building and evaluation.


hist(predict_y1, col = "red")

# change the cut-off values and check for which cutoff gives more accuracy.
cutoff = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
for (i in cutoff){
  predictions = ifelse(predict_y <=i, 0, 1)
  print(i)
  print(table(actual = test$card_offer, predicted = predictions))
  cat("\n")
}


# feature selection technique
# -----------------------------
step(glm2)


# confusion matrix statistics
library(caret)
predictions = ifelse(predict_y1 <=0.5, 0,1) 
# param1 -> actual Y
# param2 -> predicted Y
# param3 -> positive class (0/1, Yes/No etc..)
table(actual = test1$card_offer, predicted = predictions)

library(ROCR)
# ROC curve is generated by plotting TPR against the FPR
# 0.5 < GOOD_MODEL <= 1
pr = prediction(predict_y1, test1$card_offer) #it hepls to perform ROC curve
summary(pr)


#evaluation
evals = performance(pr,"acc") #it returns values in slots
evals
plot(evals)
abline(h=0.83, v=0.46) #this gives horizontal and vertical lines to detect peak point near the upper line
#h and v stands for horizontal and vertical

#identifying the optimal values for best accuracy
#performance values are stored in slots
#display "evals" to understand the output
#y.values = accuracy
#x.values = cutoff

evals
max_yval = which.max(slot(evals, "y.values")[[1]])

max_acc = slot(evals,"y.values")[[1]][max_yval]
max_cutoff = slot(evals,"x.values")[[1]][max_yval]
print(paste("Best accuracy = ",round(max_acc,4),
            "Best cutoff = ",round(max_cutoff,4)))


perf = performance(pr, measure = "tpr", x.measure = "fpr")
plot(perf)

abline(a=0, b=1)

#area under this curve(AUC)
auc = performance(pr,"auc")
round(unlist(slot(auc,"y.values")),3) #3 gives value upto 3 decimal points

#colored graph
plot(perf,colorize=T,main="ROC Curve",ylab = "sensitivity",
     xlab="1-specificity")
abline(a=0, b=1)


#------------------Random Forest -------------------------

#  call the randomforest() for Classification
# ---------------------------------------------------------------------

# 1st Model:

train_x = train[,1:10]
train_y = train[,11]
ncol(train_x)
library(randomForest)
rf1 = randomForest(train_x, factor(train_y)) #Build model 
rf1

importance(rf1)
summary(rf1)
# predict the Classification for the Testing data Using Random Forest
# ------------------------------------------------
test$pdct_rf1 = predict(rf1, test)
View(test)


# Confusion Matrix to check Accuracy
table(predicted = pdct_rf1 , actual = test$card_offer)

#-----------------------------------------
#2nd model using Random Forest

train1_x = train1[,1:6]
train1_y = train1[,7]
ncol(train1_x)

rf2 = randomForest(train1_x, factor(train1_y)) #Build model 
rf2

importance(rf2)
summary(rf2)
# predict the Classification for the Testing data Using Random Forest
# ------------------------------------------------
pdct_rf2 = predict(rf2, test1)
test1$pdct_rf2 = predict(rf2, test1)
View(test1)


# Confusion Matrix to check Accuracy
table(predicted = pdct_rf2 , actual = test1$card_offer)


########################################################################
# Applying Second model of Random Forest algorithm on Actual Testing Dataset

path = "F:/Company/TresVista/Test 2.csv"
Testing <- read.csv(path,header = T)

#check for NA, and Blanks
col_name = colnames(Testing) [apply(Testing, 2, function(n) any(is.na(n)))]
if(length(col_name) > 0)
{
  print("Blank present in columns : ")
  print(col_name)
} else 
  print("No NA")

col_name = colnames(Testing) [apply(Testing, 2, function(n) any(n==""))]
if(length(col_name) > 0)
{
  print("Blank present in columns : ")
  print(col_name)
} else 
  print("No Blanks")

col_name = colnames(Testing) [apply(Testing, 2, function(n) any(n==0))]
print(col_name)
if(length(col_name) > 0)
{
  print("Zeroes present in columns : ")
  print(col_name)
} else 
  print("No Zeroes")

Testing$customer_id <- NULL
Testing$card_offer <- NULL



Testing1 <- Testing[,c("demographic_slice","country_reg","est_income","hold_bal",
                     "pref_cust_prob" ,"imp_cscore")]
View(Testing)


# predict the Classification for the Testing data Using Random Forest
# ------------------------------------------------
Testing$card_offer = predict(rf2, Testing)
View(Testing)
str(Testing)

levels(Testing$card_offer) <- c("FALSE","TRUE")


# Exporting dataset in ds4.csv

write.csv(Testing,"F:/Company/TresVista/ds4.csv")
