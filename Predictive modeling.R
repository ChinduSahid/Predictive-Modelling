library(PRROC)
library(rpart)
library(caret)
### Predictive modelling

#German credit data: This well-known data set is used to classify customers as having good or bad credit based on customer attributes
#(e.g. information on bank accounts or property). The data can be found at the UC Irvine Machine Learning Repository and in the caret R package.   


german_credit = read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
str(german_credit)

colnames(german_credit) = c("chk_acct", "duration", "credit_his", "purpose", 
                            "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", 
                            "present_resid", "property", "age", "other_install", "housing", "n_credits", 
                            "job", "n_people", "telephone", "foreign", "response")

german_credit$response <- as.factor(german_credit$response)

# Split data into training and testing with a 70/30 split

set.seed(123)
in.train <- createDataPartition(as.factor(german_credit$response), p=0.7, list=FALSE)
train <- german_credit[in.train,]
test <- german_credit[-in.train,]
View(train)
table(train$response)

# Basic logistic regression model
model1 <- glm(response ~ ., family = binomial, train)

#variable selection using AIC (Backward elimination)

StepModel1 <- step(model1, direction = "backward")
summary(StepModel1)

#chi-square test for significance of variables
drop1(model1, test ="Chi")

# using significant variables based on AIC selection

# logistic regression
Model2<- glm(formula = response ~ chk_acct + duration + credit_his + purpose + 
               amount + saving_acct + present_emp + other_debtor + other_install + 
               housing + foreign, family = binomial, data = train)
pred <- predict(Model2, type = "response",newdata = test)
y_act <- test$response
pred1<- ifelse(pred > 0.5,2,1)

# Confusion matrix
confusionMatrix(table(pred1,y_act), positive='2')
confusionMatrix(table(pred1,y_act), positive='2',mode = "prec_recall")

fg1 <- pred[test$response == 2]
bg1 <- pred[test$response == 1]


# ROC Curve    
roc1 <- PRROC::roc.curve(scores.class0 = fg1, scores.class1 = bg1, curve = T)
plot(roc1)

# Pr CUrve
pr <- pr.curve(scores.class0 = fg1, scores.class1 = bg1, curve = T)
plot(pr)

# Decision tree
m1 <- rpart(response~.,
            method="class", data=train)

pdata <- as.data.frame(predict(m1, newdata = test, type = "p"))
pdata$my_custom_predicted_class <- ifelse(pdata$`2`> .5, 2,1)
pdata$my_custom_predicted_class<-factor(pdata$my_custom_predicted_class)
test$response<-factor(test$response)
# confusion matrix
caret::confusionMatrix(data = pdata$my_custom_predicted_class, 
                       reference = test$response, positive = "2")
fg2 <- pdata$`2`[test$response == 2]
bg2 <- pdata$`2`[test$response == 1]

roc2 <- PRROC::roc.curve(scores.class0 = fg2, 
                         scores.class1 = bg2, curve = T)
plot(roc2)

pr2 <- pr.curve(scores.class0 = fg2, scores.class1 = bg2, curve = T)
plot(pr2)


# Comparision of two models based on ROC
plot(roc1,col=1,lty=2,main="ROC")
plot(roc2,col=2,lty=2,add=TRUE)


legend(x="bottomright", 
       legend= c("Logistic regression", 
                 "Decision tree"),
       fill = 1:5)

