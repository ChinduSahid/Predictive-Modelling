library(PRROC)

## Predictive modelling

#German credit data: This well-known data set is used to classify customers as having good or bad credit based on customer attributes
#(e.g. information on bank accounts or property). The data can be found at the UC Irvine Machine Learning Repository and in the caret R package.   


german_credit = read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
str(german_credit)

colnames(german_credit) = c("chk_acct", "duration", "credit_his", "purpose", 
                            "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", 
                            "present_resid", "property", "age", "other_install", "housing", "n_credits", 
                            "job", "n_people", "telephone", "foreign", "response")

german_credit$response = german_credit$response - 1
german_credit$response <- as.factor(german_credit$response)

# Split data into training and testing with a 70/30 split

set.seed(123)
in.train <- createDataPartition(as.factor(german_credit$response), p=0.7, list=FALSE)
train <- german_credit[in.train,]
test <- german_credit[-in.train,]

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
pred1<- ifelse(pred > 0.5,1,0)

# Confusion matrix
confusionMatrix(table(pred1,y_act), positive='1')
confusionMatrix(table(pred1,y_act), positive='1',mode = "prec_recall")

fg1 <- pred[test$response == 1]
bg1 <- pred[test$response == 0]


# ROC Curve    
roc1 <- PRROC::roc.curve(scores.class0 = fg1, scores.class1 = bg1, curve = T)
plot(roc1)

# Pr CUrve
pr <- pr.curve(scores.class0 = fg1, scores.class1 = bg1, curve = T)
plot(pr)
