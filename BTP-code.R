setwd("~/Downloads/Practice Code1")
library("neuralnet")
dataset <- read.csv("CCfraudD.csv", header=T)
trainset <- dataset[1:866, ]
testset <- dataset[867:965, ]
creditnet <- neuralnet(Y~ X11+X12+	X13	+X14	+X15	+X16	+X21	+X22	+X23	+X24+	X25	+X31	+X32+	X33	+X34	+X35	+X36	+X41	+X42	+X43+	X44	+X45	+X51	+X52	+X53	+X54, trainset, hidden = 3, lifesign = "minimal", 
                       linear.output = FALSE, threshold = 0.001)
plot(creditnet, rep = "best")
temp_test <- subset(testset, select = c("X11", "X12","X13","X14","X15","X16","X21","X22","X23","X24","X25","X31","X32","X33","X34","X35","X36","X41","X42","X43","X44","X45","X51","X52","X53","X54"))
creditnet.results <- compute(creditnet, temp_test)
head(temp_test)
results <- data.frame(actual = testset$Y, prediction = ifelse(creditnet.results$net.result> 0.5,1,0))
write.csv(results,file='BTPNN.csv')
library("caret")
confusionMatrix(results$prediction,results$actual, positive = NULL, 
                dnn = c("Prediction", "Reference"), 
                prevalence = NULL)

#Using SVM
library(e1071)
nbrow=nrow(dataset)
# use 90 % of objects as train
class <-as.character(dataset[,1])
attribs <- dataset[,c(-1)]
ntrain <- round(nbrow*0.9)
# sample
tindex <- sample(nbrow,ntrain) # indices of training samples train <- attribs[tindex,]
train <- attribs[tindex,]
test <- attribs[-tindex,]
classtrain <- class[tindex]
classtrain
classtest <- class[-tindex]
svmmodel <- svm(train, classtrain,cost = 100, gamma = 0.01,type="C-classification")
summary(svmmodel)
prediction <- predict(svmmodel,test)
tab <- table(pred = prediction, true = classtest)
# write the confusion matrix...
write.table(tab,"confusion.txt")
results <- data.frame(actual = classtest, predicted = prediction)
write.csv(results,file='BTPSVM.csv')
library("caret")
confusionMatrix(results$predicted,results$actual, positive = NULL, 
                dnn = c("Prediction", "Reference"), 
                prevalence = NULL)
#Using GLM
myregress <- glm(Y~ X11+X12+	X13	+X14	+X15	+X16	+X21	+X22	+X23	+X24+	X25	+X31	+X32+	X33	+X34	+X35	+X36	+X41	+X42	+X43+	X44	+X45	+X51	+X52	+X53	+X54
                 , family = binomial(link = "logit"), data = trainset)
summary(myregress)
plot(myregress)
prediction <- predict(myregress,newdata=testset,type='response')
results <- data.frame(actual = testset$Y, prediction = ifelse(prediction > 0.5,1,0))
write.csv(results,file='BTPlogit.csv')
library("caret")
confusionMatrix(results$prediction,results$actual, positive = NULL, 
                dnn = c("Prediction", "Reference"), 
                prevalence = NULL)

#Using Decision Trees
library("rpart")


library("party")
treeout1 <- ctree( Y~ X11+X12+	X13	+X14	+X15	+X16	+X21	+X22	+X23	+X24+	X25	+X31	+X32+	X33	+X34	+X35	+X36	+X41	+X42	+X43+	X44	+X45	+X51	+X52	+X53	+X54
                  ,data = trainset)
summary(treeout1)
plot(treeout1)
table(predicted = predict(treeout1), Input = trainset$Y)
tr.pred = predict(treeout1, newdata=testset, type="prob")
treeresults <- data.frame(actual = testset$Y, prediction = ifelse(tr.pred > 0.5,1,0))
write.csv(treeresults,file='BTPtree.csv')
library("caret")
confusionMatrix(treeresults$prediction,treeresults$actual, positive = NULL, 
                dnn = c("Prediction", "Reference"), 
                prevalence = NULL)

#Using Random Forests