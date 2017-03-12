#################################################
# Author: Zach King                             #
#                                               #
# Description: Ensemble of RandomForest and     #
#  GBM models forpredicting geographic location #
#  of mobile device Facebook "check-in's"       #
#                                               # 
#################################################

library(randomForest)
library(gbm,lib.loc='dir')#use this ver of gbm since repo ver has mem leak
setwd('~/facebook/')
df_train<-read.csv('train.csv',header=T, na.strings=c("","NA"))
df_test<-read.csv('test.csv',header=T,na.strings=c("","NA"))
df_train1A<-df_train[df_train$x>=0 & df_train$x<=10 & df_train$y>=0 & df_train$y<=1,]
df_test1A<-df_test[df_test$x>=0 & df_test$x<=10 & df_test$y>=0 & df_test$y<=1,]

####Feature Engineering########
df_train1A$minute<-ceiling(df_train1A$time %% 60)
df_train1A$hour<-(df_train1A$time %% (60*24))/60
df_train1A$dayofweek<-ceiling((df_train1A$time %% (60*24*7))/(60.*24))
df_train1A$dayofmonth<-ceiling((df_train1A$time %% (60*24*30))/(60.*24))
df_train1A$dayofyear<-ceiling((df_train1A$time %% (60*24*365))/(60.*24))
df_train1A$month<-ceiling((df_train1A$time %% (60*24*30*12))/(60.*24*30))
df_train1A$quarter<-ceiling((df_train1A$time %% (60*24*90*4))/(60.*24*90))
df_train1A$year<-ceiling((df_train1A$time %% (60*24*365*10))/(60.*24*365))
df_train1A$logacc<-log(df_train1A$accuracy)
df_train1A<-within(df_train1A, {
  acc10 = ifelse(accuracy ==10, 1, 0)
  acc65 = ifelse(accuracy ==65, 1, 0)
  acc165 = ifelse(accuracy ==165,1,0)
})

df_test1A$minute<-ceiling(df_test1A$time %% 60)
df_test1A$hour<-(df_test1A$time %% (60*24))/60
df_test1A$dayofweek<-ceiling((df_test1A$time %% (60*24*7))/(60.*24))
df_test1A$dayofmonth<-ceiling((df_test1A$time %% (60*24*30))/(60.*24))
df_test1A$dayofyear<-ceiling((df_test1A$time %% (60*24*365))/(60.*24))
df_test1A$month<-ceiling((df_test1A$time %% (60*24*30*12))/(60.*24*30))
df_test1A$quarter<-ceiling((df_test1A$time %% (60*24*90*4))/(60.*24*90))
df_test1A$year<-ceiling((df_test1A$time %% (60*24*365*10))/(60.*24*365))
df_test1A$logacc<-log(df_test1A$accuracy)
df_test1A<-within(df_test1A, {
  acc10 = ifelse(accuracy ==10, 1, 0)
  acc65 = ifelse(accuracy ==65, 1, 0)
  acc165 = ifelse(accuracy ==165,1,0)
})

rm(df_train,df_test)

######GBM plus Random Forest ensemble#########
boost_iter<-function(df,df2,x,y,r) { #x and y are number of 1km sq blocks. r is starting row
  #cat.top3 returns the top 3 likely predictions. Used for map@3 indicator
  cat.top3<-function(p){
    maxn <- function(n) function(x) order(x, decreasing = TRUE)[n]
    max1<-apply(p,1,maxn(1))
    max2<-apply(p,1,maxn(2))
    max3<-apply(p,1,maxn(3))
    max1p<-vector()
    max2p<-vector()
    max3p<-vector()
    boost.top3.pred<<-vector()
    i=1
    for (i in 1:length(max1)){
      colnum1<-as.integer(max1[i])
      colname1<-colnames(p)[colnum1]
      max1p[i]<-colname1
      colnum2<-as.integer(max2[i])
      colname2<-colnames(p)[colnum2]
      max2p[i]<-colname2
      colnum3<-as.integer(max3[i])
      colname3<-colnames(p)[colnum3]
      max3p[i]<-colname3
      boost.top3.row<-paste(max1p[i],max2p[i],max3p[i],collapse=" ")
      boost.top3.pred[i]<<-boost.top3.row
    }
  }
  preds<<-list()
  L=.1 #model built for every .1km square
  i=1
  i2=1
  n=(y/L)
  n2=(x/L)
  Ly=r
  Ly2=r+L
  for (i in 1:n) {
    Lx=0
    Lx2=L
    for (i2 in 1:n2) {
      train_slice<-df[df$x>=Lx & df$x<=Lx2 & df$y>=Ly & df$y<=Ly2,]
      test_slice<-df2[df2$x>=Lx & df2$x<=Lx2 & df2$y>=Ly & df2$y<=Ly2,]
      train_slice$place_id<-as.factor(train_slice$place_id)
      train_slice<-train_slice[,-c(1)]
      test_slice<-test_slice[order(rownames(test_slice)),]
      gbm.fit<-gbm(place_id~.-time,n.trees=300, interaction.depth=5,shrinkage=.001,data=train_slice)
      gbm.pred<-predict(gbm.fit,test_slice,n.trees=300)
      print("gbm done")
      forest.fit<-randomForest(place_id~.,data=train_slice,ntree=500)
      forest.predict<-predict(forest.fit, test_slice, type="prob")      
      gbm.predplog<-matrix(ncol=ncol(forest.predict),nrow=length(forest.predict[,1]),plogis(gbm.pred))
      ensemble<-forest.predict+gbm.predplog
      cat.top3(ensemble)
      boost.top3.pred<-as.matrix(boost.top3.pred)
      rownames(boost.top3.pred)<-rownames(test_slice)
      len<-length(preds)+1
      preds[[len]]<<-boost.top3.pred
      print(Lx)
      print(Lx2)
      print(Ly)
      print(Ly2)
      Lx=Lx+L
      Lx2=Lx2+L
      i2=i2+1
      rm(boost.top3.pred,train_slice,test_slice,gbm.pred,
         gbm.fit,ensemble,gbm.predplog,forest.fit,forest.predict)
      gc()
    }
    Ly=Ly+L
    Ly2=Ly2+L
    i=i+1
  }
  preds<<-do.call("rbind",preds)
}

boost_iter(df_train1A,df_test1A,10,1,0)
write.csv(preds,file='boostloop_preds1A.csv')

#Illustration of row being operated#

# # # # # # # # # #
# # # # # # # # # #
# # # # # # # # # #
# # # # # # # # # #
# # # # # # # # # #
# # # # # # # # # #
# # # # # # # # # #
# # # # # # # # # #
#xxxxxxxxxxxxxxxxx#

#There are 9 other scripts that operate on the other 9 rows.  All
#10 scripts can be run in parrallel using seperate R instances or VM's
