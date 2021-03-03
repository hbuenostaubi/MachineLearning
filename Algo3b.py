#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 19:38:42 2020

@author: harrisonbueno
"""
import pandas as pd, os, math, numpy as np

def read_files():
    '''Movielens DATASET PATH'''
    # Read MovieLens Dataset, Mirna named the files differently than me
    path='/Users/harrisonbueno/Desktop/NEIU/20Summer/Informatics/FinalProject/MovieLens'
    os.chdir(path)
    u1_base=pd.read_csv("u1.base.txt", sep="\t", names=["USER", "ITEM", "RATING", "TIME"])
    u1_test =pd.read_csv("u1.test.txt", sep="\t", names=["USER", "ITEM", "RATING", "TIME"])
    u1_base = u1_base.drop(columns=['TIME'])
    u1_test = u1_test.drop(columns=['TIME'])
    
    ###########################################################################
    # Read and Split Yelp Dataset
    ###########################################################################
    # Read and Split Yelp Dataset
    '''YELP DATASET PATH'''
    path='/Users/harrisonbueno/Desktop/NEIU/20Summer/Informatics/FinalProject/Yelp'
    os.chdir(path)
    
    rat_df=pd.read_csv("ratings.csv")
    
    pd.options.mode.chained_assignment = None 
    
    # Split Training .8; Test .2
    rat_df['RPU'] = rat_df['User ID'].map(rat_df.groupby('User ID')['Rating'].count())
    new_df = rat_df.drop_duplicates('User ID') 
      
    new_df['Test_num']=0.0
    new_df['Training_num']=0.0
    
    ## Switch index to user ID b/c drop duplicates removed actual index
    new_df=new_df.set_index('User ID')
    
    ## Get training and test count
    for i in range(new_df.shape[0]):
        if new_df['RPU'][i+1]==1:
            new_df['Training_num'][i+1]=1
            new_df['Test_num'][i+1]=0
        elif new_df['RPU'][i+1]>=2:
            new_df['Training_num'][i+1]=math.floor(new_df['RPU'][i+1]*.8)
            new_df['Test_num'][i+1]=new_df['RPU'][i+1]-new_df['Training_num'][i+1]
    
    ## Switch from string to numbers and fill NaN w/ 0 to sum values within cells
    new_df['Train_cnt']=""
    new_df['Train_cnt']=pd.to_numeric(new_df['Train_cnt'])
    new_df['Train_cnt']=new_df['Train_cnt'].fillna(0)
    
    train_df=pd.DataFrame (columns = ['USER','ITEM','RATING'])
    test_df=pd.DataFrame (columns = ['USER','ITEM','RATING'])
    
    ### split data into training and test df
    for i in range(rat_df.shape[0]):
        if new_df['Train_cnt'][rat_df['User ID'][i]]<=new_df['Training_num'][rat_df['User ID'][i]]:
            train_df=train_df.append({'USER': rat_df['User ID'][i],
                                      'ITEM':rat_df['Business ID'][i],
                                      'RATING':rat_df['Rating'][i]},
                                      ignore_index=True)
            new_df['Train_cnt'][rat_df['User ID'][i]]=new_df['Train_cnt'][rat_df['User ID'][i]]+1
        else:
            test_df=test_df.append({'USER': rat_df['User ID'][i],
                                      'ITEM':rat_df['Business ID'][i],
                                      'RATING':rat_df['Rating'][i]},
                                       ignore_index=True)

    train_df=train_df.apply(pd.to_numeric)
    test_df=test_df.apply(pd.to_numeric)

    return u1_base, u1_test, train_df, test_df

def find_deviations(ID, rating, df, global_mean):
    dev_df = pd.DataFrame(columns = [ID,'Mean Rating', 'Deviation'])
    dev_df[ID] = df[ID]
    dev_df['Mean Rating'] = df[ID].map(df.groupby(ID)[rating].mean())
    dev_df['Deviation'] = dev_df['Mean Rating'].apply(lambda x: x - global_mean)
    dev_df = dev_df.sort_values(by=[ID])
    dev_df = dev_df.drop_duplicates(subset=ID, keep="first")
    dev_df = dev_df.set_index(ID)
    
    return dev_df

def predict_rating(testSet, itemDev, userDev, global_mean):
    testSet["PREDICTED"]=0.0
    testSet["RESIDUAL"]=0.0
    for i in range(testSet.shape[0]):
        testSet["PREDICTED"][i]=(global_mean
                + noneVal(itemDev["Deviation"].get(testSet["ITEM"][i])) 
                + noneVal(userDev["Deviation"].get(testSet["USER"][i])))
    ###had to use noneVal and .get function to modify noneType error for keyError (no index available)
    testSet["RESIDUAL"]=testSet["RATING"]-testSet["PREDICTED"]
    return testSet

#### return 0 if value is none for pred_Rating function
def noneVal(value):
    return float(0 if value is None else value)
                                    
def mean_absolute_error(testSet):
    n=len(testSet)
    sum=testSet["RESIDUAL"].apply(lambda x: abs(x)).sum()
    return round(sum/n,4)

def bu_deviation(userDev, trainSet, l, lambda1):
    for i in range(trainSet.shape[0]):
        b_u=userDev["Deviation"][trainSet["USER"][i]]
        e_ui=trainSet["RESIDUAL"][i]
        userDev["Deviation"][trainSet["USER"][i]]=b_u-(l*(-2*e_ui+2*lambda1*b_u))
        #print(userDev["Deviation"][trainSet["USER"][i]])
    return userDev

def bi_deviation(itemDev, trainSet, l, lambda1):
    for i in range(trainSet.shape[0]):
        b_i=itemDev["Deviation"][trainSet["ITEM"][i]]
        e_ui=trainSet["RESIDUAL"][i]
        itemDev["Deviation"][trainSet["ITEM"][i]]=b_i-(l*(-2*e_ui+2*lambda1*b_i))
        #print(itemDev["Deviation"][trainSet["ITEM"][i]])
    return itemDev

def loss_function(bu, bi, trainDF, lambda1):
    SSR=trainDF["RESIDUAL"].apply(lambda x: x*x).sum()
    bu_sqrd=bu["Deviation"].apply(lambda x: x*x).sum()
    bi_sqrd=bi["Deviation"].apply(lambda x: x*x).sum()
    
    temp=SSR + lambda1*(bu_sqrd+bi_sqrd)
    return round(temp,2)
    '''random list of users and items, trainingDF, learning rate, lambda, global mean 
    and number of epochs'''
def gradient(userDev, itemDev, trainingDf, l, lamb, global_mean, num):
    lst=[]
    lst.append(loss_function(userDev, itemDev, trainingDf, lamb))
    delta=999.99
    i=0
    cnt=1   ###optional
    while cnt<=num:  #< When do you want it to stop? difference of 50 is good. 
        userDev=bu_deviation(userDev, trainingDf, l, lamb)
        itemDev=bi_deviation(itemDev, trainingDf, l, lamb)
        trainingDf=predict_rating(trainingDf, itemDev, userDev, global_mean)
        i=i+1
        lst.append(loss_function(userDev, itemDev, trainingDf, lamb))
        #delta=lst[i-1]-lst[i]
        print(lst[i])
        cnt=cnt+1  ##option
    return userDev, itemDev, lst, trainingDf
#outputs userDev, itemDev, loss function list and residuals w/ new predicted values
    
def main():    
    #Read movielens and Yelp train and test files
    movieTrain, movieTest, yelpTrain,yelpTest = read_files()
    
    # 1) Statistical Computation for Movielens 
    #calculate global mean of train set
    movielens_global_mean=movieTrain["RATING"].mean()
    
    #find bu for each user and bi for each item
    movielens_userDev = find_deviations("USER", "RATING", movieTrain, movielens_global_mean)
    movielens_itemDev = find_deviations("ITEM", "RATING", movieTrain, movielens_global_mean)
    

    ###############
    ###############    Gradient Descent with random values
    ###############
    
    ### create random floats for each item list and user list
    userDevMov = movielens_userDev
    userDevMov = userDevMov.drop(columns=['Mean Rating'])
    userDevMov["Deviation"] =np.random.uniform(-1,3,size= len(userDevMov))
    
    itemDevMov = movielens_itemDev
    itemDevMov = itemDevMov.drop(columns=['Mean Rating'])
    itemDevMov["Deviation"] = np.random.uniform(-1,3,size= len(itemDevMov))
    
    
    ### generate predicted values and residuals for the training data
    dataTrain = predict_rating(movieTrain, itemDevMov, userDevMov,movielens_global_mean)
    
                                                    
    userDevMov, itemDevMov, lst, dataTrain=gradient(userDevMov,itemDevMov,
                                                    dataTrain, .01,.5,
                                                    movielens_global_mean, 10)
            
    ##output MAE the training deviations into the test set (output residuals and predictions)
    movieTest2=predict_rating(movieTest, itemDevMov, userDevMov, movielens_global_mean) 
    
    ## get Mean Absolute Error from the residuals
    movie_mae2=mean_absolute_error(movieTest2)
    print()
    print('MAE random values and gradient descent (3b):', movie_mae2)
    ## compare against 
   
    
    
    
    ###########################################################################
    
    
    # 2) Stastical Computation for Yelp Data
    #calculate global mean of train set
    yelp_global_mean=yelpTrain["RATING"].mean()
    
    #find bu for each user and bi for each item
    yelp_userDev = find_deviations("USER", "RATING", yelpTrain, yelp_global_mean)
    yelp_itemDev = find_deviations("ITEM", "RATING", yelpTrain, yelp_global_mean)

    ###############
    ###############    Gradient Descent with random values
    ###############
    
    '''
    Gradient Descent with Yelp Data Set
    '''
    
    ### create random floats for each item list and user list
    userDevY = yelp_userDev
    userDevY = userDevY.drop(columns=['Mean Rating'])
    userDevY["Deviation"] =np.random.uniform(-1,3,size= len(yelp_userDev))
    
    itemDevY = yelp_itemDev
    itemDevY = itemDevY.drop(columns=['Mean Rating'])
    itemDevY["Deviation"] = np.random.uniform(-1,3,size= len(yelp_itemDev))
    
    ### generate predicted values and residuals for the training data
    dataTrainY = predict_rating(yelpTrain, itemDevY, userDevY,yelp_global_mean)
    
    userDevY, itemDevY, lst2, dataTrainY=gradient(userDevY,itemDevY,
                                                    dataTrainY, .001,.5,
                                                    yelp_global_mean, 10)

    ## get residuals for test data vs train deviations
    yelpTest2= predict_rating(yelpTest, itemDevY,userDevY,yelp_global_mean)
    
    #Calculate the MAE for gradient descent
    yelp_MAE2 = mean_absolute_error(yelpTest2)
    print('MAE of yelp dataset w/ rand values (3b):', yelp_MAE2)
    
if __name__ == "__main__":
    main()
