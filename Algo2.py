#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:16:00 2020

@author: harrisonbueno
"""

import pandas as pd, os, math, numpy as np

def read_files():
    '''YELP DATASET PATH'''
    path='/Users/harrisonbueno/Desktop/NEIU/20Summer/Informatics/FinalProject/Yelp'
    os.chdir(path) 
    
    # Read and split Yelp dataset
    # split Training .8; Test .2
    rat_df=pd.read_csv("ratings.csv")
    pd.options.mode.chained_assignment = None
    rat_df['RPU'] = rat_df['User ID'].map(rat_df.groupby('User ID')['Rating'].count())
    new_df = rat_df.drop_duplicates('User ID')  
    new_df['Test_num']=0.0
    new_df['Training_num']=0.0
    new_df=new_df.set_index('User ID')
    
    # Get training and test count
    for i in range(new_df.shape[0]):
        if new_df['RPU'][i+1]==1:
            new_df['Training_num'][i+1]=1
            new_df['Test_num'][i+1]=0
        elif new_df['RPU'][i+1]>=2:
            new_df['Training_num'][i+1]=math.floor(new_df['RPU'][i+1]*.8)
            new_df['Test_num'][i+1]=new_df['RPU'][i+1]-new_df['Training_num'][i+1]

    # Switch from string to numbers and fill NaN w/ 0 to sum values within cells
    new_df['Train_cnt']=""
    new_df['Train_cnt']=pd.to_numeric(new_df['Train_cnt'])
    new_df['Train_cnt']=new_df['Train_cnt'].fillna(0)

    train_df=pd.DataFrame (columns = ['USER','ITEM','RATING'])
    test_df=pd.DataFrame (columns = ['USER','ITEM','RATING'])

    # split data into training and test df
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
        
    train_df=train_df.apply(pd.to_numeric)
    test_df=test_df.apply(pd.to_numeric)
    '''MOVIELENS DATASET PATH'''
    # Read movie dataset
    path='/Users/harrisonbueno/Desktop/NEIU/20Summer/Informatics/FinalProject/MovieLens'
    os.chdir(path)

    u1_base = pd.read_csv("u1.base.txt", sep="\t", names=["USER", "ITEM", "RATING", "TIME"])
    u1_test =pd.read_csv("u1.test.txt", sep="\t", names=["USER", "ITEM", "RATING", "TIME"])
    
    return train_df, test_df, u1_base, u1_test
    
# Normalize Function
def normalize(rowname, colname,vals, df):
    df[vals]=pd.to_numeric(df[vals])
    df_group=df.groupby(colname)[vals].mean()
    
    df['avg_rating']=""
    #suppress copy warning->none, warn or raise
    pd.options.mode.chained_assignment = None 
    for i in range(df.shape[0]):
        df['avg_rating'][i]=df_group[df[colname][i]]
        
    df['Norm_rating']=df[vals]-df['avg_rating']
    return round(df.pivot(index=rowname,
                     columns=colname,
                     values='Norm_rating').astype(float),3)

#gets the cos similarity between two dataframe COLUMNS (look at formula for knn)
def get_dist(df_clm, df_clm2):
    dist=(np.nansum(df_clm*df_clm2) /
               math.sqrt(np.nansum((df_clm/df_clm2)*df_clm2*df_clm)) *
               math.sqrt(np.nansum((df_clm2/df_clm)*df_clm*df_clm2)))
    if math.isnan(dist)==True:
        dist=999.99
    return dist

#Iterates through pivot table and outputs the lowest similarity tables by k
#Item_num is the column index for item, df is the normalized df, k is the total ks
def knn(Item_num, df, k):
    lst=pd.DataFrame(columns=["Item", "Dist"])
    
    for i in range(df.shape[1]):
        index=df.columns[i]
        if Item_num!=index:
            ###get distance
            dist=get_dist(df[Item_num], df[index])
            
            if lst.shape[0]<k:                
                lst=lst.append({"Item":index,
                               "Dist":dist},
                                ignore_index=True)
            else:
                temp=max(lst["Dist"])
                if(temp>dist):
                    temp2=lst["Dist"].idxmax()
                    lst["Item"][temp2]=index
                    lst["Dist"][temp2]=dist
                    
    lst=lst.sort_values(by=['Dist'])
    return lst.set_index(pd.Series(range(lst.shape[0])))

# It takes the output from knn and training set and outputs a weighted average for the original 
# Item_num and closest neights for the item in knn
def weighted_average(sim_df, df):
    numer=0.0
    denom=0.0
    for i in range(sim_df.shape[0]):
        numer=numer+np.nansum(df[sim_df["Item"][i]])
        denom=denom+abs(df[sim_df["Item"][i]].count())
    return round(numer/denom,2)    

##finds user ratings to be evaluated against test set
## user num is the user id, dataTrain is ratings, dataTrain_norm are normalized
#k is the number of ks to be compared dataTest is the test dataframe (they're subsetted)
def findUser_ratings(userNum, dataTrain, dataTrain_norm,k, dataTest):
    lst=[]
    dataTest=dataTest[dataTest.USER==userNum]
    dataTest=dataTest.set_index(pd.Series(range(dataTest.shape[0])))
    
    df=pd.DataFrame (columns = ['ITEM','RATING'])
    for i in range(dataTest.shape[0]):
        col_ind=dataTest["ITEM"][i]
        if math.isnan(dataTrain[userNum][col_ind])==True:
            lst=knn(col_ind, dataTrain_norm, k)
            df=df.append({'ITEM':col_ind,
                          'RATING': weighted_average(lst, dataTrain)}, 
                         ignore_index=True)
    
    return df, dataTest
##outputs the MAE of the test vs user rating 
def out_mae(user_rating, test_rating):
    df=user_rating
    n=len(df)
    df['RESIDUAL']=test_rating['RATING']-user_rating['RATING']
    sum=df["RESIDUAL"].apply(lambda x: abs(x)).sum()
    return round(sum/n,2)

def main():
    yelpTrain, yelpTest, movieTrain, movieTest = read_files()
    
    
    # Normalize Yelp
    df_norm = normalize('USER', 'ITEM', 'RATING', yelpTrain)
    
    #example of K-Nearest Neighbors and weighted average
    ##𝑘 􏰊 1, 5, 10, 50, and 100   
    #lst3 = knn(2, df_norm,1)
    
    #Getting table for item ratings (not normalized)
    yelpTrain2 = yelpTrain.pivot(index="USER",columns="ITEM",values="RATING")
    #print(weighted_average(lst3, yelpTrain2))
    
    ##find ratings  (user, dataTrain, norm data, k, dataTest)
    user2, test=findUser_ratings(2, yelpTrain2, df_norm, 100,yelpTest)

    
    ###output the MAE for test vs training
    mae1=out_mae(user2, test)
    print("Mean Absolute Error for Yelp Data Set: ", mae1)
    #1.33,1.02, 1.02, .87, .86
    
    ###############################################################################
    ###########################################################################
    ##     Movie Lense data set
    
    # Normalize Movie
    df2_norm=normalize('USER', 'ITEM', 'RATING', movieTrain)
    # Get Movie data training set (not normalized)
    movieTrain2 = movieTrain.pivot(index="USER",columns="ITEM",values="RATING")
    
    user3, test3=findUser_ratings(3, movieTrain2, df2_norm, 100,movieTest)
    
    mae2=out_mae(user3, test3)
    
    print("Mean Absolute Error for Movie Lens Data Set: ", mae2)
    #1.35 , 1.31, 1.26, 1.26, 1.26
   
    
if __name__ == "__main__":
    main()


#####Plotting MAE by k 
import matplotlib.pyplot as plt
x=[1,5,10,50,100]
lst3=[1.33,1.02, 1.02, .87, .86]
plt.plot(x,lst3, "bo")
plt.ylabel("Yelp MAE")
plt.xlabel("Number of Ks")

lst4=[1.35 , 1.31, 1.26, 1.26, 1.26]
plt.plot(x,lst4, "bo")
plt.ylabel("Movielens MAE")
plt.xlabel("Number of Ks")