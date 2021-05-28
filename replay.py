# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:05:38 2021

@author: xChrisYe
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import random

class rePlayBaseModel(object):
    
    def __init__(self,data,testTime =25000):
        
        random.seed(35587740)
        
        # raw data
        self.data = data

        # user list
        self.users  = self.data['userId'].unique()
        self.n_users  = len(self.users)
        
        # movie list
        self.movies  = self.data['movieId'].unique()
        self.n_movies  = len(self.movies)

        # np which is going to record how many time for a specific movie selected
        self.n_selected= np.zeros(self.n_movies) 
        # np which is going to record how is this reward for a specific movie  selected
        self.distribution= np.zeros(self.n_movies)
        
        # the number of replay in one iteration
        self.testTime = testTime 
        # algorithm name
        self.name = 'random' 
        
        
        
    def replay(self):
        
        results= []
        
        # 20 iterations for calculating the mean trend for a specific algorithm
        for iteration in tqdm(range(0,15)): 
            
            self.n_selected= np.zeros(self.n_movies)
            self.distribution= np.zeros(self.n_movies)
            if self.name=='AB_TEST':
                self.isTest = True
            if self.name =='thompsonSample':
                self.betaParam = pd.DataFrame(data=1, columns=['alpha', 'beta'],index = range(0,self.n_movies))
            totalReward = 0
            
            # number of testTime to replay and find the movie-user match in a iteration
            for testIndex in range(0, self.testTime):
                #if testIndex%1000 == 0:
                #   print(testIndex)
                find = False
                matchRows = pd.DataFrame()
                while not find:
                    
                    #  random a user_index
                    rand_user_index = np.random.randint(self.n_users)
                    user_id =  self.users[rand_user_index]
                    
                    # select movie according to current algorithm
                    movie_index = self.select_movie_index(testIndex)
                    movie_id =  self.movies[movie_index]
                    
                    # to find if the movie match the user 
                    matchRows = self.data[(self.data['movieId'] == movie_id ) & (self.data['userId'] == user_id)]
                    if matchRows.shape[0]>0 :
                        find = True
                        
                if matchRows.shape[0]>0:
                    reward = matchRows.iat[0,4]
                    self.updataDistribution(testIndex,movie_index,reward)
                    
                record = {}
                totalReward+=reward
                record['iteration'] = iteration
                record['index'] = testIndex
                record['userId'] = user_id
                record['movieId'] = movie_id
                record['reward'] = reward
                record['total_reward'] = totalReward
                record['fraction_relevant'] = totalReward * 1. / (testIndex+1)
                results.append(record)
            #print(self.distribution)
        return results
    
    #function describe how an algorithm select a movie
    def select_movie_index(self,testIndex):
        
        #random select
        return np.random.randint(self.n_movies)
        
    def updataDistribution(self,testIndex,movie_id,reward):
        
        # record the number of movie selected
        self.n_selected[movie_id] +=1 
        
        # record avaerge reward for a movie
        self.distribution[movie_id] += (1/ self.n_selected[movie_id]) * (reward - self.distribution[movie_id])
         
# AB_TEST algorithm    
class AB_TEST(rePlayBaseModel):
    
    def __init__(self,data,n_test):
        super().__init__(data)
        self.isTest = True
        self.bestMovie_index =None
        self.n_test = n_test
        self.name = 'AB_TEST'
        
        
    def select_movie_index(self,testIndex):
        
        # Test part select randomly
        if self.isTest:
            return np.random.randint(self.n_movies)
        # use the best_movie after testing
        else:
            return self.bestMovie_index
        
    def updataDistribution (self,testIndex,movie_id,reward):
        
        # Test the algorithm for testTime times, and then use the best answer so far to select in the later replays
        if self.isTest:
            super().updataDistribution(testIndex,movie_id, reward)
            if  testIndex == self.n_test :
                self.isTest = False
                self.bestMovie_index = np.argmax(self.distribution) 
                #print(self.bestMovie_index)
                #print(self.distribution)
                
# epsilonGreedy algorithm 
class epsilonGreddy(rePlayBaseModel):
    
    def __init__(self,data,epsilon=0.1):
        super().__init__(data)
        self.epsilon = epsilon # set epsilon parameter
        self.name = 'epsilonGreddy'
        
    def select_movie_index(self,testIndex):
        
        random_n = np.random.uniform()
        
        # randomly select movie when less than epsilon, 
        # otherwise select the best reward movie according the distribution so far
        if random_n < self.epsilon:
            return(np.random.randint(self.n_movies))
        else:
            return np.argmax(self.distribution)

# UCB algorithm 
class UCB(rePlayBaseModel):
    def __init__(self, data):
        super().__init__(data)
        self.name = 'UCB'
    def select_movie_index(self,testIndex):
        max_upper_bound = 0
        
        # calculate the delta and upper bound for each movie
        for movie in range(self.n_movies):
            if self.n_selected[movie]>0:
                delta_i = np.sqrt(2*np.log(testIndex+1)/self.n_selected[movie])
                upper_bound = self.distribution[movie]+delta_i
            else:
                upper_bound = 1e400
            if upper_bound> max_upper_bound:
                max_upper_bound = upper_bound
                best_arm = movie
        return best_arm
                
              
# thompsonSample algorithm 
class thompsonSample(rePlayBaseModel):
    
    def __init__(self,data):
        super().__init__(data)
        
        # dataFrame for alpha,beta to record the number of a movie get negative and positive reward.
        self.betaParam = pd.DataFrame(data=1, columns=['alpha', 'beta'],index = range(0,self.n_movies))
        self.name = 'thompsonSample'
        
    def select_movie_index(self,testIndex):
        
        # get the max index in Beta distribution for current reward situation as the choice.
        return (np.argmax([np.random.beta(self.betaParam.alpha,self.betaParam.beta) ])) 
    
    def updataDistribution (self,testIndex,movie_id,reward):
        if reward > 0:
            self.betaParam.loc[movie_id].alpha+=1
        else:
            self.betaParam.loc[movie_id].beta+=1
            
        