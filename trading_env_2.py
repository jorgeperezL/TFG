from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import pandas as pd
import tempfile
import os

from tf_agents.environments import py_environment
#from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from sklearn.preprocessing import MinMaxScaler

class TradingEnv(py_environment.PyEnvironment):
    
    def __update_state(self):        
        self.state = self.price_episode[self.rounds:self.days+self.rounds,1:].astype(np.float32)
        
    def __select_price(self,idx):
        return self.price_episode[idx-2:idx,0]
        
    def __select_episode(self):
        ind = self.index % self.data.shape[0]
        self.index += 1
        return self.data[ind]

    def __init__(self,df,verbose = False, random = False):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        
        self._observation_spec = array_spec.BoundedArraySpec(shape=(10,2), dtype=np.float32, name='observation')
        
        self._state = 0
        self._episode_ended = False
        
        self.data = df
        
        self.index = 0
        
        # dias
        self.rounds = 0
        self.days = 10
        self.max_days = 10
        
        # episode
        self.price_episode = self.__select_episode()
        
        # current state 
        self.__update_state()
                
        # reward collected
        self.collected_reward = 0
        
        self.episodes_reward = []
        
        #render
        self.verbose = verbose
        
        #episode_final
        self._episode_ended = False
        

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        
        #episode_final
        self._episode_ended = False
        
        # dias
        self.rounds = 0
        
        # episode
        self.price_episode = self.__select_episode()
        
        # current state 
        self.__update_state()
                
        # reward collected
        self.collected_reward = 0
        
        return ts.restart(self.state)

    def _step(self, action):
        
        
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        
        rw = 0
        self.rounds += 1
        price = self.__select_price(self.rounds+self.days)
        
        
        if action == 0:
            rw = price[1]
            self.collected_reward += rw
            
        #if self.collected_reward < -0.1:
        #   self.collected_reward = -2
        #   self._episode_ended = True
           
        if action == 1:
           self._episode_ended = True     
        
        if self.rounds == self.max_days:
           self._episode_ended = True
 
        if self._episode_ended == True:
           self.episodes_reward.append(self.collected_reward)

        self.__update_state()           
                               
        if self.verbose:
           self.render(action,rw,price)
            
        if self._episode_ended:
            return ts.termination(self.state, reward=self.collected_reward)
        else:
            return ts.transition(self.state, reward=self.collected_reward,discount=0.0)        
        
        
    def render(self, action, rw,price):
        print(f"Precio : {price}")
        print(f"Round : {self.rounds}\nAccion : {action}\nReward Received: {rw}")
        #print(f"Posicion: {self.position}")
        print(f"Total Reward : {self.collected_reward}")
        print("=============================================================================")
