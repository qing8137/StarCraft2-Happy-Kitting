import random
import math
import os

import numpy as np
import pandas as pd
import time

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_UNIT_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points.index
_UNIT_HIT_POINTS_RATIO = features.SCREEN_FEATURES.unit_hit_points_ratio.index
_UNIT_DENSITY_AA = features.SCREEN_FEATURES.unit_density_aa.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index


_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4

_NOT_QUEUED = [0]
_QUEUED = [1]


ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK_UP = 'attackup'
ACTION_ATTACK_DOWN = 'attackdown'
ACTION_ATTACK_LEFT = 'attackleft'
ACTION_ATTACK_RIGHT = 'attackright'

ACTION_ATTACK_UP = 'attackup'
ACTION_ATTACK_DOWN = 'attackdown'
ACTION_ATTACK_LEFT = 'attackleft'
ACTION_ATTACK_RIGHT = 'attackright'


ACTION_MOVE_UP = 'moveup'
ACTION_MOVE_DOWN = 'movedown'
ACTION_MOVE_LEFT = 'moveleft'
ACTION_MOVE_RIGHT = 'moveright'
ACTION_MOVE_UP_LEFT = 'moveupleft'
ACTION_MOVE_UP_RIGHT = 'moveupright'
ACTION_MOVE_DOWN_LEFT = 'movedownleft'
ACTION_MOVE_DOWN_RIGHT = 'movedownright'
ACTION_SELECT_ARMY_1 = 'selectarmy1'
ACTION_SELECT_ARMY_2 = 'selectarmy2'
ACTION_SELECT_ARMY_3 = 'selectarmy3'
ACTION_SELECT_UNIT_1 = 'selectunit1'
ACTION_SELECT_UNIT_2 = 'selectunit2'
ACTION_SELECT_UNIT_3 = 'selectunit3'

DATA_FILE = 'refined_agent_data'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_ARMY,
    ACTION_ATTACK_UP,
    ACTION_ATTACK_DOWN,
    ACTION_ATTACK_LEFT,
    ACTION_ATTACK_RIGHT,

    ACTION_MOVE_UP,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    ACTION_MOVE_UP_LEFT,
    ACTION_MOVE_UP_RIGHT,
    ACTION_MOVE_DOWN_LEFT,
    ACTION_MOVE_DOWN_RIGHT,
    #ACTION_SELECT_ARMY_1,
    #ACTION_SELECT_ARMY_2,
    #ACTION_SELECT_ARMY_3,
    ACTION_SELECT_UNIT_1,
    ACTION_SELECT_UNIT_2,
    ACTION_SELECT_UNIT_3
]

KILL_UNIT_REWARD = 100
LOSS_UNIT_REWARD = -0.5


# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.5, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.disallowed_actions = {}

    def choose_action(self, observation, excluded_actions=[]):
        self.check_state_exist(observation)
        
        self.disallowed_actions[observation] = excluded_actions
        
        state_action = self.q_table.ix[observation, :]
        
        for excluded_action in excluded_actions:
            del state_action[excluded_action]

        if np.random.uniform() < self.epsilon:
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            action = state_action.idxmax()
        else:
            action = np.random.choice(state_action.index)
            
        return action

    def learn(self, s, a, r, s_):
        if s == s_:
            return
        
        self.check_state_exist(s_)
        self.check_state_exist(s)
        
        q_predict = self.q_table.ix[s, a]
        
        s_rewards = self.q_table.ix[s_, :]
        
        if s_ in self.disallowed_actions:
            for excluded_action in self.disallowed_actions[s_]:
                del s_rewards[excluded_action]
        
        if s_ != 'terminal':
            q_target = r + self.gamma * s_rewards.max()
        else:
            q_target = r  # next state is terminal
            
        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

class SmartAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SmartAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        self.previous_killed_unit_score = 0
        self.previous_lost_hp_score = 0

        self.previous_action = None
        self.previous_state = None
        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def step(self, obs,):
        super(SmartAgent, self).step(obs)

        # time.sleep(0.2)

        player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        # unit_type = obs.observation['screen'][_UNIT_TYPE]
        units_count = obs.observation['multi_select'].shape[0]

        units = []
        hp = []

        for i in range(units_count):
            units.append(obs.observation['multi_select'][i])
            hp.append(obs.observation['multi_select'][i][2])

        army_health_score = sum(hp)

        killed_unit_score = obs.observation['score_cumulative'][5]
        lost_hp_score = (45*3 - sum(hp))/(45*3)
        current_state = [
            enemy_y,
            enemy_x,
            player_y,
            player_x,
            army_health_score
        ]
        

 


        # print(current_state, self.previous_state, self.steps)
        # self.print_data(unit_hit_points_ratio)

        if self.previous_action is not None:
            reward = 0

            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD

            if lost_hp_score > self.previous_lost_hp_score:
                reward -= lost_hp_score

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))
            # print(self.reward, self.steps)
            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]

        self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
        
        self.previous_killed_unit_score = killed_unit_score
        self.previous_lost_hp_score = lost_hp_score
        self.previous_state = current_state
        self.previous_action = rl_action

        return self.perform_action(obs, smart_action, player_x, player_y)


    def perform_action(self, obs, action, xloc, yloc):
        if action == ACTION_DO_NOTHING:
            return actions.FunctionCall(_NO_OP, [])

        elif action == ACTION_SELECT_ARMY:
           if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        elif action == ACTION_SELECT_UNIT_1:
            if _SELECT_UNIT in obs.observation['available_actions']:
                if len(xloc) >= 1 and len(yloc) >= 1:
                    #print(1)
                    return actions.FunctionCall(_SELECT_UNIT, [_NOT_QUEUED, [0]])

        elif action == ACTION_SELECT_UNIT_2:
            if _SELECT_UNIT in obs.observation['available_actions']:
                if len(xloc) >= 2 and len(yloc) >= 2:
                    #print(2)
                    return actions.FunctionCall(_SELECT_UNIT, [_NOT_QUEUED, [1]])

        elif action == ACTION_SELECT_UNIT_3:
            if _SELECT_UNIT in obs.observation['available_actions']:
                if len(xloc) >= 3 and len(yloc) >= 3:
                    #print(3)
                    return actions.FunctionCall(_SELECT_UNIT, [_NOT_QUEUED, [2]])

        elif action == ACTION_ATTACK_UP:
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [0, 36]])

        elif action == ACTION_ATTACK_DOWN:
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [60, 36]])

        elif action == ACTION_ATTACK_LEFT:
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [32, 0]])

        elif action == ACTION_ATTACK_RIGHT:
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [32, 60]])

        elif action == ACTION_MOVE_UP:
            if _MOVE_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [0, 83]])

        elif action == ACTION_MOVE_DOWN:
            if _MOVE_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [83, 42]])

        elif action == ACTION_MOVE_LEFT:
            if _MOVE_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [42, 0]])

        elif action == ACTION_MOVE_RIGHT:
            if _MOVE_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [42, 83]])

        elif action == ACTION_MOVE_UP_LEFT:
            if _MOVE_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [0, 0]])

        elif action == ACTION_MOVE_UP_RIGHT:
            if _MOVE_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [0, 83]])

        elif action == ACTION_MOVE_DOWN_LEFT:
            if _MOVE_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [83, 0]])

        elif action == ACTION_MOVE_DOWN_RIGHT:
            if _MOVE_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [83, 83]])

        return actions.FunctionCall(_NO_OP, [])

    # def print_data(self, unit_hit_points_ratio):
    #     print(unit_hit_points_ratio)

'''     
        elif action == ACTION_SELECT_ARMY_1:
            if _SELECT_POINT in obs.observation['available_actions']:
                if len(xloc) >= 1 and len(yloc) >= 1:
                    #print(1, xloc[0], yloc[0])
                    return actions.FunctionCall(_SELECT_POINT, [[1], [xloc[0], yloc[0]]])

        elif action == ACTION_SELECT_ARMY_2:
            if _SELECT_POINT in obs.observation['available_actions']:
                if len(xloc) >= 2 and len(yloc) >= 2:
                    #print(2, xloc[1], yloc[1])
                    return actions.FunctionCall(_SELECT_POINT, [[1], [xloc[1], yloc[1]]])

        elif action == ACTION_SELECT_ARMY_3:
            if _SELECT_POINT in obs.observation['available_actions']:
                if len(xloc) >= 3 and len(yloc) >= 3:
                   # print(3, xloc[2], yloc[2])
                    return actions.FunctionCall(_SELECT_POINT, [[1], [xloc[2], yloc[2]]])
'''