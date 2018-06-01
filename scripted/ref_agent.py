import numpy as np
import math
import time
import random
import matplotlib.pyplot as plt
from pysc2.lib import actions

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4

_UNIT_ALLIANCE = 1
_UNIT_HEALTH = 2
_UNIT_SHIELD = 3
_UNIT_X = 12
_UNIT_Y = 13
_UNIT_IS_SELECTED = 17

_NOT_QUEUED = [0]
_QUEUED = [1]

ATTACK_TARGET = 'attacktarget'
MOVE = 'move'

smart_actions = [
    ATTACK_TARGET,
    MOVE
]

# Change this if using a different map
# Currently running HK2V1
DEFAULT_ENEMY_COUNT = 1
DEFAULT_PLAYER_COUNT = 2

ENEMY_MAX_HP = 150
PLAYER_MAX_HP = 60


class SmartAgent(object):
    def __init__(self):
        # from the origin base.agent
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None
        self.enemy_hp = []
        self.acted = False

    def step(self, obs):
        time.sleep(1/60*0.5)
        # from the origin base.agent
        self.steps += 1
        self.reward += obs.reward

        self.counter += 1
        enemy_loc, player_loc, distance, selected, enemy_count = self.extract_features(obs)

        if (selected[0] == 0 and self.acted == True) or (selected[0] == 1 and selected[1] == 1):
            self.acted = False
            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, player_loc[0]])
        elif selected[1] == 0 and self.acted == True:
            self.acted = False
            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, player_loc[1]])
        
        if selected[1] == 1:
            selIndex = 1
        else:
            selIndex = 0

        self.acted = True

        if distance > 8:
            if _ATTACK_SCREEN in obs.observation["available_actions"]:
                if enemy_count >= 1:
                    return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, enemy_loc[0]])  # x,y => col,row
        else:
            p = self.get_flee_point(player_loc[selIndex], enemy_loc[0]);
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [p[0], p[1]]])

        return actions.FunctionCall(_NO_OP, [])

    # extract all the desired features as inputs for the DQN
    def extract_features(self, obs):
        var = obs.observation['feature_units']
        # get units' location and distance
        enemy, player = [], []

        # get health
        enemy_hp, player_hp = [], []

        # record the selected army
        is_selected = []

        # unit_count
        enemy_unit_count, player_unit_count = 0, 0

        for i in range(0, var.shape[0]):
            if var[i][_UNIT_ALLIANCE] == _PLAYER_HOSTILE:
                enemy.append((var[i][_UNIT_X], var[i][_UNIT_Y]))
                enemy_hp.append(var[i][_UNIT_HEALTH] + var[i][_UNIT_SHIELD ])
                enemy_unit_count += 1
            else:
                player.append((var[i][_UNIT_X], var[i][_UNIT_Y]))
                player_hp.append(var[i][_UNIT_HEALTH])
                is_selected.append(var[i][_UNIT_IS_SELECTED])
                player_unit_count += 1

        # append if necessary so that maintains fixed length for current state
        for i in range(player_unit_count, DEFAULT_PLAYER_COUNT):
            player.append((-1, -1))
            player_hp.append(0)
            is_selected.append(-1)

        for i in range(enemy_unit_count, DEFAULT_ENEMY_COUNT):
            enemy.append((-1, -1))
            enemy_hp.append(0)

        # get distance
        
        min_distance = [83 for x in range(DEFAULT_PLAYER_COUNT)]

        for i in range(0, player_unit_count):
            for j in range(0, enemy_unit_count):
                distance = int(math.sqrt((player[i][0] - enemy[j][0]) ** 2 + (
                        player[i][1] - enemy[j][1]) ** 2))

                if distance < min_distance[i]:
                    min_distance[i] = distance
        
        selIndex = 0

        if is_selected[0] == 1:
            selIndex = 0
        if is_selected[1] == 1:
            selIndex = 1

        sel_distance = []
        sel_distance.append(math.sqrt((player[selIndex][0] - enemy[0][0])**2 + (player[selIndex][1] - enemy[0][1])**2))

        self.enemy_hp.append(sum(enemy_hp))

        return enemy, player, sel_distance[0], is_selected, enemy_unit_count

    # make the desired action calculated by DQN
    def perform_action(self, obs, action, unit_locs, enemy_locs, selected, player_count, enemy_count, distance, player_hp):
        index = -1
        
        for i in range(0, DEFAULT_PLAYER_COUNT):
            if selected[i] == 1:
                index = i

        x = unit_locs[index][0]
        y = unit_locs[index][1]

        if action == ATTACK_TARGET:
            if _ATTACK_SCREEN in obs.observation["available_actions"]:
                if enemy_count >= 1:
                    return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, enemy_locs[0]])  # x,y => col,row

        elif action == MOVE:
            p = self.get_flee_point(unit_locs[index], enemy_locs[0]);
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [p[0], p[1]]])

        return actions.FunctionCall(_NO_OP, [])

    def get_flee_point(self, unit_loc, enemy_loc):
        flee_point = []

        dx = (enemy_loc[0] - unit_loc[0])
        dy = (enemy_loc[1] - unit_loc[1])
        
        vecLen = math.sqrt(dx*dx + dy*dy)
        
        dx = dx / vecLen * 12
        dy = dy / vecLen * 12


        flee_x = unit_loc[0] - dx
        flee_y = unit_loc[1] - dy


        if flee_x < 0:
            flee_x = 0
        if flee_x > 83:
            flee_x = 83
        if flee_y < 0:
            flee_y = 0
        if flee_y > 83:
            flee_y = 83

        x = unit_loc[0]
        y = unit_loc[1]
        
        # reaching corners
        if x < 8 and y < 8:
            print("TOP_LEFT")
            flee_x = 3
            flee_y = 79
        elif x < 8 and y > 54:
            print("BOTTOM_LEFT")
            flee_x = 79
            flee_y = 79
        elif x > 75 and y < 8:
            print("TOP_RIGHT")
            flee_x = 3
            flee_y = 3
        elif x > 75 and y > 54:
            print("BOTTOM_RIGHT")
            flee_x = 79
            flee_y = 3
        elif x < 8 and dy < 5:
            flee_x = 3
            flee_y = 79
        elif x > 75 and dy < 5:
            flee_x = 79
            flee_y = 79
        elif y < 8 and dx < 5:
            flee_x = 3
            flee_y = 3
        elif y > 75 and dx < 5:
            flee_x = 79
            flee_y = 79

        flee_point.append(flee_x)
        flee_point.append(flee_y)

        return flee_point

    def plot_player_hp(self, path, save):
        plt.plot(np.arange(len(self.player_hp)), self.player_hp)
        plt.ylabel('player hp')
        plt.xlabel('training steps')
        if save:
            plt.savefig(path + '/player_hp.png')
        plt.show()

    def plot_enemy_hp(self, path, save):
        plt.plot(np.arange(len(self.enemy_hp)), self.enemy_hp)
        print("Agent winning rate: " + str(self.enemy_hp.count(0) * 100 / (self.episodes - 1)) + "%")
        plt.ylabel('enemy hp')
        plt.xlabel('training steps')
        if save:
            plt.savefig(path + '/enemy_hp.png')
        plt.show()

    # from the origin base.agent
    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    # from the origin base.agent
    def reset(self):
        self.episodes += 1
        # added instead of original
        self.fighting = False
        self.counter = 0
        self.previous_player_hp = [PLAYER_MAX_HP, PLAYER_MAX_HP]
        self.previous_enemy_hp = [ENEMY_MAX_HP]



