import cv2
import numpy as np
import random
import time
import copy

class Env:
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.migong = []
        self.x1, self.y1 = 0, 0
        self.end_game = 0
        self.display1 = None
        self.last_action = None
        self.total_grid = len(self.start_env())*len(self.start_env()[0])*3

    # 建立虚拟环境
    def start_env(self):

        """
        self.migong = [[1, 0, 0, 0, 0],
                       [0, 0, 0, 3, 0],
                       [0, 0, 0, 0, 0],
                       [0, 3, 0, 0, 0],
                       [0, 0, 0, 0, 2]]
        """
        self.migong = [[0, 3, 0, 2, 0],
                       [0, 3, 0, 3, 3],
                       [0, 0, 0, 0, 0],
                       [0, 3, 0, 3, 0],
                       [0, 3, 3, 3, 0],
                       [3, 0, 0, 0, 0],
                       [0, 0, 3, 0, 0]]

        self.height = len(self.migong)
        self.width = len(self.migong[0])
        self.x1, self.y1 = 0, 6
        self.migong[self.y1][self.x1] = 1
        self.raw_mg = copy.deepcopy(self.migong)
        self.end_game = 0

        return self.migong

    def display(self):

        # cv2.imwrite('1.png', self.display1)
        self.display1 = np.ones((60*self.height, 60*self.width, 3), dtype=np.uint8)
        self.display1 = np.array(np.where(self.display1 == 1, 255, 0), dtype=np.uint8)

        for i in range(self.height):
            cv2.line(self.display1, (i * 60, 0), (i * 60, 60*self.height), (0, 0, 0), 1)
        for j in range(self.height):
            cv2.line(self.display1, (0, j * 60), (60*self.width, j * 60), (0, 0, 0), 1)

        for y in range(self.height):  # 4
            for x in range(self.width):  # 5
                if self.migong[y][x] == 1:
                    cv2.circle(self.display1, (x * 60 + 30, y * 60 + 30), 25, (255, 0, 0), -1)
                if self.migong[y][x] == 2:
                    cv2.circle(self.display1, (x * 60 + 30, y * 60 + 30), 25, (0, 255, 0), -1)
                if self.migong[y][x] == 3:
                    cv2.circle(self.display1, (x * 60 + 30, y * 60 + 30), 25, (0, 0, 255), -1)

        cv2.imshow('1', self.display1)
        cv2.waitKey(1)

    def assign_reward(self, r, his_step):
        if his_step[self.y1][self.x1] == 1:
            r -= 0.5
        else: r += 1
        if self.raw_mg[self.y1][self.x1] == 3:
            self.end_game = 1
            r += -2
        elif self.raw_mg[self.y1][self.x1] == 2:
            self.end_game = 2
            r += 15
        return r

    def step(self, action, his_step):
        r = 0
        # ['u'0, 'd'1, 'l'2, 'r'3]
        # if (self.last_action, action) in [(0, 1), (1, 0), (2, 3), (3,2)]:
        #    r += -0.5
        if action == 0:
            if self.y1 == 0:
                r += -1
            else:
                self.migong[self.y1][self.x1] = 0
                self.migong[self.y1 - 1][self.x1] = 1
                self.y1 -= 1
                r = self.assign_reward(r, his_step)

        if action == 1:
            if self.y1 == self.height-1:
                r += -1
            else:
                self.migong[self.y1][self.x1] = 0
                self.migong[self.y1 + 1][self.x1] = 1
                self.y1 += 1
                r = self.assign_reward(r, his_step)
        if action == 2:
            if self.x1 == 0:
                r += -1
            else:
                self.migong[self.y1][self.x1] = 0
                self.migong[self.y1][self.x1 - 1] = 1
                self.x1 -= 1
                r = self.assign_reward(r, his_step)
        if action == 3:
            if self.x1 == self.width-1:
                r += -1
            else:
                self.migong[self.y1][self.x1] = 0
                self.migong[self.y1][self.x1 + 1] = 1
                self.x1 += 1
                r = self.assign_reward(r, his_step)
        # return self.migong
        return self.end_game, r, self.migong
