import random as rdm

class Policy:
    RIGHT = (1, 0)
    LEFT = (-1, 0)
    DOWN = (0, 1)
    UP = (0, -1)
    ACTION_SPACE = [RIGHT, LEFT, DOWN, UP]

    def __init__(self, policy, epsilon=0):
        self.policy = policy
        self.epsilon = epsilon

    def action(self, state):
        x, y = state
        w = rdm.uniform(0, 1)
        if w > self.epsilon:
            return self.policy[x][y]
        else:
            w2 = rdm.uniform(0, 1)
            if w2 < 0.25:
                action = self.UP
            elif w2 < 0.5:
                action = self.RIGHT
            elif w2 < 0.75:
                action = self.DOWN
            else:
                action = self.LEFT
            return action



