import numpy as np

class Qlearning:
    RIGHT = (1, 0)
    LEFT = (-1, 0)
    DOWN = (0, 1)
    UP = (0, -1)
    ACTION_SPACE = [RIGHT, LEFT, DOWN, UP]

    def __init__(self, domain):
        self.gamma = domain.gamma
        self.xmax = domain.n
        self.ymax = domain.m
        self.Q = np.array(
            [[{self.UP: 0, self.RIGHT: 0, self.DOWN: 0, self.LEFT: 0} for l in range(self.xmax)] for m in
             range(self.ymax)])
        self.alpha = 0.05

    def expectedmoves(self, state,  action):
        if action in self.ACTION_SPACE:
            return (min(max(state[0] + action[0], 0), self.xmax - 1),
                    min(max(state[1] + action[1], 0), self.ymax - 1))

    def estimation(self, ht):
        for i in range((len(ht)-1)//3):
            hx = ht[3*i]
            hu = ht[3*i + 1]
            hr = ht[3*i + 2]
            hx2 = ht[3*i + 3]

            aux = 0
            for action in self.ACTION_SPACE:
                if self.Q[hx2[0]][hx2[1]][action] > aux :
                    aux = self.Q[hx2[0]][hx2[1]][action]
            self.Q[hx[0]][hx[1]][hu] = (1-self.alpha)*self.Q[hx[0]][hx[1]][hu] + self.alpha * (hr + self.gamma * aux)


    def mustar(self):
        mu = np.zeros_like(Q)
        for x in range(self.xmax):
            for y in range(self.ymax):
                mu[x, y] = max(self.Q[x, y], key=self.Q[x, y].get)
        return mu




