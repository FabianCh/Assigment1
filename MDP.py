import numpy as np


class MDP:
    RIGHT = (1, 0)
    LEFT = (-1, 0)
    DOWN = (0, 1)
    UP = (0, -1)
    ACTION_SPACE = [RIGHT, LEFT, DOWN, UP]

    def __init__(self, gamma=0.99):
        self.N1 = {}
        self.N2 = {}
        self.R = {}
        self.gamma = gamma
        self.xmax = 0
        self.ymax = 0
        self.B = 0

    def estimation(self, ht):
        xbool = False
        ybool = False
        for i in range((len(ht)-1)//3):
            hx = ht[i*3]
            hu = ht[i*3 + 1]
            hr = ht[i*3 + 2]
            hx2 = ht[i*3 + 3]

            if (hx, hu) in self.N1.keys():
                self.N1[(hx, hu)] += 1
                self.R[(hx, hu)] += hr
            else:
                self.N1[(hx, hu)] = 1
                self.R[(hx, hu)] = hr

            if (hx, hu, hx2) in self.N2:
                self.N2[(hx, hu, hx2)] += 1
            else:
                self.N2[(hx, hu, hx2)] = 1

            if hr > self.B:
                self.B = hr

            if hx[0] > self.xmax:
                self.xmax = hx[0]
                xbool = True

            if hx[1] > self.ymax:
                self.ymax = hx[1]
                ybool = True

        if xbool:
            self.xmax += 1
        if ybool:
            self.ymax += 1

    def r(self, x, u):
        if (x, u) in self.R.keys():
            return self.R[(x, u)] / self.N1[(x, u)]
        else:
            return 0

    def p(self, x, u, x2):
        if (x, u, x2) in self.N2.keys():
            return self.N2[(x, u, x2)] / self.N1[(x, u)]
        else:
            return 0

    def MatrixJN(self, policy, N):
        # method to return the list of Matrix of Expected value after N turn with a policy in a domain
        L = [np.array([[0. for k in range(self.xmax)] for l in range(self.ymax)])]
        for h in range(1, N+1):
            L.append(np.array([[0. for k in range(self.xmax)] for l in range(self.ymax)]))
            for i in range(self.xmax):
                for j in range(self.ymax):
                    action = policy.action([i, j])
                    L[-1][j][i] = self.r((i, j), action)
                    for key in self.N2.keys():
                        if key[0:2] == ((i, j), action):
                            L[-1][j][i] += self.p(key[0], key[1], key[2]) * L[-2][key[2][1]][key[2][0]]
        return L

    def MatrixJ(self, policy):
        N = 0
        while ((self.gamma ** N) * self.B) / (1 - self.gamma) > 0.01:
            N += 1
        return self.MatrixJN(policy, N)[-1]

    def MatrixQN(self, N):
        L = [np.array(
            [[{MDP.UP: 0, MDP.RIGHT: 0, MDP.DOWN: 0, MDP.LEFT: 0} for l in range(self.xmax)] for m in
             range(self.ymax)])]
        for h in range(1, N):
            L.append(np.array(
                [[{MDP.UP: 0, MDP.RIGHT: 0, MDP.DOWN: 0, MDP.LEFT: 0} for l in range(self.xmax)] for m in
                 range(self.ymax)]))
            for i in range(self.xmax):
                for j in range(self.ymax):
                    for k in MDP.ACTION_SPACE:
                        L[-1][j][i][k] = self.r((i, j), k)
                        for key in self.N2.keys():
                            if key[0:2] == ((i, j), k):
                                L[-1][j][i][k] += self.p(key[0], key[1], key[2]) * \
                                                  max(L[-2][min(max(j + k[1], 0), self.xmax - 1)]
                                                      [min(max(i + k[0], 0), self.ymax - 1)].values())
        return L

    def MatrixQ(self):
        N = 0
        while (2 * (self.gamma ** N) * self.B) / (1 - self.gamma) ** 2 > 0.01:
            N += 1
        return self.MatrixQN(N)[-1]

    def mustar(self):
        Q = self.MatrixQ()
        mu = np.zeros_like(Q)
        for x in range(self.xmax):
            for y in range(self.ymax):
                mu[x, y] = max(Q[x, y], key=Q[x, y].get)
        return mu

