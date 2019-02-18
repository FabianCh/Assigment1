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

    def estimation(self, ht):
        for i in range((len(ht)-1)//3):
            hx = ht[i]
            hu = ht[i + 1]
            hr = ht[i + 2]
            hx2 = ht[i + 3]

            if (hx, hu) in self.N1:
                self.N1[(hx, hu)] += 1
                self.R[(hx, hu)] += hr
            else:
                self.N1[(hx, hu)] = 1
                self.R = hr

            if (hx, hu, hx2) in self.N2:
                self.N1[(hx, hu, hx2)] += 1
            else:
                self.N1[(hx, hu, hx2)] = 1

    def r(self, x, u):
        return self.R[(x, u)] / self.N1[(x, u)]

    def p(self, x, u, x2):
        return self.N2[(x, u, x2)] / self.N1[(x, u)]

    def MatrixQN(self, N):
        L = [np.array(
            [[{MDP.UP: 0, MDP.RIGHT: 0, MDP.DOWN: 0, MDP.LEFT: 0} for l in range(domain.n)] for m in
             range(domain.m)])]
        for h in range(1, N):
            L.append(np.array(
                [[{MDP.UP: 0, MDP.RIGHT: 0, MDP.DOWN: 0, MDP.LEFT: 0} for l in range(domain.n)] for m in
                 range(domain.m)]))
            for i in range(domain.n):
                for j in range(domain.m):
                    for k in MDP.ACTION_SPACE:
                        L[-1][j][i][k] = domain.reward([i, j], k)
                        L[-1][j][i] += domain.gamma * (1 - domain.beta) * max(
                            L[-2][min(max(j + k[1], 0), domain.n - 1)][min(max(i + k[0], 0), domain.m - 1)].value)
                        L[-1][j][i] += domain.gamma * domain.beta * max(L[-2][0][0].value)
        return L
