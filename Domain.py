import numpy as np
import random as rdm
import Policy


class Domain:
    # Class to create a determinist domain (Beta=0) or a stochastique domain (beta!=0)
    RIGHT = (1, 0)
    LEFT = (-1, 0)
    DOWN = (0, 1)
    UP = (0, -1)
    ACTION_SPACE = [RIGHT, LEFT, DOWN, UP]

    def __init__(self, x, y, board, beta=0):
        self.board = board
        self.n = len(board[0])
        self.m = len(board)
        self.state = [x, y]
        self.B = np.max(self.board)
        self.gamma = 0.99
        self.beta = beta
        self.w = rdm.uniform(0, 1)

    def expectedmoves(self, action):
        if action in self.ACTION_SPACE:
            return [min(max(self.state[0] + action[0], 0), self.n - 1),
                    min(max(self.state[1] + action[1], 0), self.m - 1)]

    def moves(self, action):
        if action in self.ACTION_SPACE:
            if self.w <= 1 - self.beta:
                self.state[0] = min(max(self.state[0] + action[0], 0), self.n - 1)
                self.state[1] = min(max(self.state[1] + action[1], 0), self.m - 1)
            else:
                self.state[0] = 0
                self.state[1] = 0
            self.w = rdm.uniform(0, 1)

    def reward(self, state, action):
        if action in self.ACTION_SPACE:
            xa, ya = action
            xr = min(max(state[0] + xa, 0), self.n - 1)
            yr = min(max(state[1] + ya, 0), self.m - 1)
            return (1 - self.beta)*self.g(xr, yr) + self.beta * self.g(0, 0)

    def g(self, x, y):
        return self.board[y][x]

    def p(self, x, u, x2):
        res = 0
        if self.expectedmoves(u) == x2:
            res += 1 - self.beta
        elif x2 == [0, 0]:
            res += self.beta
        return res

    def generationtrajectoire(self, taille):
        self.state = [2, 2]
        ht = [tuple(self.state)]
        for i in range(taille):
            w = rdm.uniform(0, 1)
            if w < 0.25:
                action = self.UP
            elif w < 0.5:
                action = self.RIGHT
            elif w < 0.75:
                action = self.DOWN
            else:
                action = self.LEFT
            self.moves(action)
            reward = self.g(self.state[0], self.state[1])
            ht += [action, reward, tuple(self.state)]
        return ht

    def generationtrajectoirepolicy(self, taille, policy):
        self.state = [2, 2]
        ht = [tuple(self.state)]
        for i in range(taille):
            action = policy.action(self.state)
            self.moves(action)
            reward = self.g(self.state[0], self.state[1])
            ht += [action, reward, tuple(self.state)]
        return ht


def JN(domain: Domain, policy: Policy.Policy, N):
    # method to return the Expected value after N turn with a policy in a domain
    if N == 0:
        return 0
    else:
        R = domain.reward(domain.state, policy.action(domain.state))
        domain.moves(policy.action(domain.state))
        return R + domain.gamma * JN(domain, policy, N-1)


def ExpectedCumulativeRewardSignal(domain, policy):
    # method to return the Expected Cumulative Reward Signal with a policy in a domain
    state = domain.state
    S = JN(domain, policy, 10)
    domain.state = state
    return S


def MatrixJN(domain: Domain, policy: Policy.Policy, N):
    # method to return the list of Matrix of Expected value after N turn with a policy in a domain
    L = [np.array([[0. for k in range(domain.n)] for l in range(domain.m)])]
    for h in range(1, N):
        L.append(np.array([[0. for k in range(domain.n)] for l in range(domain.m)]))
        for i in range(domain.n):
            for j in range(domain.m):
                L[-1][j][i] = domain.reward([i, j], policy.action(domain.state))
                L[-1][j][i] += domain.gamma * (1 - domain.beta) * L[-2][min(max(j + policy.action(domain.state)[1], 0), domain.n - 1)][min(max(i + policy.action(domain.state)[0], 0), domain.m - 1)]
                L[-1][j][i] += domain.gamma * domain.beta * L[-2][0][0]
    return L


def MatrixJ(domain, policy):
    N = 0
    while ((domain.gamma ** N) * domain.B) / (1 - domain.gamma) > 0.01:
        N += 1
    return MatrixJN(domain, policy, N)[-1]


def MatrixQN(domain, N):
    L = [np.array([[{Domain.UP: 0, Domain.RIGHT: 0, Domain.DOWN: 0, Domain.LEFT: 0} for l in range(domain.n)] for m in range(domain.m)])]
    for h in range(1, N):
        L.append(np.array([[{Domain.UP: 0, Domain.RIGHT: 0, Domain.DOWN: 0, Domain.LEFT: 0} for l in range(domain.n)] for m in range(domain.m)]))
        for i in range(domain.n):
            for j in range(domain.m):
                for k in Domain.ACTION_SPACE:
                    L[-1][j][i][k] = domain.reward([i, j], k)
                    L[-1][j][i][k] += domain.gamma * (1 - domain.beta) * max(L[-2][min(max(j + k[1], 0), domain.n - 1)][min(max(i + k[0], 0), domain.m - 1)].values())
                    L[-1][j][i][k] += domain.gamma * domain.beta * max(L[-2][0][0].values())
    return L

def MatrixQ(domain):
    N=0
    while (2 * (domain.gamma**N) * domain.B) / (1-domain.gamma)**2 > 0.01:
        N += 1
    return MatrixQN(domain, N)[-1]

def mustar(domain):
    Q = MatrixQ(domain)
    mu = np.zeros_like(Q)
    for x in range(domain.n):
        for y in range(domain.m):
            mu[x, y] = max(Q[x, y], key=Q[x,y].get)
    return mu


