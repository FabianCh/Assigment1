class MDP:

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

