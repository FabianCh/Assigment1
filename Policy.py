
class Policy:
    def __init__(self, policy):
        self.policy = policy

    def action(self, state):
        x, y = state
        return self.policy[x][y]

