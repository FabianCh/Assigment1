import unittest
import Domain
import Policy
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_determinist_domain_g_moves_reward(self):
        determinist_domain = Domain.Domain(x=0, y=3, board=np.array([[-3, 1, -5, 0, 19],
                                                     [6, 3, 8, 9, 10],
                                                     [5, -8, 4, 1, -8],
                                                     [6, -9, 4, 19, -5],
                                                     [-20, -17, -4, -3, 9]]), beta=0)
        self.assertEqual(determinist_domain.B, 19)
        self.assertEqual(determinist_domain.g(0, 0), -3)
        self.assertEqual(determinist_domain.reward(determinist_domain.state, determinist_domain.LEFT), 6)
        determinist_domain.moves(determinist_domain.LEFT)
        self.assertEqual(determinist_domain.g(*determinist_domain.state), 6)
        self.assertEqual(determinist_domain.reward(determinist_domain.state, determinist_domain.UP), 5)
        determinist_domain.moves(determinist_domain.UP)
        self.assertEqual(determinist_domain.g(*determinist_domain.state), 5)
        self.assertEqual(determinist_domain.reward(determinist_domain.state, determinist_domain.RIGHT), -8)
        determinist_domain.moves(determinist_domain.RIGHT)
        self.assertEqual(determinist_domain.g(*determinist_domain.state), -8)
        self.assertEqual(determinist_domain.reward(determinist_domain.state, determinist_domain.DOWN), -9)
        determinist_domain.moves(determinist_domain.DOWN)
        self.assertEqual(determinist_domain.g(*determinist_domain.state), -9)

    def test_JN(self):
        determinist_domain = Domain.Domain(x=0, y=3, board=np.array([[-3, 1, -5, 0, 19],
                                                                     [6, 3, 8, 9, 10],
                                                                     [5, -8, 4, 1, -8],
                                                                     [6, -9, 4, 19, -5],
                                                                     [-20, -17, -4, -3, 9]]), beta=0)
        uppolicy = Policy.Policy(Domain.Domain.UP)
        self.assertEqual(Domain.JN(determinist_domain, uppolicy, 0), 0)
        determinist_domain.state = [0, 3]
        self.assertEqual(Domain.JN(determinist_domain, uppolicy, 1), 5)
        determinist_domain.state = [0, 3]
        self.assertEqual(Domain.JN(determinist_domain, uppolicy, 2), 10.94)

    def test_ExpectedCumulativeRewardSignal(self):
        determinist_domain = Domain.Domain(x=0, y=3, board=np.array([[-3, 1, -5, 0, 19],
                                                                     [6, 3, 8, 9, 10],
                                                                     [5, -8, 4, 1, -8],
                                                                     [6, -9, 4, 19, -5],
                                                                     [-20, -17, -4, -3, 9]]), beta=0)
        uppolicy = Policy.Policy(Domain.Domain.UP)
        self.assertEqual(Domain.ExpectedCumulativeRewardSignal(determinist_domain, uppolicy), -11.775377497358651)

    def test_MatrixJN(self):
        determinist_domain = Domain.Domain(x=0, y=3, board=np.array([[-3, 1, -5, 0, 19],
                                                                     [6, 3, 8, 9, 10],
                                                                     [5, -8, 4, 1, -8],
                                                                     [6, -9, 4, 19, -5],
                                                                     [-20, -17, -4, -3, 9]]), beta=0)
        uppolicy = Policy.Policy(Domain.Domain.UP)
        L = Domain.MatrixJN(determinist_domain, uppolicy, 4)
        self.assertEqual(L[0][0][0], 0)
        self.assertEqual(L[1][0][0], -3)
        self.assertEqual(L[2][0][0], -5.97)
        self.assertEqual(L[3][0][0], -8.9103)


if __name__ == '__main__':
    unittest.main()
