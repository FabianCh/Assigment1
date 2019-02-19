import Domain
import Policy
import numpy as np


domain = Domain.Domain(x=0, y=3, board=np.array([[-3, 1, -5, 0, 19],
                                                    [6, 3, 8, 9, 10],
                                                    [5, -8, 4, 1, -8],
                                                    [6, -9, 4, 19, -5],
                                                    [-20, -17, -4, -3, 9]]), beta=0.25)

uppolicy = Policy.Policy(Domain.Domain.UP)


print(Domain.ExpectedCumulativeRewardSignal(domain, uppolicy))
for i in range(3):
    domain.state = [0, 3]
    print(Domain.JN(domain, uppolicy, i))

domain.state = [0, 3]
L = Domain.MatrixJN(domain, uppolicy, 4)
for i in range(len(L)):
    print(L[i], "\n")

print(Domain.MatrixQ(domain))
print(Domain.mustar(domain))