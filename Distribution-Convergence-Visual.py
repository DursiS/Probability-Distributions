from Distributions_Visual import Binomial
import matplotlib.pyplot as plt
import numpy as np


class BinomialCentralLimit(Binomial):
    """A RV approximately following an N(0, 1) distribution
    for far out enough mean.
    """

    def __init__(self, n: int, theta: float) -> None:
        super().__init__(n, theta)

    def mean_sample(self, k: int) -> float:
        """Return *one* draw of the RV defined in Central-Limit-Theorem
        given a mean of <k> samples.

        As k -> infinity, this sample distributes N(0,1 )

        Precondition: k > 0
        """
        mean = self.batch_sample(k) / k
        mu = self.expectation()
        sigma = self.variance()

        return (mean - self.n * mu) / (sigma * (self.n ** (1 / 2)))

    def get_points(self, k: int, nx: int, dx: float) -> tuple[list[int], list[float]]:
        """Plot the points of a distribution of <nx> samples
        of size <k> of this RV with matplotlib.
        To be represented as a density histogram of width <dx>.
        """
        x, y = [], []
        bins = np.arange(-self.n, self.n, dx)

        for i in range(nx):
            x.append(self.mean_sample(k))

        for j in range(nx - 1):
            freq = 0
            for point in x:
                if bins[j] < point <= bins[j + 1]:
                    freq += 1
            y.append(freq / ((nx * k) * dx))

        return x, y


if __name__ == "__main__":
    n = 100
    n1, n2, n3, n4 = 1, 10, 50, 100
    _theta = 1 / 2
    _k = 20
    _dx = 1

    b1 = BinomialCentralLimit(n, _theta)

    plt.subplot(1, 2, 1)
    # plt.subplot(1, 2, 2)
    # plt.subplot(1, 2, 3)
    # plt.subplot(1, 2, 4)

    x1, y1 = b1.get_points(_k, n1, _dx)
    plt.plot(x1, y1)

    # x2, y2 = b1.get_points(_k, n2, _dx)
    # x3, y3 = b1.get_points(_k, n3, _dx)
    # x4, y4 = b1.get_points(_k, n4, _dx)
