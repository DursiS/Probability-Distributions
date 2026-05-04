from Distributions_Visual import Binomial
import matplotlib.pyplot as plt
import numpy as np
import math


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
        mean = self.batch_sample(k)
        mu = self.expectation()
        std = round(self.variance() ** 0.5, 8)

        return (mean - mu) / (std / (k**0.5))

    def get_samples(self, nx: int, k: int) -> list[float]:
        """Helper to get_points.

        Return the list of x values to plot this distribution.
        """

        lst = []
        for i in range(nx):
            lst.append(self.mean_sample(k))
        return lst

    def get_density(self, nx: int, dx: float, x: list[float]) -> list[float]:
        """Helper to get_points.

        Return the list of y values to plot this distribution of <nx> samples,
        as the density of each value in <x> in <dx> bins.
        """
        bins, lst = np.arange(-3, 3, dx), []

        d1 = bins[0]
        for d2 in bins[1:]:  # "For every bin..."
            freq = 0
            for point in x:
                if d1 < point <= d2:
                    freq += 1
            lst.append(freq / len(x))  # its probability
        return lst

    def get_average_distribution(
        self, nx: int, k: int, dx: float
    ) -> tuple[list[float], list[float]]:
        """Plot the points of the mean distribution of <nx>
        distributions of <k> size batch samples.

        To be represented as a density
        histogram of width <dx>.
        """
        x = self.get_samples(nx, k)
        y = self.get_density(nx, dx, x)
        return x, y


if __name__ == "__main__":
    n = 1000
    _theta = 1 / 2
    _k = 1  # Batch sample size
    _dx = 0.1  # Bin width
    _nx = 1000

    b1 = BinomialCentralLimit(n, _theta)
    x, y = b1.get_average_distribution(_nx, _k, _dx)
    print(x)
    plt.hist(x, bins=30, density=True)
    plt.show()
