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
        """Helper to get_average_distribution

        Return a list of <nx> batch samples values of size <k>.
        """

        lst = []
        for i in range(nx):
            lst.append(self.mean_sample(k))
        return lst

    def get_density(self, nx: int, dx: float, x: list[float]) -> list[float]:
        """Return the list of density values to plot this distribution."""
        bins, lst = np.arange(-3, 3, dx), []

        d1 = bins[0]
        for d2 in bins[1:]:  # "For every bin..."
            freq = 0
            for point in x:
                if d1 < point <= d2:
                    freq += 1
            lst.append(freq / len(x))  # its probability
        return lst

    def get_average_distribution(self, nx: int, k: int, dx: float) -> list[float]:
        """Return a list of the densities of <nx> batch samples
        of size <k>. To be represented as a density histogram of width <dx>."""

        samples = self.get_samples(nx, k)
        return self.get_density(nx, dx, samples)


if __name__ == "__main__":
    n = 500
    _nx = 1000  # Number of samples in the batches
    _theta = 1 / 2
    _k = 10  # Batch sample size
    _dx = 0.1  # Bin width

    b1 = BinomialCentralLimit(n, _theta)

    if False:

        d = b1.get_samples(1000, 50)
        plt.hist(d, bins=(math.ceil(6 / _dx)), density=True)
        plt.show()

    if True:  # Visualize how it converges for bigger samples

        n1, n2, n3, n4 = 100, 250, 500, 1000
        d1 = b1.get_samples(n1, _k)
        d2 = b1.get_samples(n2, _k)
        d3 = b1.get_samples(n3, _k)
        d4 = b1.get_samples(n4, _k)
        data = [d1, d2, d3, d4]
        labels = [n1, n2, n3, n4]

        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        for ax, data, label in zip(axs.flatten(), data, labels):
            ax.hist(data, bins=30, density=True, label=f"")
            ax.set_title(f"Batch Samples: {label}, Size: {_k}")
        plt.show()
