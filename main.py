"""
Different forms of balanced dice
Studying whether the variance goes down in the long term
Coming up with a measure of 'predictability' and seeing which
balanced dice manage to minimize predictability
"""


import matplotlib.pyplot as plt
import numpy as np
import math
import scipy



def combine_probabilities(prob_1, prob_2):
    """
    example:
    prob_1 = 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6
    prob_2 = 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6
    returns: 1/36, 2/36, 3/36, ..., 7/36, ..., 1/36

    """
    pairs = np.outer(prob_1, prob_2)
    return np.array([np.trace(pairs, ii) for ii in range(-len(prob_1) + 1, len(prob_2))])

prob_1 = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
prob_2 = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
prob_3 = combine_probabilities(prob_1, prob_2)
print(combine_probabilities(prob_1, prob_3))

class ProbabilityDistribution:
    """
    x axis - different values
    y values - probablities
    """
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.y_normal = None
        self.mean = None
        self.var = None

    def normalize_x(self):
        """
        change the discrete values into continous probability values
        :return:
        """
        self.x =self.x / len(self.x)

    def normalize_y(self):
        """
        normalize incidence to integrate to 1
        :return:
        """
        # total = scipy.integrate.trapezoid(self.y, self.x)
        total = np.sum(self.y) * (self.x[1] - self.x[0])
        self.y =self.y / total

    def mean_and_var(self):
        self.mean = scipy.integrate.trapezoid(self.y * self.x, self.x)
        self.var = scipy.integrate.trapezoid(self.y * (self.x - self.mean)**2, self.x)
        return self.mean, self.var

    def approximate_as_normal(self):
        if self.mean is None:
            self.mean_and_var()
        self.y_normal = 1/np.sqrt(2 * np.pi * self.var) * np.exp(-1/2 * (self.x - self.mean) ** 2 / self.var)
        # error = np.sum((self.y_normal - self.y)**2)
        error = np.sum(abs(self.y_normal - self.y))
        # error = scipy.integrate.trapezoid(abs(self.y_normal - self.y), self.x)
        return error

    def show_histogram(self):
        plt.plot(self.x, self.y)

two_dice = ProbabilityDistribution(list(range(2, 13)), combine_probabilities([1/ 6] * 6, [1/ 6] * 6))

# prob dist on number of 6s, normal dice
p = 5/36
# n rolls: 36
ns = list(range(36, 36*150, 36*30))
ns = list(36 * np.array([1, 4, 7, 10, 15, 20]))
ns = list(36 * np.array(range(1, 20)))
vars = list()
errs = list()
for n in ns:
    print(n)
    possible_values = list(range(n + 1))
    prob = [scipy.special.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n+1)]
    series = ProbabilityDistribution(possible_values, prob)
    series.normalize_x()
    series.normalize_y()
    mean, var = series.mean_and_var()
    vars.append(var)
    error = series.approximate_as_normal()
    errs.append(error)
    print(error)
    # print(var / n)
    # series.show_histogram()
plt.legend(ns)
plt.xlabel('portion of rolls')
plt.plot(ns, vars)
plt.plot(ns, errs)