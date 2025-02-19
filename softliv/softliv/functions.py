"""
Author: Takumi Matsuzawa
Last updated: 2023/11/25

Description: This file contains standard functions for fitting.
"""

import numpy as np
from scipy.optimize import curve_fit

###############################################################
# Standard functions
def linear(x, a, b):
    """a * x + b"""
    return a * x + b
def quad(x, a, b, c):
    """ a*x**2 + b*x + c """
    return a*x**2 + b*x + c
def quad2(x, a, b, c):
    """a*(x-b)**2 + c"""
    return a*(x-b)**2 + c
def power(x, a, b):
    """a * x ** b"""
    return a * x ** b
def power2(x, x0, a, b):
    """a * (x-x0) ** b"""
    return a * (x-x0) ** b
def exp(x, a, b):
    """a * np.exp(-b * x)"""
    return a * np.exp(-b * x)

def exp2(x, a, b, x0):
    """a * np.exp(-b * (x-x0))"""
    return a * np.exp(-b * (x-x0))

def gaussian(x, a, x0, sigma, b):
    """ a * np.exp(- (x - x0) ** 2. / (2. * sigma ** 2.)) + b"""
    return a * np.exp(- (x - x0) ** 2. / (2. * sigma ** 2.)) + b

def gaussian_norm(x, x0, sigma):
    """ 1. / (np.sqrt(2 * np.pi * sigma ** 2.)) * np.exp(- (x - x0) ** 2. / (2. * sigma ** 2.)) """
    return 1. / (np.sqrt(2 * np.pi * sigma ** 2.)) * np.exp(- (x - x0) ** 2. / (2. * sigma ** 2.))

def lorentzian(x, x0, gamma, alpha):
    """alpha * gamma / ((x-x0)**2 + gamma ** 2)"""
    return alpha * gamma / ((x-x0)**2 + gamma ** 2)

def lorentzian_norm(x, x0, gamma):
    """1. / np.pi * gamma / ((x-x0)**2 + gamma ** 2)"""
    return 1. / np.pi * gamma / ((x-x0)**2 + gamma ** 2)


###############################################################

def fit(x, y, func=linear, **kwargs):
    """
    Fit data with a function (default: linear function)

    Parameters
    ----------
    xdata arr, data to fit
    ydata: arr, data to fit
    func: function, default: linear, a function to fit the data
        ... e.g. linear, quad, power, exp, gaussian, lorentzian, lambda x: a*x+b, etc.
    kwargs: dict, optional, keyword arguments to pass to scipy.optimize.curve_fit

    Returns
    -------
    popt: arr, optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized
    pcov: 2d arr, the estimated covariance of popt
    ... error of fit: perr = np.sqrt(np.diag(pcov))
    """
    popt, pcov = curve_fit(func, x, y, **kwargs)
    return popt, pcov
