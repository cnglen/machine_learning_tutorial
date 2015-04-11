#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo of gradient descent algorithm:
   - How gradient descent works for 1D, 2D, 3D (e.g, Find arg min J(\theta) where \theta ~ (1, 2x1, 3x1))
   - Effect of normalization, what happened if no normalization happend
Note: we use (X'X)^{-1}Xy to verify
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_data(n_samples=1024, n_features=1, n_informative=1, n_targets=1, random_state=1987, bias=13.17):
    """
    Using sklearn package to generate (X, y, theta) where theta = (theta_0, theta_1, ..., theta_{n_features})^T are parameters of the linear model

    Input: See sklearn.datasets.samples_generator.make_regression for more details
    Output: X     ~ n_samples * (n_features+1) including the addtional 1 vector
            y     ~ n_samples * n_targets
            theta ~ (n_features+1) * 1
    Usage: (X, y, theta) = load_data( ... )
    """

    from sklearn.datasets.samples_generator import make_regression
    X, y, theta = make_regression(n_samples=n_samples,
                                  n_features=n_features,
                                  n_informative=n_informative,
                                  n_targets=n_targets,
                                  random_state=random_state,
                                  bias=bias,
                                  coef=True,)
    theta = np.insert(theta, 0, bias)
    X = np.insert(X, 0, 1, axis=1)
    return (X, y, theta)


def compute_cost(X, y, theta, model='linear'):
    """
    Compute the cost function J(theta)

    Usage: J = compute_cost(X, y, theta)
    """
    n_samples = np.size(X, 0)
    J = np.sum((np.dot(X, theta) - y) ** 2) / n_samples / 2
    return J


def gradient_descent(X, y, alpha=0.01, n_iter=100, model='linear'):
    """
    Get model's parameter theta which minimize the MSE (Mean Squared Error)

    Usage: (theta, J_history, theta_history) = gradient_descent(X, y, ...)
    """
    n_samples = np.size(X, 0)
    n_features = np.size(X, 1) - 1
    theta = np.zeros(n_features + 1)
    J_history = np.zeros(n_iter)
    theta_history = np.zeros((n_iter, n_features + 1))

    # for linear model
    for i_iter in np.arange(0, n_iter):
        # theta = theta - alpha / n_samples * (X' * (X*theta - y));
        theta = theta - alpha / n_samples * np.dot(X.T, np.dot(X, theta) - y)
        J_history[i_iter] = compute_cost(X, y, theta)
        theta_history[i_iter, :] = theta

    return theta, J_history, theta_history


def demo_gradient_descent(scaling=True):
    """
    plot: J(# of iteratioin), J(theta) in {3D, contour}
    """
    X, y, coef = load_data(random_state=1988)
    if not scaling:
        X[:, 0] = X[:, 0] * 3
        filename_J_n_iter = 'J_n_iter_no_scaling.png'
        filename_J_theta_3D = 'J_theta_3D_no_scaling.png'
        filename_J_theta_contour = 'J_theta_contour_no_scaling.png'
    else:
        filename_J_n_iter = 'J_n_iter.png'
        filename_J_theta_3D = 'J_theta_3D.png'
        filename_J_theta_contour = 'J_theta_contour.png'

    alpha, n_iter = 0.1, 100
    theta, J_history, theta_history = gradient_descent(X, y, alpha=alpha, n_iter=n_iter)

    # J(# of iteratioin)
    fig_J_iter = plt.figure(1)
    axes_J_normal = fig_J_iter.add_subplot(121)
    index_iteration = np.arange(1, np.size(J_history) + 1)
    axes_J_normal.plot(index_iteration, J_history)
    plt.title(r'J(#iter), learning rate $\alpha=%0.2f$' % (alpha))
    axes_J_normal.set_xlabel('# of iteration')
    axes_J_normal.set_ylabel(r'Cost J(# of iter)')
    axes_J_normal.grid(True)
    axes_J_log = fig_J_iter.add_subplot(122)
    index_iteration = np.arange(1, np.size(J_history) + 1)
    axes_J_log.semilogy(index_iteration, J_history)
    plt.title(r'J(#iter), learning rate $\alpha=%0.2f$' % (alpha))
    axes_J_log.set_xlabel('# of iteration')
    axes_J_log.grid(True)
    plt.savefig('./figure/%s' % filename_J_n_iter)
    plt.close(fig_J_iter)

    # J(theta)
    theta_1 = np.arange(0, 2 * theta[1], 2 * theta[1] / 20)
    theta_0 = np.arange(theta[0] - theta[1], theta[0] + theta[1], 2 * theta[1] / 20)

    # theta_1 = np.arange(theta[1] - theta[0], theta[1] + theta[0], 5)
    # theta_0 = np.arange(0, 2 * theta[0], 5)

    Theta_0, Theta_1 = np.meshgrid(theta_0, theta_1)
    J = np.zeros(np.shape(Theta_0))
    for i0 in range(np.size(Theta_0, 0)):
        for i1 in range(np.size(Theta_0, 1)):
            tmp = np.array([Theta_0[i0][i1], Theta_1[i0][i1]])
            J[i0, i1] = compute_cost(X, y, tmp)

    fig_J_theta = plt.figure(2)
    axes_J_theta = fig_J_theta.gca(projection='3d')
    axes_J_theta.plot_wireframe(Theta_0, Theta_1, J, rstride=2, cstride=2)
    axes_J_theta.plot(np.insert(theta_history[:, 0], 0, 0),
                      np.insert(theta_history[:, 1], 0, 0),
                      np.insert(J_history, 0, compute_cost(X, y, [0, 0])),
                      marker='o', color='r', label=r'track of $(\theta_0, \theta_1)$')
    axes_J_theta.set_xlabel(r'$\theta_0$ ')
    axes_J_theta.set_ylabel(r'$\theta_1$ ')
    axes_J_theta.set_zlabel(r'$J(\theta_0, \theta_1)$')
    axes_J_theta.set_title(r'$J(\theta_0, \theta_1)$')
    axes_J_theta.view_init(azim=19, elev=44)
    plt.legend()
    plt.savefig('./figure/%s' % filename_J_theta_3D)
    plt.close(fig_J_theta)

    fig_J_theta_contour = plt.figure(3)
    cs = plt.contour(Theta_0, Theta_1, J, levels=np.arange(0, 200, 20))
    plt.clabel(cs, inline=1, fontsize=10)
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.plot(np.insert(theta_history[:, 0], 0, 0),
             np.insert(theta_history[:, 1], 0, 0),
             marker='x', color='k', label=r'track of $(\theta_0, \theta_1)$')
    plt.legend()
    plt.savefig('./figure/%s' % filename_J_theta_contour)
    plt.close(fig_J_theta_contour)

if __name__ == '__main__':
    demo_gradient_descent(scaling=True)
    demo_gradient_descent(scaling=False)
