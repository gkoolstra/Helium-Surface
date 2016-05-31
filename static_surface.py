from matplotlib import pyplot as plt
from scipy.sparse import dia_matrix, block_diag
from scipy.sparse.linalg import spsolve
from numpy import *
import numpy as np
import time
from Common import common

def reset_constants(verbose=False):
    """
    Returns constants and parameters.
    """
    constants = {
            'sigma' : 3.78E-4, # Surface tension
            'rho' : 145, # Density
            'g' : 9.81, # Gravity
            'kappa' : 0.7E-6, # Substrate to liquid Helium Vanderwaals parameter
            'eps0' : 8.85E-12, # Permittivity of free space
            'epsHe' : 1.056 # Relative permittivity of liquid He
            }

    parameters = {
                'w' : 20E-6, # Channel width (x-dimension)
                'l' : 20E-6, # Channel length (y-dimension)
                'h' : 1E-3, # Reservoir height difference
                'd0' : 0.4E-6, # Height of the walls of the channel
                 }

    if verbose:
        print_constants_parameters(constants, parameters)

    return constants, parameters


def print_constants_parameters(constants, parameters):
    """
    :param constants: Dictionary with constants
    :param parameters: Dictionary with parameters
    :return:
    """
    for c in constants.keys():
        print "%s = %e" % (c, constants[c])
    for p in parameters.keys():
        print "%s = %e" % (p, parameters[p])


def z(constants, parameters, x):
    """
    Shooting method for the boundary value problem. Solves two problems, the first problem has boundary
    conditions y(0) = d0, y'(0) = 0. The second problem has boundary conditions y(0) = 0, y'(0) = 1.
    The solutions are then combined to get the correct solution.

    Input
    constants: dictionary of constants
    parameters:
    xsteps:
    dx:

    Returns
    solution to the boundary value problem
    """

    sigma = constants['sigma']
    rho = constants['rho']
    g = constants['g']
    d0 = parameters['d0']
    h = parameters['h']

    dx = np.diff(x)[0]

    # Solve problem (I)
    u = d0*np.ones(len(x))
    u[0] = d0
    u[1] = d0
    for idx, X in enumerate(x[2:]):
        u[idx+2] = dx**2/sigma * (rho*g*(h+u[idx+1])) - (u[idx]-2*u[idx+1])

    # Solve problem (II)
    v = d0*ones(len(x))
    v[0] = 0
    v[1] = dx
    for idx, X in enumerate(x[2:]):
        v[idx+2] = dx**2/sigma * (rho*g*(h+v[idx+1])) - (v[idx]-2*v[idx+1])

    # Determine theta
    theta = (d0 - u[-1])/v[-1]

    # Get the final solution
    D = u + v*theta

    return D


def z_with_E(constants, parameters, x, Esq):
    """
    Shooting method for the boundary value problem.

    Input
    constants: dictionary of constants
    parameters: dictionary of parameters (mostly lengths)
    x: x-coordinates
    Esq: squared magnitude of the electric field as a function of x. Must have same length as x.

    Returns
    solution to the boundary value problem
    """

    sigma = constants['sigma']
    rho = constants['rho']
    g = constants['g']
    d0 = parameters['d0']
    h = parameters['h']
    epsHe = constants['epsHe']
    eps0 = constants['eps0']

    dx = np.diff(x)[0]

    # Solve problem (I)
    u = d0*np.ones(len(x))
    u[0] = d0
    u[1] = d0
    for idx, X in enumerate(x[2:]):
        u[idx+2] = dx**2/sigma * (rho*g*(h+u[idx+1]) - eps0*(epsHe-1)/2.* Esq[idx]) - (u[idx]-2*u[idx+1])

    # Solve problem (II)
    v = d0*np.ones(len(x))
    v[0] = 0
    v[1] = dx
    for idx, X in enumerate(x[2:]):
        v[idx+2] = dx**2/sigma * (rho*g*(h+v[idx+1]) - eps0*(epsHe-1)/2.* Esq[idx]) - (v[idx]-2*v[idx+1])

    # Determine theta
    theta = (d0 - u[-1])/v[-1]

    # Get the final solution
    D = u + v*theta

    return D


def z_with_E(constants, parameters, x, Esq):
    """
    Shooting method for the boundary value problem.

    Input
    constants: dictionary of constants
    parameters: dictionary of parameters (mostly lengths)
    x: x-coordinates
    Esq: squared magnitude of the electric field as a function of x. Must have same length as x.

    Returns
    solution to the boundary value problem
    """

    sigma = constants['sigma']
    rho = constants['rho']
    g = constants['g']
    d0 = parameters['d0']
    h = parameters['h']
    epsHe = constants['epsHe']
    eps0 = constants['eps0']

    dx = np.diff(x)[0]

    # Solve problem (I)
    u = d0*np.ones(len(x))
    u[0] = d0
    u[1] = d0
    for idx, X in enumerate(x[2:]):
        u[idx+2] = dx**2/sigma * (rho*g*(h+u[idx+1]) - eps0*(epsHe-1)/2.* Esq[idx]) - (u[idx]-2*u[idx+1])

    # Solve problem (II)
    v = d0*np.ones(len(x))
    v[0] = 0
    v[1] = dx
    for idx, X in enumerate(x[2:]):
        v[idx+2] = dx**2/sigma * (rho*g*(h+v[idx+1]) - eps0*(epsHe-1)/2.* Esq[idx]) - (v[idx]-2*v[idx+1])

    # Determine theta
    theta = (d0 - u[-1])/v[-1]

    # Get the final solution
    D = u + v*theta

    return D


def z_2D(x, y, constants, parameters, Esquared=None, verbose=True):
    """
    Calculates a 2D profile of the liquid helium interface using a grid of x and y points.
    Boundary conditions are assumed to be specified in parameters['d0'] around the entire border.
    The spacing in the x and y arrays must be constant, but does not have to be equal, i.e. dx != dy.

    :param x: Array of x-points
    :param y: Array of y-points
    :param constants: Dictionary with constants
    :param parameters: Dictionary with parameters
    :return: 2D Profile of the surface
    """

    d0 = parameters['d0']
    rho = constants['rho']
    g = constants['g']
    h = parameters['h']
    sigma = constants['sigma']
    eps0 = constants['eps0']
    epsHe = constants['epsHe']

    dx = np.diff(x)[0]
    dy = np.diff(y)[0]

    # Construct the matrix
    N = len(x)
    M = len(y)
    diag_x = -2*(1/dx**2 + 1/dy**2)*np.ones(N)
    off_diag_x = 1/dx**2 * np.ones(N)
    data = np.vstack((diag_x, off_diag_x, off_diag_x))
    offsets = np.array([0, -1, +1])
    diag_block = dia_matrix((data, offsets), shape=(N,N))
    diag_matrix = block_diag([diag_block]*M)

    off_diag_y = 1/dy**2 * ones(M*N)
    off_diag_matrix = dia_matrix((vstack((off_diag_y, off_diag_y)), np.array([-N, +N])), shape=(M*N, M*N))

    full_matrix = diag_matrix + off_diag_matrix

    # Deal with the electric field
    if Esquared is None:
        E2 = np.zeros((M*N))
        E2 = E2.flatten()
    else:
        # Reshape the matrix
        if shape(Esquared) == (M,N):
            E2 = Esquared.flatten()
        else:
            E2 = np.zeros((M,N))
            E2 = E2.flatten()
            print "Althought Esquared was specfied, it doesn't have the right shape: %d x %d. I expected %d x %d. Ignoring Esquared for now." \
                  % (np.shape(Esquared)[0], np.shape(Esquared)[1], M, N)

    # Construct the right hand side of the equation and implement boundary conditions
    rhs = rho*g*h/sigma * np.ones(M*N) - eps0*(epsHe-1)/(2*sigma)*E2
    # Boundary at y = 0 and y = ymax:
    rhs[1:N-1] -= d0/dy**2
    rhs[N*M-N+1:N*M-1] -= d0/dy**2

    # Boundary at x = 0 and x = xmax
    for k in range(1, M-1):
        rhs[N*k] -= d0/dx**2
        rhs[N*(k+1)-1] -= d0/dx**2

    # Corner points
    rhs[0] -= d0*(1/dx**2 + 1/dy**2)
    rhs[N-1] -= d0*(1/dx**2 + 1/dy**2)
    rhs[N*M-N] -= d0*(1/dx**2 + 1/dy**2)
    rhs[N*M-1] -= d0*(1/dx**2 + 1/dy**2)


    # Solve the system of equations
    t0 = time.time()
    solution = spsolve(full_matrix, rhs)
    t1 = time.time()

    if verbose:
        print "Solution took %.3f s" % (t1-t0)

    return solution.reshape((M,N))


def draw_channel(parameters, x):
    """
    Draw a gray block that represents a channel wall.
    :param parameters: Parameters dictionary
    :param x: An array of at least two points defining the x-coordinates of the wall.
    :return:
    """
    y = 1E6*parameters['d0']*np.ones(len(x))
    plt.fill_between(x, y, y2=0, alpha=0.2, color='gray', lw=0)


def load_maxwell_data(df, do_plot=True, do_log=True, xlim=None, ylim=None, clim=None,
                       figsize=(6.,12.), plot_axes='xy', cmap=plt.cm.Spectral):
    """
    :param df: Path of the Maxwell data file (fld)
    :param do_plot: Use pcolormesh to plot the 3D data
    :param do_log: Plot the log10 of the array. Note that clim has to be adjusted accordingly
    :param xlim: Dafaults to None. May be any tuple.
    :param ylim: Defaults to None, May be any tuple.
    :param clim: Defaults to None, May be any tuple.
    :param figsize: Tuple of two floats, indicating the figure size for the plot (only if do_plot=True)
    :param plot_axes: May be any of the following: 'xy' (Default), 'xz' or 'yz'
    :return:
    """

    data = np.loadtxt(df, skiprows=2)
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    magE = data[:,3]

    # Determine the shape of the array:
    if 'x' in plot_axes:
        for idx, X in enumerate(x):
            if X != x[0]:
                ysize=idx
                xsize=np.shape(magE)[0]/ysize
                break
    else:
        for idx, Y in enumerate(y):
            if Y != y[0]:
                ysize=idx
                xsize=np.shape(magE)[0]/ysize
                break

    # Cast the voltage data in an array:
    if plot_axes == 'xy':
        X = x.reshape((xsize, ysize))
        Y = y.reshape((xsize, ysize))
    if plot_axes == 'xz':
        X = x.reshape((xsize, ysize))
        Y = z.reshape((xsize, ysize))
    if plot_axes == 'yz':
        X = y.reshape((xsize, ysize))
        Y = z.reshape((xsize, ysize))

    E = magE.reshape((xsize, ysize))

    if do_plot:
        plt.figure(figsize=figsize)
        common.configure_axes(15)
        if do_log:
            plt.pcolormesh(X*1E6, Y*1E6, np.log10(E), cmap=cmap)
        else:
            plt.pcolormesh(X*1E6, Y*1E6, E, cmap=cmap)

        plt.colorbar()

        if clim is not None:
            plt.clim(clim)
        if xlim is None:
            plt.xlim([np.min(x)*1E6, np.max(x)*1E6]);
        else:
            plt.xlim(xlim)
        if ylim is None:
            plt.ylim([np.min(y)*1E6, np.max(y)*1E6]);
        else:
            plt.ylim(ylim)
        plt.xlabel('x ($\mu\mathrm{m}$)')
        plt.ylabel('y ($\mu\mathrm{m}$)')

    return X, Y, E


def select_domain(X, Y, Esquared, xdomain=None, ydomain=None):
    """
    Selects a specific area determined by xdomain and ydomain in X, Y and Esquared. X, Y and Esquared may be
    obtained from the function load_maxwell_data. To retrieve a meshgrid from the returned 1D arrays x_cut and y_cut,
    use Xcut, Ycut = meshgrid(x_cut, y_cut)
    :param X: a 2D array with X coordinates
    :param Y: a 2D array with Y coordinates
    :param Esquared: Electric field squared. Needs to be the same shape as X and Y
    :param xdomain: Tuple specifying the minimum and maximum of the x domain
    :param ydomain: Tuple specifying the minimum and the maximum of the y domain
    :return:
    """

    if xdomain is None:
        xmin = np.min(X[:,0])
        xmax = np.max(X[:,0])
        if ydomain is None:
            ymin = np.min(Y[0,:])
            ymax = np.max(Y[0,:])
    elif ydomain is None:
        ymin = np.min(Y[0,:])
        ymax = np.max(Y[0,:])
    else:
        xmin, xmax = xdomain
        ymin, ymax = ydomain

    if np.shape(X) == np.shape(Y) == np.shape(Esquared):
        if len(np.shape(X)) > 1 and len(np.shape(Y)) > 1:
            x = X[:,0]
            y = Y[0,:]
        else:
            print "The shape of X and/or Y are not consistent. Aborting. Please Check."
            return

        x_cut = x[np.logical_and(x>=xmin, x<=xmax)]
        y_cut = y[np.logical_and(y>=ymin, y<=ymax)]

        xidx = np.where(np.logical_and(x>=xmin, x<=xmax))[0]
        yidx = np.where(np.logical_and(y>=ymin, y<=ymax))[0]

        Esquared_cut = np.transpose(Esquared[xidx[0]:xidx[-1]+1, yidx[0]:yidx[-1]+1])

        return x_cut, y_cut, Esquared_cut
    else:
        print r"Shapes of X, Y and Esquared are not consistent:\nShape X: %d x %d\nShape Y: %d x %d\nShape Esquared: %d x %d "\
              %(np.shape(X)[0], np.shape(X)[1], np.shape(Y)[0], np.shape(Y)[1], np.shape(Esquared)[0], np.shape(Esquared)[1])


def z_from_fld(df, V=1.0, h=0.1E-3, d0=0.4E-6, xdomain=None, ydomain=None,
               plot_Efield=True, plot_surface=True, verbose=True, **kwargs):
    """
    Calculate the surface deformation from an fld file from Maxwell. This file should contain electric field
    data (Not E**2) and the contents can be scaled using the argument V.

    **kwargs may be any argument that can be passed into pcolormesh to adjust the surface plot.
    Of course this is only applicable if plot_surface=True. To change the color limits, use
    vmin=min_value and vmax=max_value.

    :param df: Maxwell datafile in fld format (exported from the field calculator).
    :param V: Applied voltage to the electrodes determines the electric field
    :param plot_Efield: Plot the E^2
    :param plot_surface: Plot the surface deformation.
    :return:
    """

    # Reset the constants and create the parameters for the model
    constants, parameters = reset_constants(verbose=False)
    parameters['h'] = h
    parameters['d0'] = d0

    # Load the data from the file
    X,Y,E0 = load_maxwell_data(df, do_plot=False)

    if verbose:
        print "Loaded data from FLD file..."

    # Cut the data for processing.
    xcut, ycut, Esquaredcut = select_domain(X, Y, V**2 * E0**2, xdomain=xdomain, ydomain=ydomain)

    # Plot, if necessary
    if plot_Efield:
        plt.figure(figsize=(12.,6.))
        common.configure_axes(13)
        plt.pcolormesh(xcut*1E6, ycut*1E6, np.log10(Esquaredcut), cmap=plt.cm.Spectral)
        plt.xlabel(r'x ($\mu$m)'); plt.xlim(min(xcut)*1E6, max(xcut)*1E6);
        plt.ylabel(r'y ($\mu$m)'); plt.ylim(min(ycut)*1E6, max(ycut)*1E6);
        plt.colorbar(); plt.title(r'$|E|^2$ for selected domain')

    parameters['w'] = xcut[-1] - xcut[0]
    parameters['l'] = ycut[-1] - ycut[0]

    if verbose:
        print_constants_parameters(constants, parameters)

    d = z_2D(xcut, ycut, constants, parameters, Esquared=Esquaredcut, verbose=verbose)

    # Plot surface profile, if necessary
    if plot_surface:
        fig2 = plt.figure(figsize=(12.,6.))
        common.configure_axes(13)
        plt.pcolormesh(xcut*1E6, ycut*1E6, (parameters['d0']-d)*1E9, **kwargs)
        plt.xlabel("x ($\mu$m)")
        plt.ylabel("y ($\mu$m)")
        plt.colorbar()
        plt.title("2D surface profile: $d_0 - d$, color in nm")
        plt.xlim(min(xcut)*1E6, max(xcut)*1E6);
        plt.ylim(min(ycut)*1E6, max(ycut)*1E6);

    return xcut, ycut, d


def integrate_energy(X, Y, magE, xdomain, ydomain, epsilon_r=1.0, do_plot=False, cmap=plt.cm.viridis, figsize=(7.,4.)):
    """
    Returns the electric field energy in a portion of the domain specified by (xmin, xmax) and (ymin, ymax)
    If the region is filled with a different dielectric than vacuum, you can set epsilon_r > 1.0
    :param x: 2D array of X data
    :param y: 2D array of Y data
    :param magE: 2D array containing the magnitude of the electric field
    :param xdomain: (xmin, xmax)
    :param ydomain: (ymin, ymax)
    :param epsilon_r: Must be >= 1.0
    :param do_plot: Plot the cropped magE matrix
    :return: 0.5* epsilon_0 * epsilon_r * np.sum(np.abs(magE_selection)**2) * dx * dy
    """
    epsilon_0 = 8.85E-12
    x = X[:,0]
    y = Y[0,:]

    xmin, xmax = xdomain
    ymin, ymax = ydomain

    # Select the region that you want to integrate over:
    n_start = common.find_nearest(x, xmin)
    n_stop = common.find_nearest(x, xmax)

    m_start = common.find_nearest(y, ymin)
    m_stop = common.find_nearest(y, ymax)

    #print n_start, n_stop, m_start, m_stop

    magE_selection = magE[n_start:n_stop, m_start:m_stop]

    if do_plot:
        plt.figure(figsize=figsize)
        plt.pcolormesh(X[n_start:n_stop, m_start:m_stop]*1E6, Y[n_start:n_stop, m_start:m_stop]*1E6,
                       np.log10(magE_selection), cmap=cmap)
        plt.colorbar()
        plt.xlim([xmin*1E6, xmax*1E6]);
        plt.ylim([ymin*1E6, ymax*1E6]);
        plt.xlabel('x ($\mu\mathrm{m}$)')
        plt.ylabel('y ($\mu\mathrm{m}$)')

    dx = np.diff(x)[0]
    dy = np.diff(y)[0]

    return 0.5 * epsilon_0 * epsilon_r * np.sum(np.abs(magE_selection)**2) * dx * dy














