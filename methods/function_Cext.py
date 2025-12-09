#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 4 09:12:11 2022

@author: Luca Teruzzi
"""


############################################################################################################################
############################################################################################################################


import numpy as np, matplotlib.pyplot as plt
from math import nan
from scipy import optimize
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition


############################################################################################################################
############################################################################################################################


def matrix_R(N, pixel_size):

    """
    Measurements of the positions matrix and of the angles matrix

    Args
    -------------------------------------------------------
    N (int):
        Shape of the hologram
    pixel_size (float):
        Value of the pixel size (um)

    Returns
    -------------------------------------------------------
    R (:class:`.Image` or :class:`.VectorGrid`):
       Matrix of positions
    A (:class:`.Image` or :class:`.VectorGrid`):
       Matrix of angles (°)
    """

    onesvec = np.ones(N)
    inds = (np.arange(N)+.5 - N/2.) / (N-1.)
    X = np.outer(onesvec, inds)
    Y = np.transpose(X)
    R = np.sqrt(X**2. + Y**2.) * pixel_size * N
    A = -np.arctan2(X, Y)*180/np.pi+180

    return R, A


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#


def media_angolare(R, holo, pixel_size, lim):

    """
    Measurements of the angular average of the hologram and the angular integration

    Args
    -------------------------------------------------------
    R (:class:`.Image` or :class:`.VectorGrid`):
       Matrix of positions
    holo (:class:`.Image` or :class:`.VectorGrid`):
       Hologram in function of x,y
    pixel_size (float):
        Value of the pixel size (um)
    lim (int):
        Half length of the hologram (center of the hologram)

    Returns
    -------------------------------------------------------
    Integrale_array: (float or list of floats)
       Angular integration of the hologram
    total_aver: (float or list of floats)
       Angular average of the hologram
    """

    total_aver = np.array([])
    restrict = np.array([])
    Integrale_array = np.array([])

    freq = R[int(lim)][int(lim)+1]-R[int(lim)][int(lim)]
    x = np.arange(freq, R[int(lim)][0]+freq, freq)

    for i in x:

        Integrale = np.sum(holo.values[R <= i] * pixel_size**2)
        Integrale_array = np.append(Integrale_array, Integrale)

        restrict = np.sum(holo.values[(R < i) & (
            R >= i-freq)])/len(holo.values[(R < i) & (R >= i-freq)])
        total_aver = np.append(total_aver, restrict)

    return Integrale_array, total_aver


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#


def Cext_FIT(holo, pixel_size, z, fuoco, lim, k, x_fit_1, media, name_graph_3d, name_graph_2d):

    """
    Measurements of the FIT 2d of the hologram.
    By the fit you can have
    1) module of S(0)
    2) phase
    3) so by the Optical Theorem you can obtain the Cext

    Args
    -----------------------------------------------------
    holo (:class:`.Image` or :class:`.VectorGrid`):
        Hologram in function of x,y
    pixel_size (float):
        Value of the pixel size (um)
    z (float or list of floats):
       Distance of propagation
    fuoco (float):
        Position of the focal point (um)
    lim (int):
        Half length of the hologram (center of the hologram)
    k (float):
        Wavevector
    x_fit_1 (float or list of floats):
        X-array of the angular average
    media (float or list of floats):
        Angular average of the hologram
    name_graph_3d (str):
       Name of the image saved, Cext fit 3d
    name_graph_2d:
        Name of the image saved, Cext fit 2d

    Returns
    -------------------------------------------------------
    c (float):
        Value of the Cext
    err_c (float):
        error of the Cext value
    residui (:class:`.Image` or :class:`.VectorGrid`):
        residuals of the fit
    params (float):
        params of the fit: S(0), phase, sigma (of the exp), A (constant), zeta
    """

    x_fit = np.arange(len(holo))*pixel_size
    y_fit = np.arange(len(holo))*pixel_size
    x_f, y_f = np.meshgrid(x_fit, y_fit)

    def func_hologram_2d(xy_mesh, S, P, sigma, A, zeta):
        (x, y) = xy_mesh
        g = (A+2*(S)/(k*zeta) * np.cos((k/(2*zeta))*((x-xo)**2 + (y-yo)** 2) + P))*np.exp(-((x-xo)**2+(y-yo)**2)/(2*sigma**2))

        return g.ravel()

    zeta = z[fuoco]
    S = 13000
    P = np.pi/2
    sigma = 60
    xo = lim*pixel_size
    yo = lim*pixel_size
    A = 0.2


    srot_holo = np.array([])
    for i_img in np.arange(0, len(holo.values)):
        srot_holo = np.append(srot_holo, holo[i_img].values)

    try:
        params, params_covariance = optimize.curve_fit(func_hologram_2d, (x_f, y_f), srot_holo, p0=[S, P, sigma, A, zeta])

        data_fitted = func_hologram_2d((x_f, y_f), *params)
        perr = np.sqrt(np.diag(params_covariance))
        err_S = perr[0]
        err_P = perr[1]

        residui = np.abs(data_fitted.reshape(
            int(lim*2), int(lim*2))-srot_holo.reshape(int(lim*2), int(lim*2)))

        fig, (ax, ax2, cax) = plt.subplots(ncols=3, figsize=(
            12, 6), gridspec_kw={"width_ratios": [1, 1, 0.01]})
        fig.subplots_adjust(wspace=0.5)

        im = ax.imshow(srot_holo.reshape(int(lim*2), int(lim*2)), cmap='viridis', alpha=1,
                       extent=[0, lim*2*pixel_size, 0, lim*2*pixel_size], origin='bottom', vmin=-0.1, vmax=0.1)
        ax.set_xlabel("x ($\mu$m)", fontsize=20)
        ax.set_ylabel("y ($\mu$m)", fontsize=20)
        ax.tick_params(axis='both', which='both', labelsize=18)

        ax2.imshow(data_fitted.reshape(int(lim*2), int(lim*2)), cmap='viridis', extent=[
                         0, lim*2*pixel_size, 0, lim*2*pixel_size], origin='bottom', vmin=-0.1, vmax=0.1)
        ax2.set_xlabel("x ($\mu$m)", fontsize=20)
        ax2.set_ylabel("y ($\mu$m)", fontsize=20)
        ax2.tick_params(axis='both', which='both', labelsize=18)

        ip = InsetPosition(ax2, [1.05, 0, 0.05, 1])
        cax.set_axes_locator(ip)

        fig.colorbar(im, cax=cax, ax=[ax, ax2])
        plt.savefig(name_graph_3d)
        plt.clf()
        plt.close()

        c = 4*np.pi/(k**2)*np.real(params[0])*np.cos(np.pi/2-params[1])
        err_c = ((4*np.pi/(k**2)*np.cos(np.pi/2-params[1]))**2*err_S**2+(
            4*np.pi/(k**2)*np.real(params[0])*np.sin(np.pi/2-params[1]))**2*err_P**2)**0.5

        plt.plot(x_fit_1, media, '-.', label='data')

        plt.plot(x_fit[0:int(lim+1)], data_fitted.reshape(int(lim*2),
                                                          int(lim*2))[int(lim), int(lim):], '-.', label='fit')

        plt.plot(x_fit[0:int(lim)], params[3]+np.e**(-(x_fit[0:int(lim)]**2)/(2 * params[2]**2))
                 * params[0]*2/(k*params[4]), '-k', alpha=1, label='Gaussian Envelope')
        plt.title('Cext = {:.2f}'.format(c) + ' +- = {:.2f}'.format(err_c))
        plt.xlabel('x($\mu$m)')
        plt.ylabel('Intensity a.u')
        plt.legend()
        plt.savefig(name_graph_2d)
        plt.clf()
        plt.close()

    except RuntimeError:
        print("Error - curve_fit failed")
        c = 0
        err_c = 0
        residui = 0
        params = np.array([0, 0])

    return c, err_c, residui, params


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#


def Integration_tw_square(holo, lim, pixel_size):

    """
    Integration of the hologram from the center point to the edges throw square

    Args
    -----------------------------------------------------
    holo (:class:`.Image` or :class:`.VectorGrid`):
       Hologram in function of x,y
    lim (int):
        Half length of the hologram (center of the hologram)
    pixel_size (float):
        Value of the pixel size (um)

    Returns
     -----------------------------------------------------
    Integrale_array (float or list of floats):
       Integration of the hologram tw square
    """

    Integrale_array = np.array([])
    for r in np.arange(0, int(lim)+1, 1):

        Integrale = np.sum(
            holo[int(lim-r):int(lim+r), int(lim-r):int(lim+r)])*pixel_size**2
        Integrale_array = np.append(Integrale_array, Integrale)

    return Integrale_array


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#


def Cext_tw_integration(Integrale_array, save_label, save_path, save_format):

    """
    Plot of the integration of the hologram.
    By this you can have
    1) Cext
    2) The real part of S(0) with the Optical Theorem

    Args
    -----------------------------------------------------
    Integrale_array (float or list of floats):
       Integration of the hologram, can be tw circle or square

    Returns
    ------------------------------------------------------
    y[0]: (float)
        Value of the Cext
    """

    Integrale_array = Integrale_array[:]*1E-6

    inviluppo_sup = argrelextrema(Integrale_array[:], np.greater)[0]
    inviluppo_min = argrelextrema(Integrale_array[:], np.less)[0]

    x = np.arange(0, len(Integrale_array), 1)
   # x1 = np.arange(inviluppo_sup[0], len(Integrale_array), 1)



    print("inviluppo_min[@], inviluppo_sup[@] = "+str(inviluppo_min)+", "+str(inviluppo_sup))
    print("len(Integrale[inviluppo_min]) = "+str(len(Integrale_array[inviluppo_min])))
    print("len(Integrale[inviluppo_sup) = "+str(len(Integrale_array[inviluppo_sup])))
    
    
    
    
    # sup = interp1d(inviluppo_sup, Integrale_array[inviluppo_sup], kind='slinear', fill_value='extrapolate')
    # inf = interp1d(inviluppo_min, Integrale_array[inviluppo_min], kind='slinear', fill_value='extrapolate')

    # if len(inviluppo_sup) > 0 and len(inviluppo_min) > 0:

    #     y = (sup(x1) + inf(x1))/2
    #     y_interp = interp1d(x1, y, kind='slinear', fill_value='extrapolate')

    #     fig, ax = plt.subplots()
    #     ax.plot(x, Integrale_array, 'b', linewidth=2)
    #     plt.title('Extinction cross-section')
    #     ax.set_xlabel('x [pxl]')
    #     ax.set_ylabel('C$_{ext}$ [mm$^2$]')
    #     plt.tick_params(axis='both', which='both')
    #     ax_zoom = ax.inset_axes([0.5, 0.15, 0.46, 0.55])
    #     ax_zoom.set_xlim(40, 120)
    #     ax_zoom.set_ylim(1.34, 1.60)
    #     ax_zoom.plot(x, Integrale_array, 'b', linewidth=2)
    #     plt.tight_layout()
    #     if save_label==True: plt.savefig(save_path+'integrated_cext'+save_format)
    #     plt.show()
    #     plt.clf()

    fig1, ax1 = plt.subplots()
    ax1.plot(x, Integrale_array, 'b', linewidth=2)
    # ax1.plot(x1, sup(x1), '-.', linewidth=1)
    # ax1.plot(x1, inf(x1), '-.', linewidth=1)

    plt.title('Extinction cross-section')
    ax1.set_xlabel('x [pxl]')
    ax1.set_ylabel('C$_{ext}$ [mm$^2$]')
    #     ax1_zoom = ax1.inset_axes([0.5, 0.15, 0.46, 0.55])
    #     ax1_zoom.set_xlim(40, 120)
    #     ax1_zoom.set_ylim(1.34, 1.60)
    #     ax1_zoom.plot(x, Integrale_array, 'b', linewidth=2)
    #     ax1_zoom.plot(x1, sup(x1), '-.', color='dodgerblue', linewidth=1)
    #     ax1_zoom.plot(x1, inf(x1), '-.', color='orange', linewidth=1)
    plt.tick_params(axis='both', which='both')
    plt.tight_layout()
    if save_label==True: plt.savefig(save_path+'integrated_cext_2'+save_format)
    #plt.show()
    plt.clf()

    #     fig2, ax2 = plt.subplots()
    #     ax2.plot(x[x1[0]-7:], Integrale_array[x1[0]-7:], 'b', linewidth=2)
    #     ax2.plot(x1, sup(x1), '-.', linewidth=1, label='upper envelope')
    #     ax2.plot(x1, inf(x1), '-.', linewidth=1, label='lower envelope')
    # ax2.plot(x1, y_interp(x1), 'darkred', linewidth=1, label='average')
    # plt.title('Extinction cross-section --- ZOOMED ---')
    # ax2.set_xlabel('x [pxl]')
    # ax2.set_ylabel('C$_{ext}$ [mm$^2$]')
    # plt.arrow(x1[0]-1.5, y_interp(x1[0]), -6.0, 0.0, fc="darkred", ec="darkred",
    #              head_width=0.007, head_length=1.5, linestyle='-', linewidth=0.3)
    #    plt.tick_params(axis='both', which='both')
    #    ax2.set_ylim(1.34, 1.59)
    #     ax2.set_xlim(x1[0]-10, x1[len(x1)-1]+5)
    #     plt.legend(loc='lower right')
    #     plt.tight_layout()
    #     if save_label==True: plt.savefig(save_path+'cext_envelope'+save_format)
    #     plt.show()
    #     plt.clf()

    # return y_interp(x1[0])

    #  else:
    #     y = nan
    #     x = nan

    # return y


############################################################################################################################
############################################################################################################################
