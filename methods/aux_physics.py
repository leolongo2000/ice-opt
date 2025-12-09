#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 19:28:32 2024

@author: leo
"""
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from math import atan2, sqrt

# Pressione di vapore saturo [Pa]
def p_sat1(T): #tetens equation (T>0 in gradi)
    A = 610.78
    B = 17.27
    C = 237.3
    return A * np.exp(B * T / (T + C))

def p_sat2(T): #tetens equation (T<0 in gradi)
    A = 610.78
    B = 21.875
    C = 265.5
    return A * np.exp(B * T / (T + C))

def p_sat(T):
    if isinstance(T, np.ndarray):
        return np.where(T > 0, p_sat1(T), p_sat2(T))
    else:
        if T > 0:
            return p_sat1(T)
        elif T < 0:
            return p_sat2(T)
        else:
            return (p_sat1(T) + p_sat2(T)) / 2
    

def rho_air(T, RH=0, p=101320):
    R = 8.31446  # J/Mol K
    pv = RH*p_sat(T)  # pressure of water vapor (Pa)
    MD = 0.0289652  # molar mass of dry air, 0.0289652 kg/mol
    MV = 0.018016  # molar mass of water vapor, 0.018016 kg/mol
    return ((p-pv)*MD + pv*MV)/(R*(T+273.15))




def beta_h2o(T):
    # Dati per l'acqua
    x_water = np.array([0.01, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    y_water = np.array([1.96, 2.09, 2.18, 2.23, 2.26, 2.265, 2.25, 2.22, 2.17, 2.11, 2.04]) * (10**9)

    # Dati per il ghiaccio
    #x_ice = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 273]) - 273.15
    
    
    x_ice = np.array([-223.15, -213.15, -203.15, -193.15, -183.15,
                  -173.15, -163.15, -153.15, -143.15, -133.15,
                  -123.15, -113.15, -103.15, -93.15, -83.15,
                  -73.15, -63.15, -53.15, -43.15, -33.15,
                  -23.15, -13.15, -3.15, -0.15])
    y_ice = np.array([1.100, 1.092, 1.084, 1.075, 1.066, 1.056, 1.046, 1.035, 1.024, 1.012, 1.000, 0.9879, 0.9752, 0.9622, 0.9489, 0.9353, 0.9214, 0.9073, 0.8934, 0.8796, 0.8661, 0.8529, 0.8400, 0.8361])*10**10

    # Interpolazione dei dati
    fK_bulk_water = interp1d(x_water, y_water, kind='quadratic', fill_value='extrapolate')
    ice_bulk_mod = interp1d(x_ice, y_ice, kind='linear', fill_value='extrapolate')
    
    
    if isinstance(T, np.ndarray):
        return np.where(T > 0, fK_bulk_water(T), ice_bulk_mod(T))
    if T < 0:
        return ice_bulk_mod(T)
    if T > 0:
        return fK_bulk_water(T)
    else:
        return (8.36700918367347 + 1.9689552775082535)/2
    
    
def beta_air(T):
    # Dati di temperatura e modulo di bulk per l'aria
    temperatures = np.array([35, 30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25])
    bulk_modulus = np.array([141835.2766552, 141841.33988976, 141838.29605191, 141834.67664681, 141834.9993025, 141835.69980226,141835.9553856, 141831.471418, 141828.77466875, 141831.78805812, 141828.77778377, 141831.99351548, 141828.48958096])

    # Interpolazione dei dati
    bulk_modulus_interp = interp1d(temperatures, bulk_modulus, kind='quadratic', fill_value='extrapolate')
    
    if isinstance(T, np.ndarray):
        return bulk_modulus_interp(T)
    else:
        return bulk_modulus_interp([T])[0]
    


# VELOCITÀ DEL SUONO IN Aria
def c_air(T, p=101320, RH=0): #wikipedia modulo di bulk con temperatura
    K = beta_air(T)  # Pascal wikipedia, modulo compressibilità aria 0.142 MPa
    return np.sqrt((K/rho_air(T, RH, p)))




def c_h2o(T):
    #K=2.2e9 [Pascal] modulo di compressibilità dell'acqua a pressione atmosferica
    return np.sqrt((beta_h2o(T)/rho_h2o(T)))
################################################################################



# LEGGE SPERIMENTALE DI KELL PER LA DENSITÀ DELL'ACQUA IN FUNZIONE DELLA TEMPERATURA
# Range of validity : [-30 ; 150] c
def rho_h2o(T):
    a = -2.8054253e-10
    b = 1.0556302e-7
    c = -4.6170461e-5
    d = -0.0079870401
    e = 16.945176
    f = 999.83952
    g = 0.01687985
    return (a*T**5 + b*T**4 + c*T**3 + d*T**2 + e*T + f)/(1+g*T)

#####################################################################################


# Spherical Coordinates
def cart2sph(x, y, z):
    r = sqrt(x**2 + y**2 + z**2)               
    theta = atan2(z, np.sqrt(x**2 + y**2))     
    phi = atan2(y, x)                           
    return r, theta, phi


def sph2cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def cart2sphA(pts):
    return np.array([cart2sph(x, y, z) for x, y, z in pts])


def appendSpherical(xyz):
    np.stack((xyz, cart2sphA(xyz)))
    
    
###############################################################################    
# funzione che disegna e restituisce le posizioni dei trasduttori del levitatore 
# acustico tiny-lev, utilizzato durante la tesi3
def scatter_calotte_sferiche(R, theta_max):

    n_giri = 3
    theta_set = np.linspace(0, theta_max, n_giri+1)  # theta

    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')

    coordinates = []
    for i in [1, 2, 3]:
        phi_set = np.linspace(0, 2 * np.pi, 6*(i)+1)  # phi
        theta = theta_set[i]

        for j in range(6*i):
            phi = phi_set[j]
            x, y, z = sph2cart(R, theta, phi)

            coordinates.append([x, y, z])
            coordinates.append([x, y, -z])

    coordinates = np.array(coordinates)
    X = coordinates[:, 0]*1e2
    Y = coordinates[:, 1]*1e2
    Z = coordinates[:, 2]*1e2

    ax.scatter(X, Y, Z, cmap="viridis", s=100)

    R = R*1e2  # (in cm per la visualizzazione)
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_zlim(-R, R)
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_zlabel('z [cm]')
    #plt.title("Tiny Lev configuration")
    ax.axis("off")
    plt.show()
    return coordinates

