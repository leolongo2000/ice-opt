#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 00:18:31 2023

@author: leo
"""

from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import numpy as np
from math import atan2, sqrt


plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


#############################################

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






from scipy.interpolate import interp1d
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



def cart2sph(x, y, z):
    r = sqrt(x**2 + y**2 + z**2)                # r
    theta = atan2(z, sqrt(x**2 + y**2))         # theta
    phi = atan2(y, x)                            # phi
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



class trasducer:
    def __init__(self, x0, y0, z0, diameter, P0, V, frequency, c_sound, phase=0):
        self.x = x0
        self.y = y0
        self.z = z0
        self.a = diameter
        self.P0 = P0
        self.V = V
        self.f = frequency
        self.lamda = c_sound/frequency
        self.k = 2*np.pi/self.lamda
        self.phi0=phase
        
    def cartesian_coordinates(self):
        return self.x0, self.y0, self.z0

    def spherical_coordinates(self):
        return cart2sph(self.x0, self.y0, self.z0)

    def P(self, dens=100):

        x = np.linspace(-R_sphere, R_sphere, dens)
        y = np.linspace(-R_sphere, R_sphere, dens)
        z = np.linspace(-R_sphere, R_sphere, dens)
        x, y, z = np.meshgrid(x, y, z)

        r = np.sqrt((x - self.x)**2 + (y - self.y)**2 + (z - self.z)**2)

        P = self.P0 * self.V * np.exp(1j*(self.phi0+self.k*r))/(r)

        P = np.array(P)

        return P

    # def U(self, k1, k2):
    #     U = 2*k1*(np.abs( self.P() )**2) - 2*k2*np.linalg.norm(np.gradient( self.P() ))**2
    #     return U

    # def F(self):
    #     F = np.gradient(self.U())
    #     F = np.array(F)
    #     return F


###############################################################################


# "COSTANTI"
omega = 2 * np.pi * 40000  # rad/s
R_sphere = 13.17/2 * 1e-2  # m raggio della sfera grande del tiny lev
h = 0.0117  # m altezza calotta
theta_max = np.arccos(1-h/R_sphere)  # circa 35°


# diametro della gocciolina???
dg = 0.001  # m
V = 4/3*np.pi*((dg/2)**3)  # m^3 volume della gocciolina


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

tiny_lev_config = scatter_calotte_sferiche(R_sphere, theta_max)

dens = 100  # densità dei linspace
# T=20 #Fisso per il momento la temperatura a 20 gradi
# TT=[30, 10, 0, -10, -30]
TT = np.linspace(30, -30, 10)
TT = [20]
l = 0

sections_x = []
sections_z = []

for T in TT:
    l += 1
    rhop = rho_h2o(T)
    rho0 = rho_air(T)
    cp = c_h2o(T)
    c0 = c_air(T)

    P, U = 0, 0
    Fx, Fy, Fz = 0, 0, 0

    diameter = 0.010
    P0 = 0.13
    Volt = 11
    frequency = 40000
    phi0 = 13.9 * np.pi/180  # sfasamento in rad

    L = 2*R_sphere*1e2  # [cm]
    s = int(dens/4)
    ds = L/dens   
    _extent = [-L/2 + s*ds, L/2 - s*ds, -L/2 + s*ds, L/2 - s*ds]
    

    
    i=0
    for x0, y0, z0 in tiny_lev_config:

        t = trasducer(x0, y0, z0, diameter, P0, Volt, frequency, c0, phi0)

        P += t.P(dens)
        i+=1
        acoustic_field_xz = np.abs(P[s:-s, P.shape[0]//2, s:-s].T)
        plt.imshow(acoustic_field_xz, extent=_extent)
        plt.xlabel("x [cm]")
        plt.ylabel("z [cm]")
        plt.savefig("/home/leo/Scrivania/unimi/lab-ottica/presentazione_lab_ottica/ice-opt/20250606/gif_accnesione_singola_trasduttori/" + str(i) + '.png')
        plt.show()

    
    # IMSHOW DEL MODULO DEL CAMPO DI PRESSIONE
    acoustic_field_xz = np.abs(P[s:-s, P.shape[0]//2, s:-s].T)
    plt.imshow(acoustic_field_xz/np.amax(acoustic_field_xz),
               extent=_extent, cmap='viridis')
    plt.title(r"$P(\vec{r}\ ; T="+str(int(T))+"°C)$", fontsize=14)
    plt.colorbar()
    plt.xlabel('x [cm]')
    plt.ylabel('z [cm]')
    plt.minorticks_on()
    # plt.grid(True, which='both')
    plt.savefig("/home/leo/Scrivania/unimi/lab-ottica/TESI3-ICE-OPT/tesi/presentazione/nodi/2d/pressure/nuovo/p_temperature_"+str(int(T))+".png")
    plt.show()

    # IMSHOW DEL POTENZIALE DI GORK'OV

    k1 = V/4 * (cp**2 * rhop - c0**2 * rho0)/(c0**2 * cp**2 * rho0 * rhop)
    k2 = 3*V/4 * (rho0 - rhop)/(omega**2 * rho0 * (rho0 + 2*rhop))

    U = 2*k1*(np.abs(P)**2) - 2*k2*np.linalg.norm(np.gradient(P))**2

    potential = U[s:-s, U.shape[0]//2, s:-s].T
    plt.imshow(potential/np.amax(potential), extent=_extent, cmap='viridis')
    # plt.title(r"Gork'ov potential $U(\vec{r})$", fontsize=14)
    plt.colorbar()
    plt.xlabel('x [cm]')
    plt.ylabel('z [cm]')
    plt.minorticks_on()
    plt.tight_layout()
    # plt.grid(True, which='both')
    # plt.savefig("/home/leo/Scrivania/unimi/TESI3-ICE-OPT/tesi/presentazione/nodi/2d_1/potential/u_temperature_"+str(T)+".png")
    
    # # IMSHOW DEL CAMPO VETTORIALE F
    # Definisci la matrice del potenziale di forza U (100x100)
    U = potential
    
    
    xmin = _extent[0]
    xmax = _extent[1]

    # Genera griglie di coordinate x e y dense
    x_dense = np.linspace(xmin, xmax, potential.shape[0])
    y_dense = np.linspace(xmin, xmax, potential.shape[0])

    # Genera griglie di coordinate X e Y
    X_dense, Y_dense = np.meshgrid(x_dense, y_dense)

    # Calcola il gradiente del potenziale di forza utilizzando np.gradient
    Fy_dense, Fx_dense = np.gradient(-U, (y_dense[1]-y_dense[0]), (x_dense[1]-x_dense[0]))
    
    # Sottocampiona il campo di forze risultante per visualizzarlo con meno frecce
    stride = 2
    X = X_dense[::stride, ::stride]
    Y = Y_dense[::stride, ::stride]
    Fx = Fx_dense[::stride, ::stride]
    Fy = Fy_dense[::stride, ::stride]
    
    # Visualizza la mappa del campo di forze
    plt.quiver(X, Y, Fx, Fy)
    plt.show()

    
    

    # STIMA DELLA DIMENSIONE DEI NODI

    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False)

    # xmin = _extent[0]
    # xmax = _extent[1]
    # x = np.linspace(xmin, xmax, potential.shape[0])
    y1 = potential[:, potential.shape[0]//2]
    sections_x.append(y1)

    # ax1.plot(x,y1/np.amax(y1))
    # ax1.set_xlabel('z [cm]')
    # ax1.minorticks_on()
    # ax1.set_xlim(-1.5,1.5)
    # ax1.set_ylim(-0.1,1.1)
    # ax1.grid(True, which='both')

    # massimi = argrelextrema(y1, np.less)[0]
    # z_massimi = x[massimi]
    # dim_nodi = np.diff(z_massimi)
    # print("dimensione dei nodi in z:" + str(dim_nodi))

    # #ax1.legend()

    y2 = potential[potential.shape[0]//2, :]
    sections_z.append(y2)
    # ax2.plot(x,y2/np.amax(y2), label='T[°C]='+str(round(T,1)))
    # ax2.set_xlabel('x [cm]')
    # #plt.ylabel(r'$U(0,y)$', rotation=0)
    # ax2.minorticks_on()
    # ax2.set_xlim(-1.5,1.5)
    # ax2.set_ylim(-0.1,1.1)

    # massimi_1 = argrelextrema(y2, np.less)[0]
    # x_massimi = x[massimi_1]
    # dim_nodi_1 = np.diff(x_massimi)
    # print("dimensione dei nodi in x:" + str(dim_nodi_1))
    # ax2.legend()

    # ax2.grid(True, which='both')

    # #plt.suptitle("T="+str(T))
    # plt.tight_layout()

    # plt.savefig("/home/leo/Scrivania/unimi/TESI3-ICE-OPT/tesi/presentazione/nodi/2d_1/1d_nodi/dim_nodi_temperature"+str(T)+".png")

    # plt.show()


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, sharey=False)

# inizializzo matrice che contiente tutte le distanze tra i picchi: matrice delle dimensioni dei nodi
matrice = None
# MOSTRO 7 PICCHI (6 DISTANZE TRA I PICCHI)
for i, T in enumerate(TT):

    print("#######################################")
    print("T = "+str(T))
    xmin = _extent[0]
    xmax = _extent[1]
    x = np.linspace(xmin, xmax, potential.shape[0])

    y = sections_x[i]

    cs = CubicSpline(x, y)

    # Nuovi punti x per la curva smooth
    x_smooth = np.linspace(x.min(), x.max(), 1000)

    # Valori y smooth
    y_smooth = cs(x_smooth)  # /np.amax(cs(x_smooth))
    
    
    a = -2
    b = 2
    ax1.plot(x_smooth, y_smooth)
    # ax1.plot(x,y/np.amax(y))
    ax1.set_xlabel('z [cm]')
    ax1.set_ylabel(r'$U(0,0,z)$', rotation=90)
    ax1.minorticks_on()
    ax1.set_xlim(a, b)
    # ax1.set_ylim(-0.1,1.1)
    ax1.grid(True, which='both')

    massimi = find_peaks(y_smooth)[0]
    z_massimi = x_smooth[massimi]

    
    maxima_in_interval = [z for z in z_massimi if a <= z <= b]
    dim_nodi = np.diff(maxima_in_interval)

    print(
        "Massimi relativi nell'intervallo [", a, ",", b, "]:", maxima_in_interval)
    print("dimensione dei nodi in z:" + str(dim_nodi))

    if matrice is None:
        matrice = dim_nodi.reshape(-1, 1)
    else:
        print(dim_nodi.shape)
        matrice = np.column_stack((matrice, dim_nodi))

    y = sections_z[i]

    cs = CubicSpline(x, y)

    # Valori y smooth
    y_smooth = cs(x_smooth)  # /np.amax(cs(x_smooth))

    ax2.plot(x_smooth, y_smooth, label='T[°C]='+str(round(T, 1)))
    # ax2.plot(x,y/np.amax(y), label='T[°C]='+str(round(T,1)))
    ax2.set_xlabel('x [cm]')
    ax2.set_ylabel(r'$U(x,0,0)$', rotation=90)
    ax2.legend()
    ax2.minorticks_on()
    ax2.set_xlim(a, b)
    # ax2.set_ylim(-0.1,1.1)
    ax2.grid(True, which='both')

    massimi_1 = find_peaks(y_smooth)[0]
    x_massimi = x_smooth[massimi_1]

    maxima_in_interval_1 = [x for x in x_massimi if a <= x <= b]
    dim_nodi_1 = np.diff(maxima_in_interval_1)

    print("Massimi relativi nell'intervallo [", a, ",", b, "]:", maxima_in_interval_1)
    print("dimensione dei nodi in x:" + str(dim_nodi_1))

    plt.tight_layout()


plt.show()

for i in range(matrice.shape[0]):
    plt.plot(TT,matrice[i,:], label="i="+str(i))
    
plt.xlabel('T [°C]')
plt.ylabel(r'$\Delta z [cm]$')

# Aggiungi una legenda
plt.legend()

# Mostra il grafico
plt.show()