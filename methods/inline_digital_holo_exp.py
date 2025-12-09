#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:35:24 2023

@author: leo
"""
import numpy as np
from math import atan2, sqrt
# from aux_physics import cart2sph

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


# class object:
    
#     def __init__(self, x=0, y=0, z=0):
    
#     def hologram():
#         # object.hologram() -> holo (np.array)
    
#     def avg_hologram():
#         #obj.avg_holo() -> avg_holo = [ holograms ]


class Transducer:
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

    def P(self, R_sphere, dens=100):

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


class IceOptInlineDigitalHoloExp:

    def __init__(self, illum_wavelen=0.6335, medium_index=1.00027, f=76.2, q=60.0, v_levitator=None, T=293):
        
        self.illum_wavelen = illum_wavelen                              # lunghezza d'onda laser
        self.medium_index = medium_index                                # indice rifrazione
        self.k = np.pi*2/(illum_wavelen/medium_index)                   # numero d'onda
        self.f = f                                                      # focale del secondo specchio
        self.q = q                                                      # distanza ccd-specchio
        self.p = f*q/(np.abs(f-q))                                                          

        self.v_levitator = v_levitator                                  # frequenza levitatore
        self.T = T                                                      # temperature
    
    def show(self):
        print("########### experimental setup: ############")
        print("illum_wavelen = " + str(self.illum_wavelen*1000) + " nm")
        print("medium_index = " + str(self.medium_index))
        print("focal distance: f = " + str(self.f) + " mm")
        print("distance(ccd, parabolic mirror): q = " + str(self.q) + " mm")
        print("levitator frequency: v = " + str(self.v_levitator) + " Hz")
        print("temperature: T = " + str(self.T) + " K")
        print("############################################")
        
    # forse per il levitatore dovrei fare una classe a parte        
        
    def calibrate(self, dimX, dimY, errX, errY):
        print("")
        # restituisce la q t.c. Dx,Dy = 0,0
        #Dx = np.abs(dimA - dimX)
        #Dy = np.abs(dimB - dimY)
    
    # def start_hologram_acquisition():
    
    # def hologram-analysis():
        
    # def simulate():
    #     # esempio: sf = Sphere(x,y,z) ->
    #     #           return var_images, hologram
        

#############################################





