#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 9 13:25:41 2022

@author: Luca Teruzzi
"""


############################################################################################################################
############################################################################################################################


import os
import numpy as np, matplotlib.pyplot as plt
import cv2
import imutils
from pylab import *
from PIL import Image
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours


############################################################################################################################
############################################################################################################################



def midpoint(ptA, ptB):

    """
    Calculates the middle point of two point

    Args
    -------------------------------------------------------
    ptA (int):
       Position A
    ptB (int):
       Position B

    Returns
    --------------------------------------------------------
    The middle point: (float)
    """

    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#


def maximum_minimum(array, z):

    """
    Calculates the max and min value of an array

    Args
    -------------------------------------------------------
    array (float or list of floats):
        Array of which you want calculate the extremes
    z (float or list of floats):
        Distance to propagate

    Returns
    --------------------------------------------------------
    d_max (float):
        Distance[pixels] at which the array have the maximum value
    d_min (float):
        Distance[pixels] at which the array have the minimun value
    z_max (int):
        Array position at which the array have the maximum value
    z_min (int):
        Array position at which the array have the minium value
    """

    max_array = np.amax(array)
    d_max = z[np.where(array == max_array)[0]]
    z_max = np.where(array == max_array)[0]

    min_array = np.amin(array)
    d_min = z[np.where(array == min_array)[0]]
    z_min = np.where(array == min_array)[0]

    return d_max, d_min, z_max, z_min


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#


def plot_twin_propagation(z, module_arr, phase_arr, save_label, save_path, save_format, plot_label=False):

    """
    Calculates the plot of the propagation of the hologram along the optical
    axis and at the center of the hologram both studying the module and the phase of the field.

    Args
    -------------------------------------------------------
    z (float or list of floats):
        Distance to propagate
    module_arr (float or list of floats):
        Array of the intensity of the field propagated
    phase_arr (float or list of floats):
        Array of the phase of the field propagated

    Returns
    -------------------------------------------------------
    0: the graph is saved authomatically.
        By the graph the point of discontinuity can be seen and it can be possible
        calculate the z position of the particle
    """

    fig, ax1 = plt.subplots()
    ax1.plot(z, module_arr, '-b', label='module')

    ax1.set_xlabel('z [cm]')
    ax1.set_ylabel('|U|', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    phase_arr[phase_arr == 0] = None
    ax2.plot(z, phase_arr, '-r', label='phase')
    ax2.set_ylabel('$\phi$(U)', color='r')
    ax2.tick_params('y', colors='r')

    plt.title("Field module and phase propagation")
    plt.tight_layout()
    if save_label==True: plt.savefig(save_path+'module_and_phase'+save_format)
    if plot_label == True: plt.show()
    plt.clf()

    return 0


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#


def object_dimension(path, pixel_size, area, z, save_path, save_format):

    """
    Calculates the diameters of an object, not circular.

    It first performs edge detection, then performs a dilation + erosion to
    close gaps in between object edges.
    Then for each object in the image, it calcaluates the contourns of the
    minimum box (minimum rectangle that circumvent the object) and it sorts
    them from left-to-right (allowing us to extract our reference object).
    It unpacks the ordered bounding box and computes the midpoint between the
    top-left and top-right coordinates, followed by the midpoint between
    bottom-left and bottom-right coordinates.
    Finally it computes the Euclidean distance between the midpoints.

    Args
    -------------------------------------------------------
    path:
       Path of the directory of the image of the object reconstructed at the
       focal point.
    pixel_size (float):
        Value of the pixel size (um)

    Returns
    -------------------------------------------------------
    dimS (float):
        Value of the the smaller diameter
    dimL (float):
        Value of the the longer diameter
    ratio (float):
        Value of the ratio of the two diameters
    """
    #print(path)
    image = cv2.imread(path)
    #print(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # perform the countour
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    # find contours in the edge map
    cnts = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    dimUno = np.array([])
    dimDue = np.array([])
    ratio_array = np.array([])

    if cnts != []:
        # sort the contours from left-to-right
        (cnts, _) = contours.sort_contours(cnts)
        n = 0
        dimUno = np.array([])
        dimDue = np.array([])
        ratio_array = np.array([])

        for c in cnts:
            # compute the rotated bounding box of the contour
            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(
                box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
        # order the points in the contour such that they appear in top-left
            box = perspective.order_points(box)
        # Compute the midpoint
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            dx = np.abs(tltrX-blbrX)
            dy = np.abs(tlblY-trbrY)
        # draw lines between the midpoints
            cv2.line(orig, (int(tltrX), int(tltrY)),
                     (int(blbrX), int(blbrY)), (5, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)),
                     (int(trbrX), int(trbrY)), (255, 0, 255), 2)
        # compute the Euclidean distance between the midpoints
        # dA  variable will contain the height distance (pixels)
        # dB  will hold our width distance (pixels).
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            if dA != 0 and dB != 0:
                # compute the size of the object
                dimA = dA * pixel_size
                dimB = dB * pixel_size

                diff = dA - dB
                if diff < 0:
                    ratio = dA/dB
                    dimS = dimA
                    dimL = dimB
                else:
                    ratio = dB/dA
                    dimS = dimB
                    dimL = dimA

                cv2.putText(orig, "{:.1f}um".format(dimA), (int(
                    tltrX - 5), int(tltrY - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (100, 100, 100), 1)
                cv2.putText(orig, "{:.1f}um".format(dimB), (int(
                    trbrX + 8), int(trbrY + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (100, 100, 100), 1)
                result = Image.fromarray((orig).astype('uint8'))
                
                #print("max tra dimA e dimB:\t\t" + str(max(dimA, dimB)))
                
                if max(dimA, dimB)>=200: result.save(save_path+'rec_'+'{:.03f}'.format(z)+'_cm_dimensions_'+str(n)+'.tif')

            n = n+1
            dimUno = np.append(dimUno, dimS)
            dimDue = np.append(dimDue, dimL)
            ratio_array = np.append(ratio_array, ratio)
    else:
        dimS = 0
        dimL = 0
        ratio = 0
        dx = 0
        dy = 0
        dimUno = np.append(dimUno, dimS)
        dimDue = np.append(dimDue, dimL)
        ratio_array = np.append(ratio_array, ratio)

    return dimUno, dimDue, ratio_array, dx, dy




def object_dimension16(image, pixel_size, area, z, save_label, save_path, save_format):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Normalizzazione Intelligente da 16 a 8 bit
    # Se l'immagine non è vuota, normalizza per massimizzare il contrasto
    if gray.max() > gray.min():
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    else:
        # Se l'immagine è piatta (tutta nera o tutta grigia), normalizzare darebbe errore o rumore
        # In questo caso scala semplicemente
        pass 
        
    gray = np.uint8(gray)
    
    # Applica un filtro di sfocatura gaussiana
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # Applica il rilevatore di bordi Canny
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    
    # Trova i contorni nella mappa dei bordi
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    dimUno = np.array([])
    dimDue = np.array([])
    ratio_array = np.array([])

    if cnts:
        # Ordina i contorni da sinistra a destra
        cnts, _ = contours.sort_contours(cnts)
        n = 0

        for c in cnts:
            # Calcola il rettangolo ruotato del contorno
            # Converti l'immagine in BGR (3 canali) per disegnare linee colorate
            if len(image.shape) == 2:
                orig = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
            else:
                orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box) if imutils.is_cv3() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            
            # Calcola i punti medi
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            dx = np.abs(tltrX - blbrX)
            dy = np.abs(tlblY - trbrY)
            
            # Disegna linee tra i punti medi con colori accesi e spessore maggiore
            # dimA: rosso acceso (BGR: 0, 0, 255) - linea verticale
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (0, 0, 255), 5)
            # dimB: blu acceso (BGR: 255, 0, 0) - linea orizzontale
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 0), 5)
            
            # Calcola la distanza euclidea tra i punti medi
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            if dA != 0 and dB != 0:
                # Calcola la dimensione dell'oggetto
                dimA = dA * pixel_size
                dimB = dB * pixel_size

                diff = dA - dB
                if diff < 0:
                    ratio = dA / dB
                    dimS = dimA
                    dimL = dimB
                else:
                    ratio = dB / dA
                    dimS = dimB
                    dimL = dimA

                # Testo grigio chiaro (BGR: 200, 200, 200), non bianco
                text_color = (200, 200, 200)
                cv2.putText(orig, "{:.1f}um".format(dimA), (int(tltrX - 5), int(tltrY - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)
                cv2.putText(orig, "{:.1f}um".format(dimB), (int(trbrX + 8), int(trbrY + 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)
                result = Image.fromarray(orig.astype('uint8'))
                
                if save_label:
                    if max(dimA, dimB) >= 200:
                        # Crea sottocartella rec_{z}_cm dentro save_path
                        rec_subdir = f"rec_{z:.03f}_cm"
                        rec_path = os.path.join(save_path, rec_subdir)
                        os.makedirs(rec_path, exist_ok=True)
                        # Assicurati che rec_path finisca con /
                        rec_path_fixed = rec_path if rec_path.endswith(os.sep) else rec_path + os.sep
                        result.save(f"{rec_path_fixed}dimensions_{n}.{save_format}")

                n += 1
                dimUno = np.append(dimUno, dimS)
                dimDue = np.append(dimDue, dimL)
                ratio_array = np.append(ratio_array, ratio)
    else:
        dimS = 0
        dimL = 0
        ratio = 0
        dx = 0
        dy = 0
        dimUno = np.append(dimUno, dimS)
        dimDue = np.append(dimDue, dimL)
        ratio_array = np.append(ratio_array, ratio)

    return dimUno, dimDue, ratio_array, dx, dy


######################################################################################################################
def object_dimension2(image, pixel_size, area, z, save_label, save_path, save_format):


    gray = cv2.GaussianBlur(image, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    # find contours in the edge map
    cnts = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    dimUno = np.array([])
    dimDue = np.array([])
    ratio_array = np.array([])

    if cnts != []:
        # sort the contours from left-to-right
        (cnts, _) = contours.sort_contours(cnts)
        n = 0
        dimUno = np.array([])
        dimDue = np.array([])
        ratio_array = np.array([])

        for c in cnts:
            # compute the rotated bounding box of the contour
            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(
                box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
        # order the points in the contour such that they appear in top-left
            box = perspective.order_points(box)
        # Compute the midpoint
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            dx = np.abs(tltrX-blbrX)
            dy = np.abs(tlblY-trbrY)
        # draw lines between the midpoints
            cv2.line(orig, (int(tltrX), int(tltrY)),
                     (int(blbrX), int(blbrY)), (5, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)),
                     (int(trbrX), int(trbrY)), (255, 0, 255), 2)
        # compute the Euclidean distance between the midpoints
        # dA  variable will contain the height distance (pixels)
        # dB  will hold our width distance (pixels).
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            if dA != 0 and dB != 0:
                # compute the size of the object
                dimA = dA * pixel_size
                dimB = dB * pixel_size

                diff = dA - dB
                if diff < 0:
                    ratio = dA/dB
                    dimS = dimA
                    dimL = dimB
                else:
                    ratio = dB/dA
                    dimS = dimB
                    dimL = dimA

                cv2.putText(orig, "{:.1f}um".format(dimA), (int(
                    tltrX - 5), int(tltrY - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (100, 100, 100), 1)
                cv2.putText(orig, "{:.1f}um".format(dimB), (int(
                    trbrX + 8), int(trbrY + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (100, 100, 100), 1)
                result = Image.fromarray((orig).astype('uint8'))
                
                #print("max tra dimA e dimB:\t\t" + str(max(dimA, dimB)))
                
                if save_label == True:
                    if max(dimA, dimB)>=200: result.save(save_path+'rec_'+'{:.03f}'.format(z)+'_cm_dimensions_'+str(n)+'.tif')

            n = n+1
            dimUno = np.append(dimUno, dimS)
            dimDue = np.append(dimDue, dimL)
            ratio_array = np.append(ratio_array, ratio)
    else:
        dimS = 0
        dimL = 0
        ratio = 0
        dx = 0
        dy = 0
        dimUno = np.append(dimUno, dimS)
        dimDue = np.append(dimDue, dimL)
        ratio_array = np.append(ratio_array, ratio)

    return dimUno, dimDue, ratio_array, dx, dy