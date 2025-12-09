#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 13:54:46 2025

@author: leo
"""

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
import time

# Questo blocco 'with' gestisce l'apertura e chiusura dell'SDK automaticamente
with TLCameraSDK() as sdk:
    print("Cerco le camere connesse...")
    available_cameras = sdk.discover_available_cameras()
    
    if len(available_cameras) < 1:
        print("NESSUNA CAMERA TROVATA.")
        print("Suggerimento: se è collegata, controlla di aver copiato il file usb.rules in /etc/udev/rules.d/ e di aver riavviato.")
    else:
        print(f"Trovata camera: {available_cameras[0]}")
        
        # Ci connettiamo alla prima camera
        with sdk.open_camera(available_cameras[0]) as camera:
            # Impostiamo un'esposizione di 20 ms (il valore è in microsecondi)
            camera.exposure_time_us = 20000 
            
            # Prepariamo lo scatto (Arm) e scattiamo (Trigger)
            camera.frames_per_trigger_zero_for_unlimited = 1
            camera.arm(2)
            camera.issue_software_trigger()
            
            # Aspettiamo il frame
            frame = camera.get_pending_frame_or_null()
            
            if frame is not None:
                print(f"SUCCESSO! Immagine acquisita.")
                print(f"Dimensioni: {frame.image_buffer.shape}")
                print(f"Valore massimo pixel: {frame.image_buffer.max()}")
                # frame.image_buffer è un array numpy, pronto per essere analizzato o salvato
            else:
                print("Timeout: nessun frame ricevuto.")

print("Fine.")