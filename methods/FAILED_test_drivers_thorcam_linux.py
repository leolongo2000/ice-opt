#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 15:58:00 2025

@author: leo
"""

import os
import sys

# Impostiamo il percorso
path_hint = '/usr/local/lib/'
os.environ['THORLABS_TSI_SDK_BIN_PATH_HINT'] = path_hint

print(f"1. Sto cercando i driver in: {path_hint}")

# Verifichiamo manualmente se i file esistono prima di chiamare l'SDK
sdk_file = os.path.join(path_hint, 'libthorlabs_tsi_camera_sdk.so')
if os.path.exists(sdk_file):
    print(f"2. OK: Ho trovato il file principale: {sdk_file}")
else:
    print(f"2. ERRORE: Non trovo il file {sdk_file}!")
    print("   Hai dimenticato di fare 'sudo cp ... /usr/local/lib' ?")

try:
    from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
    print("3. Libreria Python importata correttamente.")
    
    with TLCameraSDK() as sdk:
        print("4. SDK Inizializzato.")
        # Questo comando lista quali moduli DLL/SO sono stati caricati internamente
        # (Non tutte le versioni dell'SDK lo supportano, ma proviamo)
        try:
            print(f"   Versione SDK: {sdk.sdk_version}")
        except:
            pass
            
        print("5. Ricerca camere...")
        cameras = sdk.discover_available_cameras()
        print(f"6. Camere trovate: {len(cameras)}")
        print(cameras)

except Exception as e:
    print(f"\nERRORE CRITICO DURANTE L'ESECUZIONE:\n{e}")