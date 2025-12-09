#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import holopy as hp, matplotlib
import matplotlib.pylab as plt
from scipy.signal import argrelextrema
from function_auxiliary import cut_image
from termcolor import colored
from PIL import Image
from holopy.core.metadata import data_grid

from function_auxiliary import ProgressBar
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


##########################################################################################################
import gc # Garbage Collector per forzare la pulizia RAM

# Funzione repropagate_fast, meno leggibile di repropagate_fast, ma molto piu veloce (il trucco è ripropagare un libro di <batch_size> immagini e non una immagine singola alla volta, perche holopy lo gestisce piu velocemente)
def repropagate_fast(hologram, start_zrange, end_zrange, steps, 
                pixel_size, medium_index, illum_wavelen,
                save_label, save_path, save_format,
                incremental_plot_label=False, incremental_save_label=False,
                plot_label=False, # <- per non rallentare troppo 
                batch_size=5, verbose=False):
    
    # Oss: in questa funzione incremental_plot_label e incremental_save_label non sono neache definite, ma le lascio tra gli argomenti cosi posso intercambiare l'utilizza di repropagate e repropagate_fast senza dover starmi a preoccupare del fatto che non vogliono gli stessi identici argomenti
    
    # 1. Genera tutti i valori Z
    z_all = np.linspace(start_zrange, end_zrange, steps)
    
    # 2. Prepara Ologramma
    hologram_grid = data_grid(hologram, spacing=pixel_size, 
                              medium_index=medium_index, illum_wavelen=illum_wavelen)

    if verbose:
        print(f"Propagazione a lotti (Batch size: {batch_size})...")

    # Variabili per tracciare il vincitore globale
    global_max_var = -1.0
    best_recon = None
    holo_variance_list = [] # Qui accumuliamo le varianze di tutti i batch

    # 3. CICLO SUI BATCH (es. 0-5, 5-10, 10-15...)
    for i in range(0, steps, batch_size):
        
        # Seleziona il sotto-gruppo di Z (es. z[0:5])
        z_batch = z_all[i : i + batch_size]
        current_steps = len(z_batch) # Potrebbe essere meno di batch_size all'ultimo giro
        
        if verbose:
            print(f"Processing batch {i} -> {i+current_steps} / {steps}...", end='\r')

        # --- PROPAGAZIONE VETTORIZZATA (Solo per questo gruppetto) ---
        # HoloPy calcola solo questi 5 piani -> RAM sotto controllo
        reconstructions = hp.propagate(hologram_grid, z_batch, 
                                       illum_wavelen=illum_wavelen, 
                                       medium_index=medium_index)
        
        # Conversione sicura in numpy e intensità
        if hasattr(reconstructions, 'values'):
            rec_np = np.abs(reconstructions.values)**2
        else:
            rec_np = np.abs(reconstructions)**2

        # Capire gli assi (HoloPy a volte inverte Z)
        # Se shape[0] corrisponde al numero di Z nel batch, allora Z è l'asse 0
        if rec_np.shape[0] == current_steps:
            batch_vars = np.var(rec_np, axis=(1, 2))
            z_axis = 0
        else:
            batch_vars = np.var(rec_np, axis=(0, 1))
            z_axis = 2
        
        # Aggiungi le varianze alla lista globale
        holo_variance_list.extend(batch_vars)

        # --- CACCIA AL TESORO (Trova il migliore in questo batch) ---
        local_max_idx = np.argmax(batch_vars)
        local_max_val = batch_vars[local_max_idx]

        if local_max_val > global_max_var:
            global_max_var = local_max_val
            
            # Estraiamo SOLO il piano vincente per salvarlo
            if z_axis == 0:
                best_recon = rec_np[local_max_idx, :, :].copy()
            else:
                best_recon = rec_np[:, :, local_max_idx].copy()
                
        # --- PULIZIA RAM FONDAMENTALE ---
        # Cancelliamo il batch pesante appena elaborato
        del reconstructions
        del rec_np
        gc.collect() # Forza Python a liberare la RAM subito

    if verbose: 
        print("\nPropagazione completata.")
    
    # 4. Analisi Finale
    holo_variance_list = np.array(holo_variance_list)
    
    # Trova indice del massimo globale
    idx = argrelextrema(holo_variance_list, np.greater)[0]
    # L'indice del massimo assoluto
    max_idx = np.argmax(holo_variance_list)

    # Stampa solo se verbose è True
    if verbose:
        print(f"Propagazione a lotti (Batch size: {batch_size})...")

    # ... (ciclo for dei batch identico a prima) ...
    for i in range(0, steps, batch_size):
        if verbose: # Stampa solo se richiesto
            print(f"Processing batch {i} -> {i+current_steps} / {steps}...", end='\r')

    if verbose: 
        print("\nPropagazione completata.")
    
    if plot_label:
        fig = plt.figure() # Assegna a una variabile 'fig'
        plt.scatter(z_all/10000, holo_variance_list, c=holo_variance_list, cmap='jet', marker='^')
        plt.axvline(z_all[max_idx]/10000, color='k', ls='--')
        plt.title('Batched Variance Scan')
        
        if save_label: 
            plt.savefig(save_path +'/variance_graph'+save_format)
            
        # Se stai processando uno stack, probabilmente NON vuoi fare plt.show() 
        # perché bloccherebbe il codice 65 volte. 
        # Se proprio vuoi vederlo, lascia plt.show(), ma per pulizia meglio:
        plt.show()
        plt.close(fig) # <--- QUESTO RIMUOVE LA SCRITTA <Figure size...>

    # Ritorna l'immagine 2D migliore e i dati
    return best_recon, holo_variance_list, z_all/10000, idx, max_idx
    

# Funzione repropagate, piu leggibile di repropagate_fast, ma molto piu lenta
def repropagate(hologram, start_zrange, end_zrange, steps, 
                pixel_size, medium_index, illum_wavelen, plot_label,
                save_label, save_path, save_format,
                incremental_plot_label=False, incremental_save_label=False,
                batch_size=1, verbose = False):
    
    
    # oss: in queta funzione batch size è inutile, ma la lascio cosi posso cambiare velocemente tra repropagate e repropagate_fast
    z = np.linspace(start_zrange, end_zrange, steps)
    holo_propagation_list, holo_variance_list = [], []
    
    hologram = data_grid(hologram, spacing=pixel_size, medium_index=medium_index, illum_wavelen=illum_wavelen)

    for k in range(0, len(z)):
        if verbose:
            ProgressBar(1 - (len(z) - (k+1))/len(z))
        holo_propagation = hp.propagate(hologram, z[k],                         
                                        illum_wavelen = illum_wavelen,          
                                        medium_index = medium_index)
        
        holo_propagation = np.abs(holo_propagation[:, :, 0])**2
        holo_variance = np.var(holo_propagation)
        holo_variance_list.append(holo_variance)
        holo_propagation_list.append(holo_propagation)

        if incremental_plot_label==True:
            plt.imshow(holo_propagation/np.amax(holo_propagation))
            plt.title('Image rec @ '+'{:.03f}'.format(z[k]/10000)+' cm')
            plt.xlabel('x [pxl]')
            plt.ylabel('y [pxl]')
            plt.clim(0, 1)
            plt.colorbar()
            plt.tight_layout()
            if incremental_save_label == True:
                if not os.path.exists(save_path+'/incremental_plot/'): os.makedirs(save_path+'/incremental_plot/')
                plt.savefig(save_path +'/incremental_plot/image_rec_'+'{:.03f}'.format(z[k]/10000)+'_cm'+save_format)
            plt.show()
            
            plt.clf()

    holo_variance_list = np.array(holo_variance_list)
    
    
    idx = argrelextrema(holo_variance_list*100, np.greater)[0]                  #lista dei massimi relativi di holo_var_list
    # faccio per cento perche secondo me argrelextrema senno pensa che la lista sia vuota
    
    var_max_values = holo_variance_list[idx]
    try: 
        max_idx = idx[np.argmax(var_max_values)] 
    except ValueError: 
        max_idx = steps//2
        print("Non sono stati trovati massimi relativi: idx = [].")
        print("Ricostruisco a z="+str(end_zrange+start_zrange//20000) +" cm")
        print("max_idx = "+str(max_idx))               
       
    
    plt.scatter(z/10000, holo_variance_list*1000, s=10,
                c=np.array(holo_variance_list), cmap='jet', marker='^')     
    z_max = z[max_idx]/10000
    plt.axvline(z_max, color='black', ls='-.', lw=1)
    # plt.text(z_max - 0.2, plt.ylim()[0] + 0.3, f'z={z_max:.3f}', fontsize=12)
    plt.minorticks_on()
    plt.grid()
    plt.xlabel('z [cm]')                                                    
    plt.ylabel('$\sigma^2$ [a. u.] $\cdot$ 10$^3$')
    try: plt.title('Image '+str(save_path[-6:])+' variance')
    except: plt.title('Image variance')
    #plt.grid()
    if save_label==True: 
        plt.savefig(save_path +'/variance_propagation'+save_format)
    if plot_label==True: 
        plt.show()
    plt.clf()
    
    return holo_propagation_list[max_idx], holo_variance_list, z/10000, idx, max_idx
    
def hologram_analysis(image, img_path, pixel_size, medium_index, illum_wavelen,
                      var_bkg, avg_background, number_of_centers, center_threshold, 
                      center_blursize, dim, off_x, off_y,
                      start_zrange, end_zrange, steps, center_find_label, plot_label, 
                      save_label, save_path, save_format, output_file, img_limits, 
                      incremental_plot_label, incremental_save_label):


    raw_holo = hp.load_image(img_path+'/'+image,                   # Carico le immagini di ologrammi
                             spacing = pixel_size,
                             medium_index = medium_index,
                             illum_wavelen = illum_wavelen)
    
    
    holo_cut = cut_image(raw_holo, number_of_centers, center_threshold,
                          center_blursize, dim, off_x, off_y, img_limits, center_find_label)
    

    #holo_cut[0,:,:] = (holo_cut[0,:,:] - np.amin(holo_cut[0,:,:]) )/(np.amax(holo_cut[0,:,:]) - np.amin(holo_cut[0,:,:]) )*255    

    
    norm = sum(sum(holo_cut[0, :, :]))
       
    holo_cut = holo_cut/norm
    # print("norm holo =" + str(norm))
    # print("sum holo = "+str(sum(sum(holo_cut[0, :, :]))))
    
                                                             # x sta in [0, 255]
    
    
    # norm1 = sum(sum(avg_background))
    # avg_background = avg_background/norm1
    
    
    holo_cut = 1 - (holo_cut)/(avg_background)              # sottraggo background (~ variazione %)
    x = holo_cut[0,:,:]                                                         # è x l'immagine dell'ologramma
    x = np.array(x)                                  
    x = (x - np.amin(x) )/(np.amax(x) - np.amin(x) )*255          
    
    img_holo_cut = Image.fromarray(x.astype('uint8'))                         # immagine(x)
    
    
    
    plt.imshow(holo_cut[0,:,:], cmap='viridis')                    # Show an initial image first
    plt.xlabel('x [pxl]')                            # per verificare che sia ben centrato
    plt.ylabel('y [pxl]')
    try: plt.suptitle('Image #'+str(int(image[-11:-4])))
    except: plt.suptitle(str(image))
    plt.colorbar()
    plt.tight_layout()
    ########################### SAVE HOLO_CUT ################################
    if save_label==True: 
        plt.savefig(save_path + image[:-4] + '/hologram_cut' + save_format)
        plt.savefig(save_path + 'xgif/holo_cut/'+str(image[+6:-4]) + save_format)
        img_holo_cut.save(save_path+'xgif/holo_cut/HOLO_'+str(image[+6:-4]) + ".tif")
        #img_holo_cut.show()
        ##########################################################################
    if plot_label==True:    
        plt.show()
    plt.clf()
    
            
        

       

    output_file.write('\nImage number:\t\t\t'+image[6:]+
                      '\nNumber of NULL pixels:\t\t'+str(len(np.where(raw_holo[0, :, :] == 0.0)[0]))+
                      '\nNumber of saturated pixels:\t'+str(len(np.where(raw_holo[0, :, :] == 255.0)[0]))+
                      '\nMinimum image value:\t\t'+str(int(raw_holo[0, :, :].min()))+
                      '\nMaximum image value:\t\t'+str(int(raw_holo[0, :, :].max()))+
                      '\nAverage image intensity:\t'+'{:.3f}'.format(float(np.mean(raw_holo[0, :, :])))+
                      '\nImage standard deviation:\t'+'{:.3f}'.format(float(raw_holo[0, :, :].std()))+
                      '\n####################################################################')

    print(colored('\nImage number:\t\t\t\t\t', 'green')+image[6:]+
                      colored('\nNumber of NULL pixels:\t\t\t', 'green')+str(len(np.where(raw_holo[0, :, :] == 0.0)[0]))+
                      colored('\nNumber of saturated pixels:\t\t', 'green')+str(len(np.where(raw_holo[0, :, :] == 255.0)[0]))+
                      colored('\nMinimum image value:\t\t\t', 'green')+str(int(raw_holo[0, :, :].min()))+
                      colored('\nMaximum image value:\t\t\t', 'green')+str(int(raw_holo[0, :, :].max()))+
                      colored('\nAverage image intensity:\t\t', 'green')+'{:.3f}'.format(float(np.mean(raw_holo[0, :, :])))+
                      colored('\nImage standard deviation:\t\t', 'green')+'{:.3f}'.format(float(raw_holo[0, :, :].std()))+
                      '\n\n####################################################################')


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


    propagation_list, var_list, z, idx, max_idx = image_reconstruction(holo_cut,
                                                              start_zrange,
                                                              end_zrange,
                                                              steps,
                                                              illum_wavelen,
                                                              medium_index,
                                                              plot_label,
                                                              incremental_plot_label,
                                                              incremental_save_label,
                                                              save_label,
                                                              save_path+image[:-4], save_format)

    return holo_cut, z, propagation_list, idx, max_idx, var_list


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


def image_reconstruction(holo_cut, start_zrange, end_zrange, steps, illum_wavelen,
                         medium_index, plot_label, incremental_plot_label, incremental_save_label, save_label,
                         save_path, save_format):

    z = np.linspace(start_zrange, end_zrange, steps)
    holo_propagation_list, holo_variance_list = [], []


    for k in range(0, len(z)):
        holo_propagation = hp.propagate(holo_cut, z[k],                         
                                        illum_wavelen = illum_wavelen,          
                                        medium_index = medium_index)
        
        holo_propagation = np.abs(holo_propagation[:, :, 0])**2
        holo_variance = np.var(holo_propagation)
        holo_variance_list.append(holo_variance)
        holo_propagation_list.append(holo_propagation)

        if incremental_plot_label==True:
            plt.imshow(holo_propagation/np.amax(holo_propagation))
            plt.title('Image rec @ '+'{:.03f}'.format(z[k]/10000)+' cm')
            plt.xlabel('x [pxl]')
            plt.ylabel('y [pxl]')
            plt.clim(0, 1)
            plt.colorbar()
            plt.tight_layout()
            if incremental_save_label == True:
                if not os.path.exists(save_path+'/incremental_plot/'): os.makedirs(save_path+'/incremental_plot/')
                plt.savefig(save_path +'/incremental_plot/image_rec_'+'{:.03f}'.format(z[k]/10000)+'_cm'+save_format)
            plt.show()
            
            plt.clf()

    holo_variance_list = np.array(holo_variance_list)
    
    
    idx = argrelextrema(holo_variance_list*100, np.greater)[0]                  #lista dei massimi relativi di holo_var_list
    # faccio per cento perche secondo me argrelextrema senno pensa che la lista sia vuota
    
    var_max_values = holo_variance_list[idx]
    try: 
        max_idx = idx[np.argmax(var_max_values)] 
    except ValueError: 
        max_idx = steps//2
        # print("Non sono stati trovati massimi relativi: idx = [].")
        # print("Ricostruisco a z="+str(end_zrange+start_zrange//20000) +" cm")
        # print("max_idx = "+str(max_idx))               
       
    
    plt.scatter(z/10000, holo_variance_list*1000, s=10,
                c=np.array(holo_variance_list), cmap='jet', marker='^')     
    x_coordinate = z[max_idx]/10000
    plt.axvline(x=z[max_idx]/10000, color='black', ls='-.', 
                    lw=1)
    # plt.text(x_coordinate - 0.2, plt.ylim()[0] + 0.3, f'z={x_coordinate:.3f}', fontsize=12)
    plt.minorticks_on()
    plt.grid()
    plt.xlabel('z [cm]')                                                    
    plt.ylabel('$\sigma^2$ [a. u.] $\cdot$ 10$^3$')
    try: plt.title('Image '+str(save_path[-6:])+' variance')
    except: plt.title('Image variance')
    #plt.grid()
    if save_label==True: 
        plt.savefig(save_path +'/variance_propagation'+save_format)
    if plot_label==True: 
        plt.show()
    plt.clf()
    
        

    return holo_propagation_list, holo_variance_list, z/10000, idx, max_idx


###############################################################################################################
