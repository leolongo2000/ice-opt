#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 01:03:03 2025

@author: leo
"""
# =============================================================================
# IMPORT LIBRERIE
# =============================================================================
import os
import numpy as np
import tifffile
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from function_auxiliary import avg16
from analysis_functions import process_single_image, stack_iterator,merge_temp_files, clean_duplicate_rows_and_sort_results

def get_optimal_core_count(requested_jobs, RAM_perc = 0.5):
    """
    Restituisce il numero di core da usare.
    Impedisce all'utente di selezionare più core di quelli fisicamente disponibili.
    """
    # 1. Conta i core reali del computer
    max_physical_cores = multiprocessing.cpu_count()
    
    # 2. Se l'utente NON ha specificato nulla (-j non usato)
    if requested_jobs is None:
        # Default conservativo: 50% della potenza (lascia il PC usabile)
        return max(1, int(max_physical_cores * RAM_perc))

    # 3. Se l'utente HA specificato un numero (-j N)
    else:
        # Caso A: L'utonto chiede più core di quelli che ha (es. -j 100 su 8 core)
        if requested_jobs > max_physical_cores:
            print(f" BLOCCO SICUREZZA: Hai richiesto {requested_jobs} core, ma la macchina ne ha solo {max_physical_cores}.")
            print(f"   -> Il valore è stato forzato al 50%% del massimo di core disponibili ({max_physical_cores}): ")
            return max(1, int(max_physical_cores * RAM_perc))
        
        # Caso B: L'utente mette 0 o numeri negativi
        elif requested_jobs < 1:
            print(f" ERRORE INPUT: Hai richiesto {requested_jobs} core.")
            print("   -> Il valore è stato forzato al minimo: 1")
            return 1
            
        # Caso C: Richiesta valida, bravo
        else:
            return requested_jobs
    
def init_pool_processes(the_lock):
    """
    Questa funzione viene eseguita all'avvio di ogni processo figlio.
    Serve a condividere il Lock (il pennarello per scrivere alla lavagna (terminale))
    di TQDM tra tutti i processi per evitare che le barre si rompano graficamente.
    """
    tqdm.set_lock(the_lock)
    
def worker_process_frames_chunk(args):
    """
    Analizza un pezzo (chunk) dello stack (Mode SINGLE) con progress bar.
    Processa solo gli indici specificati in chunk_indices (selezione uniforme).
    """
    stack_path, start_idx, end_idx, bkg_path, params, chunk_indices = args
    frame_indices = chunk_indices
    
    try:
        proc_id = multiprocessing.current_process()._identity[0] - 1
    except IndexError:
        proc_id = 0

    # 3. Setup vari
    filename = os.path.basename(stack_path)
    stack_name = os.path.splitext(filename)[0]
    
    try: stack_id = stack_name.split("_")[-1]
    except: stack_id = "0"

    avg_background, _, _ = avg16(bkg_path + '/', plot_label=False)
    
    # Calcola stack_save_path per questo stack
    filename = os.path.basename(stack_path)
    stack_name = os.path.splitext(filename)[0]
    stack_save_path = os.path.join(params['f0_results_path'], stack_name)
    os.makedirs(stack_save_path, exist_ok=True)
    
    results_list = []
    
    try:
        with tifffile.TiffFile(stack_path) as tif:
            
            # PROGRESS BAR ---
            iterator_with_bar = tqdm(
                frame_indices,
                total=len(frame_indices),
                position=proc_id,
                desc=f"Process {proc_id+1} (Fr. {min(frame_indices)}-{max(frame_indices)})", # Descrizione utile
                leave=True, # Non pulisce alla fine
                ncols=100,
                mininterval=0.2
            )
            
            for current_frame in iterator_with_bar:
                try:
                    
                    img_matrix = tif.pages[current_frame].asarray()
                
                    res = process_single_image(
                        img_matrix, 
                        current_frame, 
                        avg_background, 
                        params, 
                        stack_save_path,
                        stack_id
                    )
                    
                    log_str = f"{stack_name}\t{res['frame_idx']}\t{res['z_focus']:.4f}\t{res['area']:.4e}\t{res['dim_A']}\t{res['dim_B']}"
                    results_list.append(log_str)
                    
                except IndexError:
                    break 
                
    except Exception as e:
        tqdm.write(f" Errore worker chunk {start_idx}-{end_idx}: {e}")
        return []

    return results_list


def process_stack_multicore(stack_path, bkg_path, params, N_images_per_stack=None):
  
    # ... Codice conteggio frame  ...
    
    stack_name = os.path.splitext(os.path.basename(stack_path))[0]
    # Path per questo stack: results/nome_campione/f0/nome_stack/
    stack_save_path = os.path.join(params['f0_results_path'], stack_name)
    os.makedirs(stack_save_path, exist_ok=True)
    # results.txt va a livello di f0
    results_csv = params['results_txt_path']


    with tifffile.TiffFile(stack_path) as tif:
        total_frames_in_file = len(tif.pages)
    
    # Seleziona frame uniformemente distribuiti se N_images_per_stack è specificato
    if N_images_per_stack is not None:
        real_N = min(total_frames_in_file, N_images_per_stack)
        # Seleziona indici uniformemente distribuiti
        frame_indices = np.linspace(0, total_frames_in_file - 1, real_N, dtype=int)
        total_frames = len(frame_indices)
    else:
        frame_indices = np.arange(total_frames_in_file)
        total_frames = total_frames_in_file

    # --- CALCOLO CORE ---
    user_jobs = params.get('cpu_count')
    
    num_cores = get_optimal_core_count(user_jobs) 
    
    chunk_size = min( int(np.ceil(total_frames / num_cores)), 10 ) # al massimo 10 frame per core alla volta, di piu no senno esplode tutto
    
    
    # Creazione Tasks: divido gli indici selezionati in chunk
    tasks = []
    for i in range(0, len(frame_indices), chunk_size):
        # Prendo gli indici del chunk corrente dagli indici selezionati
        chunk_indices = frame_indices[i:min(i + chunk_size, len(frame_indices))]
        if len(chunk_indices) > 0:
            start_idx = chunk_indices[0]
            end_idx = chunk_indices[-1] + 1  # +1 perché end è esclusivo nel range (per compatibilità)
            tasks.append((stack_path, start_idx, end_idx, bkg_path, params, chunk_indices))

    print("\n" + "="*60)
    print(f" Analisi Parallela Stack: {stack_name}")
    print(f"   Totale Frame: {total_frames} | Core: {num_cores} | Chunk size: {chunk_size}")
    print("="*60 + "\n") # Spazio per le barre

    # --- IL LOCK (analogia del pennarello) ---
    # tqdm e multiprocess: il core 1 vuole scrivere alla lavagna (aggiornare la propria progressbar), ma poi anche il core 2 vuole farlo ecc... -> caos. 
    # -> Regola!!! : Si puo scrivere alla lavagna solo con un pennarello (il lock) e il pennarello è unico e condiviso tra tutti i core. 
    # Pertanto ogni core prima di scrivere alla lavagna deve aspettare che il pennarello sia libero
    lock = multiprocessing.RLock()
    
    all_results = []
    
    # Passiamo initializer=init_pool_processes
    # Pool() è il gestore di tutti i sottoprocessi: assume num_cores operai, e li dirige
    with Pool(processes=num_cores, initializer=init_pool_processes, initargs=(lock,)) as pool:
        
        results_of_chunks = pool.map(worker_process_frames_chunk, tasks)
        
        for chunk_res in results_of_chunks:
            all_results.extend(chunk_res)
            
    print("\n" + "="*60) # Chiusura visuale
    print(f"Scrittura risultati su {os.path.basename(results_csv)}...")
    
    # Funzione helper per ordinare i risultati per numero di frame
    def get_frame_num(line):
        try:
            return int(line.split('\t')[1])
        except:
            return 0
    
    all_results.sort(key=get_frame_num)
    
    with open(results_csv, "w") as f:
        f.write("Stack_Name\tFrame\tZ_Focus[cm]\tArea[um2]\tDimA[mm]\tDimB[mm]\n")
        for line in all_results:
            f.write(line + "\n")
            
    print("Completato.")

    
def process_stack_image_safe(args_tuple):
    """
    Wrapper per il multiprocessing.
    """
    import os  # Import esplicito per multiprocessing
    try:
        # 1. Spacchettamento
        stack_path, bkg_path, params, N_images_per_stack, task_index = args_tuple

        # Setup Nomi
        filename = os.path.basename(stack_path)
        stack_name = os.path.splitext(filename)[0]
        try: stack_id = stack_name.split("_")[-1]
        except: stack_id = "0"

        # Setup Output
        # Path per questo stack: results/nome_campione/f0/nome_stack/
        stack_save_path = os.path.join(params['f0_results_path'], stack_name)
        os.makedirs(stack_save_path, exist_ok=True)
        # File temporaneo per questo stack (verrà poi unito)
        temp_dir = os.path.join(params['f0_results_path'], 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        results_csv = os.path.join(temp_dir, f"temp_results_{stack_name}.txt")
        
        # Aggiorna params con stack_save_path per questo stack
        params['save_path'] = stack_save_path
        
        with open(results_csv, "w") as f:
             f.write("Stack_Name\tFrame\tZ_Focus[cm]\tArea[um2]\tDimA[mm]\tDimB[mm]\n")

        # -----------------------------------------------------------
        # DEFINIZIONE DI total_for_bar 
        # -----------------------------------------------------------
        # 1. Contiamo quanti frame ha davvero il file
        
        try:
            with tifffile.TiffFile(stack_path) as tif:
                total_frames_in_file = len(tif.pages)
        except Exception:
            total_frames_in_file = 0 # Fallback se non riesce a leggere

        # 2. Decidiamo il totale della barra
        # Se c'è un limite imposto (es. 5), il totale è min(5, frame_reali)
        # Altrimenti è frame_reali.
        if N_images_per_stack is not None:
            total_for_bar = min(total_frames_in_file, N_images_per_stack)
        else:
            total_for_bar = total_frames_in_file
        # -----------------------------------------------------------

        # Caricamento Background
        avg_background, _, _ = avg16(bkg_path + '/', plot_label=False)
        
        # Chiamata al generatore (verbose=False per zittire la barra interna)
        generator = stack_iterator(stack_path, verbose=False, N_frames=N_images_per_stack)
        
        with open(results_csv, "a") as f:
            
            iterator_with_bar = tqdm(
                generator,
                total=total_for_bar,
                # USA L'INDICE DEL FILE, NON DEL CORE
                position=task_index,  
                # MOSTRA IL NOME DELLO STACK
                desc=f"Stack {task_index + 1}...",
                leave=True, # Lascia la barra visibile quando finisce
                ncols=100,
                mininterval=0.5
            )
            
            for idx, img_matrix in iterator_with_bar:
                
                # Interruzione se superiamo il limite
                if N_images_per_stack is not None and idx >= N_images_per_stack:
                    # Aggiorniamo la barra al 100% visivo prima di uscire
                    iterator_with_bar.update(total_for_bar - iterator_with_bar.n)
                    break
                
                # Analisi
                res = process_single_image(
                    img_matrix, idx, avg_background, params, stack_save_path, stack_id
                )
                
                log_str = f"{stack_name}\t{res['frame_idx']}\t{res['z_focus']:.4f}\t{res['area']:.4e}\t{res['dim_A']}\t{res['dim_B']}"
                f.write(log_str + "\n")
        
        return results_csv

    except Exception as e:
        # Usa tqdm.write per stampare errori senza rompere le barre degli altri
        try:
            stack_name = os.path.basename(stack_path) if 'stack_path' in locals() else "unknown"
        except:
            stack_name = "unknown"
        tqdm.write(f" Errore su {stack_name}: {e}")
        import traceback
        tqdm.write(traceback.format_exc())
        return None
            


def scan_and_process_all_parallel(img_dir, N_files_to_analyze, params, N_images_per_stack=None):
    """
    Gestisce l'analisi parallela.
    Args:
        img_dir: Cartella dati
        N_files_to_analyze: Quanti file processare (es. 10 su 100)
        params: Dizionario parametri
        N_images_per_stack: Limite frame per ogni stack (None = tutti)
    """
    img_path = os.path.normpath(img_dir)

    if not os.path.exists(img_path):
        print(f"Errore: La cartella {img_path} non esiste.")
        return

    # 1. Selezione e Ordinamento File
    raw_files = [f for f in os.listdir(img_path) if f.endswith(".tif")]
    
    # Ordinamento numerico robusto (gestisce split)
    files = sorted(raw_files, key=lambda f: int(f.split('_')[-1].replace('.tif', '')))
    
    # Selezione degli N file distribuiti uniformemente
    if N_files_to_analyze is not None:
        real_N = min(N_files_to_analyze, len(files))
        # Seleziona file uniformemente distribuiti usando linspace
        indices = np.linspace(0, len(files) - 1, real_N, dtype=int)
        list_N_files = [files[i] for i in indices]
    else:
        list_N_files = files

    print("--- SETUP MULTIPROCESSING ---")
    print(f"File totali: {len(list_N_files)}")
    print(f"Frame per file: {'TUTTI' if N_images_per_stack is None else N_images_per_stack}")

    # 2. Creazione Task
    # La tupla deve contenere 4 elementi: (path, bkg, params, frame_limit)
    tasks = []
    for i, f in enumerate(list_N_files):
        full_path = os.path.join(img_dir, f)
        
        # Aggiungiamo 'i' (l'indice di riga) alla tupla
        tasks.append((full_path, params['bkg_path'], params, N_images_per_stack, i))
    

    # 3. Avvio Pool
    user_jobs = params.get('cpu_count') 
    num_cores = get_optimal_core_count(user_jobs)
    
    print("\n" + "="*60)
    print(f" AVVIO PARALLELO SU {num_cores} CORE")
    print(f"   Analisi di {len(tasks)} file totali (Limite frame: {N_images_per_stack})")
    print("-" * 60)
    
    # --- Stampa Elenco File PRIMA delle barre ---
    print(" Coda di lavorazione:")
    for i, t in enumerate(tasks):
        filename = os.path.basename(t[0]) # Estrae solo il nome file dal path
        print(f"  {i+1}. {filename}")
    print("=" * 60 + "\n") # Spazio vuoto per le barre

    # --- CREAZIONE DEL LOCK PER TQDM ---
    lock = multiprocessing.RLock()
    
    temp_files_created = []
    with Pool(processes=num_cores, initializer=init_pool_processes, initargs=(lock,)) as pool:
        temp_files_created = pool.map(process_stack_image_safe, tasks)
        
        print("temp_file_created")
        print(temp_files_created)
        

    # 4. Merge e Pulizia Finale
    # results.txt va a livello di f0
    final_output = params['results_txt_path']
    
    merge_temp_files(temp_files_created, final_output)
    clean_duplicate_rows_and_sort_results(final_output)
