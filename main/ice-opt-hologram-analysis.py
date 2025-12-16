#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 15:40:58 2025

@author: leo
"""
# %%

# =============================================================================
# IMPORT LIBRERIE
# =============================================================================
import argparse
import sys
import os
import time
import warnings
import matplotlib as mpl
import multiprocessing
import tifffile
import numpy as np

# -----------------------------------------------------------------------------
# SETUP PATH DI SISTEMA PER I MODULI CUSTOM
# -----------------------------------------------------------------------------
# Calcola il percorso assoluto della directory dello script corrente
script_dir = os.path.dirname(os.path.abspath(__file__))
# Risale di un livello e entra in 'methods'
methods_dir = os.path.join(script_dir, '../methods')

# Aggiungi il percorso dei metodi al PYTHONPATH
sys.path.insert(0, os.path.abspath(methods_dir))

from analysis_functions import validate_consistency, get_params_from_init, process_stack_singlecore, scan_and_process_all

from functions_for_multiprocessing import scan_and_process_all_parallel, process_stack_multicore

warnings.filterwarnings("ignore")
mpl.rcParams['figure.dpi'] = 200


def run_normal_analysis(params):
    """
    Gestisce l'esecuzione SEQUENZIALE (1 core).
    Utile per debug o macchine senza supporto multiprocessing.
    """
    print(f"\n ESECUZIONE NORMALE (Sequenziale) - Mode: {params['mode']}")
    
    if params['mode'] == 'single':
        if not os.path.exists(params['input_file']):
            print(f" Errore: File non trovato: {params['input_file']}")
            return

        process_stack_singlecore(
            stack_path = params['input_file'], 
            bkg_path = params['bkg_path'], 
            params = params, 
            N_images_per_stack = params['frames_limit']
        )

    elif params['mode'] == 'all':
        
        scan_and_process_all(
            img_dir = params['input_dir'], 
            N = params['N_file_da_analizzare'], 
            params = params,
            N_images_per_stack= params['frames_limit']
        )
        
def run_multiprocessing_analysis(params):
    """
    Gestisce l'esecuzione PARALLELA (Multi-core).
    Massima velocità.
    """
    print(f"\n ESECUZIONE MULTIPROCESSING (Parallela) - Mode: {params['mode']}")

    if params['mode'] == 'single':
        if not os.path.exists(params['input_file']):
            print(f" Errore: File non trovato: {params['input_file']}")
            return

        # Parallelismo sui frame dello stesso stack
        process_stack_multicore(
            stack_path = params['input_file'], 
            bkg_path = params['bkg_path'], 
            params = params, 
            N_images_per_stack = params['frames_limit']
        )

    elif params['mode'] == 'all':
        # Parallelismo sui file della cartella
        scan_and_process_all_parallel(
            img_dir = params['input_dir'], 
            N_files_to_analyze = params['N_file_da_analizzare'], 
            params = params, 
            N_images_per_stack = params['frames_limit']
        )
        
def print_summary_and_confirm(params, args, files_to_analyze):
    """
    Stampa un riepilogo completo dell'analisi e chiede conferma all'utente.
    """
    print("\n" + "="*70)
    print(" " * 20 + "RIEPILOGO ANALISI OLOGRAMMI")
    print("="*70 + "\n")
    
    # Modalità
    mode_str = "SINGOLO FILE" if params['mode'] == 'single' else "DIRECTORY (BATCH)"
    print(f"MODALITÀ: {mode_str}")
    
    # Input
    if params['mode'] == 'single':
        print(f"FILE INPUT: {params['input_file']}")
        if os.path.exists(params['input_file']):
            file_size = os.path.getsize(params['input_file']) / (1024*1024)  # MB
            print(f"   Dimensione: {file_size:.2f} MB")
            try:
                with tifffile.TiffFile(params['input_file']) as tif:
                    num_frames = len(tif.pages)
                    print(f"   Numero frame nello stack: {num_frames}")
            except Exception as e:
                print(f"   [!] Impossibile determinare il numero di frame: {e}")
    else:
        print(f"DIRECTORY INPUT: {params['input_dir']}")
        print(f"   File trovati: {len(os.listdir(params['input_dir']))}")
        if args.N is not None:
            print(f"   File da analizzare: {args.N} (specificato con -N)")
        else:
            print(f"   File da analizzare: TUTTI ({len(files_to_analyze)})")
    
    # Lista file (solo per mode='all', max 20 file)
    if params['mode'] == 'all' and files_to_analyze:
        print(f"\nLISTA FILE DA ANALIZZARE:")
        max_show = min(20, len(files_to_analyze))
        input_dir = params['input_dir']
        for i, fname in enumerate(files_to_analyze[:max_show], 1):
            # Costruisci il path completo del file
            full_path = os.path.join(input_dir, fname) if not os.path.isabs(fname) else fname
            try:
                with tifffile.TiffFile(full_path) as tif:
                    num_frames = len(tif.pages)
                    print(f"   {i:3d}. {fname} ({num_frames} frame)")
            except Exception as e:
                print(f"   {i:3d}. {fname} [!] Impossibile determinare il numero di frame: {e}")
        if len(files_to_analyze) > max_show:
            print(f"   ... e altri {len(files_to_analyze) - max_show} file")
        
    
    # Parametri elaborazione
    print(f"\nPARAMETRI ELABORAZIONE:")
    print(f"   Multiprocessing: {'SÌ' if params['use_parallel'] else 'NO'}")
    if params['use_parallel']:
        cpu_info = f"{params['cpu_count']} core" if params['cpu_count'] else "75% CPU (automatico)"
        print(f"   Core utilizzati: {cpu_info}")
    frames_info = f"{params['frames_limit']} frame" if params['frames_limit'] is not None else "Tutti i frame"
    print(f"   Frame per stack: {frames_info}")
    
    # Parametri output
    print(f"\nPARAMETRI OUTPUT:")
    print(f"   Salva immagini: {'SÌ' if params['save_label'] else 'NO'}")
    print(f"   Mostra plot: {'SÌ' if params['plot_label'] else 'NO'}")
    if params['mode'] == 'single':
        print(f"   Path output: {params.get('gif_path', 'N/A')}")
    else:
        print(f"   Path output: {params.get('gif_path', 'N/A')}")
    
    # Config file
    if args.config_file:
        print(f"\nFILE CONFIGURAZIONE: {args.config_file}")
    else:
        print(f"\nFILE CONFIGURAZIONE: Valori di default")
    
    # Parametri fisici principali
    print(f"\nPARAMETRI FISICI:")
    pixel_size = params.get('pixel_size', None)
    if pixel_size is not None:
        print(f"   Pixel size: {pixel_size:.2f} µm")
    else:
        print(f"   Pixel size: N/A")
    illum_wavelen = params.get('illum_wavelen', None)
    if illum_wavelen is not None:
        print(f"   Lunghezza d'onda: {illum_wavelen} µm")
    else:
        print(f"   Lunghezza d'onda: N/A")
    start_z = params.get('start_zrange', None)
    end_z = params.get('end_zrange', None)
    if start_z is not None and end_z is not None:
        print(f"   Range Z: {start_z} - {end_z} µm")
    else:
        print(f"   Range Z: N/A")
    steps = params.get('steps', None)
    if steps is not None:
        print(f"   Step Z: {steps}")
    else:
        print(f"   Step Z: N/A")
    
    print("\n" + "="*70)
    
    # Chiedi conferma
    while True:
        response = input("\nProcedere con l'analisi? [y/n]: ").strip().lower()
        if response in ['y', 'yes', 's', 'si', 'sì']:
            print("\nAvvio analisi...\n")
            return True
        elif response in ['n', 'no']:
            print("\nAnalisi annullata dall'utente.\n")
            return False
        else:
            print("Risposta non valida. Inserisci 'y' per procedere o 'n' per annullare.")


def main():
    
    parser = argparse.ArgumentParser(description='Hologram Analysis')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help="Path al singolo file .tif o directory con file .tif (opzionale se specificato nel config file)")
    parser.add_argument('-N', type=int, default=None,
                        help="Numero di file da analizzare (solo per directory, default: tutti i file trovati)")
    parser.add_argument('--frames', '-f', type=int, default=None,
                        help="Limite frame per stack (opzionale, default: tutti i frame)")
    # Flag per attivare il Multiprocessing
    parser.add_argument('--parallel', '-mp', action='store_true', 
                        help="Abilita il multiprocessing (più veloce)")
    parser.add_argument('--jobs', '-j', type=int, default=None,
                        help="Numero di core da utilizzare (es. -j 4). Se non specificato usa il 75%% della CPU.")
    parser.add_argument('--plot_label', '-p', action='store_true')
    parser.add_argument('--save_label', '-s', action='store_true')
    parser.add_argument('--config-file', type=str, default=None,
                        help="File di configurazione YAML o JSON (opzionale). Se non specificato, usa valori di default.")
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help="Directory di output (default: sostituisce 'measurements' con 'results' nel path di input)")
    parser.add_argument('--yes', '-y', action='store_true',
                        help="Salta la conferma e procede automaticamente")
    args = parser.parse_args()
    
    # Determina se siamo in CLI mode (len(sys.argv) > 1) o IDE mode
    is_cli_mode = len(sys.argv) > 1
    
    # 1. Caricamento parametri: da config file se specificato, altrimenti config_default.yml
    try:
        init_params = get_params_from_init(config_file=args.config_file, is_cli_mode=is_cli_mode)
    except ValueError as e:
        # Gestisce errori di configurazione (es. input_file e input_dir entrambi specificati)
        print(f"\n❌ {str(e)}\n")
        sys.exit(1)
    
    # 2. Gestione input: priorità a -i se specificato, altrimenti usa il config file
    if args.input is not None:
        # Input da CLI ha priorità
        if not os.path.exists(args.input):
            parser.error(f"ERRORE: Il path '{args.input}' non esiste.")
        
        # Inferisci mode da -i
        if os.path.isdir(args.input):
            args.mode = 'all'
            args.img_dir = args.input
        elif os.path.isfile(args.input):
            args.mode = 'single'
            args.img_dir = None
        else:
            parser.error(f"ERRORE: Il path '{args.input}' non è né un file né una directory")
    elif is_cli_mode:
        # In CLI mode, -i è obbligatorio se non è nel config
        if init_params.get('input') is None and init_params.get('input_file') is None and init_params.get('input_dir') is None:
            parser.error("ERRORE: Devi specificare --input (-i) o definire 'input' nel config file.")
        # Usa input dal config (può essere già convertito in input_file/input_dir o ancora come 'input')
        config_input = init_params.get('input') or init_params.get('input_file') or init_params.get('input_dir')
        if config_input:
            args.input = config_input
            if os.path.isfile(config_input):
                args.mode = 'single'
                args.img_dir = None
            elif os.path.isdir(config_input):
                args.mode = 'all'
                args.img_dir = config_input
            else:
                parser.error(f"ERRORE: Il path 'input' nel config file '{config_input}' non esiste.")
    else:
        # IDE mode: usa input dal config
        config_input = init_params.get('input') or init_params.get('input_file') or init_params.get('input_dir')
        if config_input:
            args.input = config_input
            if os.path.isfile(config_input):
                args.mode = 'single'
                args.img_dir = None
            elif os.path.isdir(config_input):
                args.mode = 'all'
                args.img_dir = config_input
            else:
                parser.error(f"ERRORE: Il path 'input' nel config file '{config_input}' non esiste.")
        else:
            parser.error("ERRORE: Devi specificare 'input' nel config file per la modalità IDE.")

    # 3. Validazione e Creazione Parametri: unisce args e init_params
    params = validate_consistency(args, parser, init_params, is_cli_mode=is_cli_mode)

    # 3. Prepara lista file da analizzare per il riepilogo
    files_to_analyze = []
    if params['mode'] == 'all':
        if os.path.exists(params['input_dir']):
            all_files = sorted([f for f in os.listdir(params['input_dir']) if f.endswith(".tif")])
            # Ordinamento numerico robusto (come nelle funzioni di processing)
            all_files = sorted(all_files, key=lambda f: int(f.split('_')[-1].replace('.tif', '')) if f.split('_')[-1].replace('.tif', '').isdigit() else 0)
            if args.N is not None:
                # Seleziona file uniformemente distribuiti
                real_N = min(args.N, len(all_files))
                indices = np.linspace(0, len(all_files) - 1, real_N, dtype=int)
                files_to_analyze = [all_files[i] for i in indices]
            else:
                files_to_analyze = all_files
    elif params['mode'] == 'single':
        files_to_analyze = [os.path.basename(params['input_file'])]
    
    # 4. Mostra riepilogo e chiedi conferma (a meno che non sia specificato --yes)
    if not args.yes:
        if not print_summary_and_confirm(params, args, files_to_analyze):
            sys.exit(0)
    
    use_parallel = params['use_parallel']
    
    # 5. ROUTING PRINCIPALE
    if use_parallel:
        run_multiprocessing_analysis(params)
    else:
        run_normal_analysis(params)


if __name__ == '__main__':
    # Necessario per Multiprocessing su Windows/Linux
    multiprocessing.freeze_support()
    
    start_time = time.time()
    
    main()
    
    print("\n--- Tempo totale esecuzione: %.2f seconds ---" % (time.time() - start_time))