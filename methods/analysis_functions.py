#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 01:05:33 2025

@author: leo
"""
# =============================================================================
# IMPORT LIBRERIE
# =============================================================================
import sys
import os
import warnings
import numpy as np
import matplotlib as mpl
import tifffile
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, label, sum_labels
from termcolor import colored
from function_hologram import repropagate, repropagate_fast
from function_auxiliary import crop_image, avg16
from function_sizes import object_dimension16
import json
from pathlib import Path

warnings.filterwarnings("ignore")
mpl.rcParams['figure.dpi'] = 200
method_dir = os.getcwd()
main_dir = os.path.join(method_dir, '../main')
sys.path.insert(0, os.path.abspath(main_dir))

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

def validate_consistency(args, parser, init_params, is_cli_mode=False):
    """
    Unisce Argparse e InitParams in un unico dizionario 'params'.
    Effettua la validazione e gestisce il cambio di nome delle variabili.
    """
    
    # 1. Creiamo il dizionario finale partendo da init_params
    # Questo è fondamentale per mantenere le costanti fisiche (pixel_size, lambda, ecc.)
    # che non passiamo da terminale.
    params = init_params.copy() 
    params['use_parallel'] = init_params.get('use_parallel_multiprocessing', False)
    # Mappa N_cores (dal config) a cpu_count (usato nel codice)
    params['cpu_count'] = init_params.get('cpu_count', None) or init_params.get('N_cores', None)
    # Normalizza frames_limit: converte stringhe vuote, "None", "null" in None, e stringhe numeriche in int
    frames_limit_raw = init_params.get('N_frames_per_stack', None)
    if frames_limit_raw is None or frames_limit_raw == '' or str(frames_limit_raw).lower() in ['none', 'null']:
        params['frames_limit'] = None
    elif isinstance(frames_limit_raw, str) and frames_limit_raw.isdigit():
        params['frames_limit'] = int(frames_limit_raw)
    elif isinstance(frames_limit_raw, (int, float)):
        params['frames_limit'] = int(frames_limit_raw)
    else:
        params['frames_limit'] = None

    # Aggiungi output_dir se specificato
    params['output_dir'] = getattr(args, 'output_dir', None) or init_params.get('output_dir', None)
            
    # --- CASO A: CLI MODE (is_cli_mode=True) - L'utente può sovrascrivere qualsiasi variabile ---
    if is_cli_mode:
        # Sovrascriviamo i valori nel dizionario con quelli del terminale.
        # È qui che avviene la "Traduzione" dei nomi.
        
        params['mode'] = args.mode

        # Input da terminale (priorità)
        if params['mode'] == 'single':
            params['input_file'] = args.input
            params['input_dir'] = None
        elif params['mode'] == 'all':
            params['input_file'] = None
            # args.img_dir è impostato in ice-opt-hologram-analysis.py quando args.input è una directory
            params['input_dir'] = getattr(args, 'img_dir', None) or args.input
        
        # Sovrascrivi i parametri da terminale solo se esplicitamente specificati
        # use_parallel: sovrascrivi solo se --parallel è passato (True), altrimenti mantieni il valore dal config
        if args.parallel:
            params['use_parallel'] = True
        # cpu_count: sovrascrivi solo se --jobs è specificato
        if args.jobs is not None:
            params['cpu_count'] = args.jobs
        # plot_label e save_label: sovrascrivi solo se specificati
        # Converti interi (0/1) in booleani
        if args.plot_label is not None:
            params['plot_label'] = bool(args.plot_label)
        if args.save_label is not None:
            params['save_label'] = bool(args.save_label)
        # frames_limit: sovrascrivi se --frames è specificato
        if hasattr(args, 'frames') and args.frames is not None:
            params['frames_limit'] = int(args.frames)
        
        # output_dir: sovrascrivi se -o è specificato
        if hasattr(args, 'output_dir') and args.output_dir is not None:
            params['output_dir'] = args.output_dir
        
        # N: sovrascrivi se -N è specificato
        if hasattr(args, 'N') and args.N is not None:
            params['N_files_to_analyze'] = args.N
        
    # --- CASO B: IDE MODE (is_cli_mode=False) - Usa valori dal config, sovrascrivi solo se esplicitamente specificato ---
    else:
        # Modalità IDE: usa i valori dal config, ma sovrascrivi solo quelli esplicitamente specificati
        params['mode'] = args.mode
        
        # Input dal config (già impostato in main)
        if params['mode'] == 'single':
            params['input_file'] = args.input
            params['input_dir'] = None
        elif params['mode'] == 'all':
            params['input_file'] = None
            params['input_dir'] = getattr(args, 'img_dir', None) or args.input
        
        # Sovrascrivi solo se esplicitamente specificato
        if hasattr(args, 'parallel') and args.parallel:
            params['use_parallel'] = True
        if hasattr(args, 'jobs') and args.jobs is not None:
            params['cpu_count'] = args.jobs
        if hasattr(args, 'plot_label') and args.plot_label:
            params['plot_label'] = True
        if hasattr(args, 'save_label') and args.save_label:
            params['save_label'] = True
        if hasattr(args, 'frames') and args.frames is not None:
            params['frames_limit'] = int(args.frames) if args.frames != '' else None
        if hasattr(args, 'output_dir') and args.output_dir is not None:
            params['output_dir'] = args.output_dir
        if hasattr(args, 'N') and args.N is not None:
            params['N_files_to_analyze'] = args.N

    # Gestione comune per entrambi i casi (CLI e IDE): validazione e calcolo parametri derivati
    if args.mode is not None:
        # --- VALIDAZIONE RIGOROSA (Sui nuovi valori di params) ---
        if params['mode'] == 'single':
            if params['input_file'] is None:
                parser.error("ERRORE: In mode='single' devi specificare --input (-i) con un file.")
            # Calcola parametri derivati (path, ecc.) basandosi su input_file
            params = _calculate_derived_params(params, params['input_file'])
                
        elif params['mode'] == 'all':
            if params['input_dir'] is None:
                parser.error("ERRORE: In mode='all' devi specificare --input (-i) con una directory.")
            
            # Gestione intelligente di -N: se non specificato, analizza tutti i file
            if hasattr(args, 'N') and args.N is not None:
                params['N_file_da_analizzare'] = args.N
            else:
                # Conta i file .tif nella directory
                if os.path.exists(params['input_dir']):
                    files = [f for f in os.listdir(params['input_dir']) if f.endswith(".tif")]
                    params['N_file_da_analizzare'] = len(files)
                    print(f"ATTENZIONE: -N non specificato: analizzerò tutti i {len(files)} file trovati nella directory")
                else:
                    parser.error(f"ERRORE: La directory '{params['input_dir']}' non esiste.")
            
            # Per mode='all', calcola i path basandosi sulla directory
            # Prendiamo il primo file .tif come riferimento per calcolare i path
            if params['input_dir'] and os.path.exists(params['input_dir']):
                files = sorted([f for f in os.listdir(params['input_dir']) if f.endswith(".tif")])
                if files:
                    first_file = os.path.join(params['input_dir'], files[0])
                    params = _calculate_derived_params(params, first_file)
                else:
                    # Se non ci sono file, calcola comunque i path base dalla directory
                    input_dir = os.path.normpath(params['input_dir'])
                    params['data_path'] = input_dir
                    params['bkg_path'] = input_dir.replace("data", "background")
                    # Estrai nome_campione e f0 dal path (stessa logica di sopra)
                    path_parts = Path(input_dir).parts
                    if 'measurements' in path_parts:
                        meas_idx = path_parts.index('measurements')
                        if meas_idx + 2 < len(path_parts):
                            nome_campione = path_parts[meas_idx + 1]
                            f0_dir = path_parts[meas_idx + 2]
                        else:
                            nome_campione = Path(input_dir).parent.name
                            f0_dir = Path(input_dir).name
                    else:
                        nome_campione = Path(input_dir).parent.name if Path(input_dir).parent.name != 'data' else Path(input_dir).parent.parent.name
                        f0_dir = Path(input_dir).name if Path(input_dir).name != 'data' else 'f0'
                    
                    # Usa output_dir se specificato, altrimenti sostituisci "measurements" con "results"
                    if params.get('output_dir'):
                        results_base = params['output_dir']
                    else:
                        results_base = input_dir.replace("measurements", "results")
                    results_base_path = Path(results_base).parent.parent
                    f0_results_path = os.path.join(results_base_path, f0_dir)
                    params['f0_results_path'] = f0_results_path
                    params['gif_path'] = os.path.join(f0_results_path, 'xgif')
                    params['bn_rec_path'] = os.path.join(params['gif_path'], "rec_bn")
                    params['col_rec_path'] = os.path.join(params['gif_path'], "rec_col")
                    params['holo_cut_path'] = os.path.join(params['gif_path'], "holo_cut")
                    params['variance_plots_path'] = os.path.join(params['gif_path'], "variance_plots")
                    params['results_txt_path'] = os.path.join(f0_results_path, "results.txt")
            # Assicurati che i parametri fisici derivati (k, M, p, satured_pixel_value) siano calcolati
            # Non passiamo input_file perché i path sono già stati calcolati sopra
            _calculate_derived_params(params, None)

    # --- CASO B: ESECUZIONE DA IDE (args.mode è None) ---
    else:
        # Questo caso non dovrebbe più verificarsi perché --input è ora sempre obbligatorio
        # Ma lo lasciamo per sicurezza
        if params.get('mode') == 'single' and params.get('input_file') is None:
            print("ERRORE: Mode è 'single' ma 'input_file' non è specificato.")
            sys.exit(1)
        elif params.get('mode') == 'all' and params.get('input_dir') is None:
            print("ERRORE: Mode è 'all' ma 'input_dir' non è specificato.")
            sys.exit(1)
    
    return params

def merge_temp_files(temp_files, final_output_path):
    print(f" Unione di {len(temp_files)} file temporanei...")
    
    with open(final_output_path, 'w') as outfile:
        # Scrivi Header una volta sola
        outfile.write("Stack_Name\tFrame\tZ_Focus[cm]\tArea[um2]\tDimA[mm]\tDimB[mm]\n")
        
        for fname in temp_files:
            # Salta se fname è None o non esiste
            if fname is None:
                continue
            if not os.path.exists(fname):
                continue
            
            try:
                with open(fname, 'r') as infile:
                    # Salta la prima riga (header del file temporaneo)
                    lines = infile.readlines()[1:]
                    outfile.writelines(lines)
                
                # Cancella il file temporaneo dopo l'unione
                os.remove(fname)
            except Exception as e:
                print(f"ATTENZIONE: Errore durante l'unione del file {fname}: {e}")
                continue
            
    print(f" File finale creato: {os.path.basename(final_output_path)}")

def clean_duplicate_rows_and_sort_results(file_path):
    """
    1. Rimuove le righe duplicate basandosi su (Stack_Name, Frame).
    2. Ordina le righe rimaste in base al numero progressivo dello stack.
    3. Sovrascrive il file ordinato e pulito.
    """
    if not os.path.exists(file_path):
        return

    print(f" Pulizia e Riordino file: {os.path.basename(file_path)}...")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2: return # C'è solo l'header o è vuoto

    header = lines[0]
    data_lines = lines[1:] # Tutte le righe tranne la prima
    
    # --- 1. PULIZIA DUPLICATI ---
    seen_keys = set()
    unique_lines = []

    for line in data_lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            # La chiave univoca è la coppia (NomeStack, FrameIndex)
            key = (parts[0], parts[1]) 
            
            if key not in seen_keys:
                seen_keys.add(key)
                unique_lines.append(line)

    # --- 2. ORDINAMENTO ---
    # Definiamo una funzione interna per estrarre la chiave di ordinamento numerica
    def sort_key_func(line):
        parts = line.strip().split('\t')
        stack_name = parts[0]
        frame_idx = int(parts[1])
        
        # Estrarre il numero dallo stack name (es. "blabla_121" -> 121)
        try:
            # Prende l'ultima parte dopo l'ultimo underscore
            stack_number = int(stack_name.split('_')[-1])
        except ValueError:
            stack_number = -1 # Fallback se il nome non ha numeri
            
        # Restituisce una tupla: (NumeroStack, NumeroFrame)
        return (stack_number, frame_idx)

    # Ordiniamo la lista pulita usando la chiave
    unique_lines.sort(key=sort_key_func)

    # --- 3. SALVATAGGIO ---
    with open(file_path, 'w') as f:
        f.write(header)
        f.writelines(unique_lines)
        
    print(f" File riordinato. Righe uniche totali: {len(unique_lines)}.")

def _get_default_params():
    """
    Restituisce un dizionario con i valori di default per tutti i parametri.
    Utile quando si usa solo CLI senza file di configurazione.
    """
    return {
        'mode': None,
        'input_file': None,
        'input_dir': None,
        'N_file_da_analizzare': None,
        'save_label': False,
        'plot_label': True,
        'center_find_label': False,
        'save_format': '.png',
        'dim': 1400,
        'img_limits': [500, 1900, 1250, 2650],
        'pixel_size': 5.5,  # um
        'bit_camera': 16,
        'N_frames_per_stack': 1,
        'use_parallel_multiprocessing': True,
        'N_cores': None,
        'medium_index': 1.00027,
        'illum_wavelen': 0.6335,  # um
        'start_zrange': 260000,  # um
        'end_zrange': 270000,  # um
        'steps': 20,
        'f': float('nan'),
        'q': 0,
        'offset_x': 400,
        'offset_y': 100,
        'number_of_centers': 1,
        'center_threshold': 0.1,
        'center_blursize': 20.0,
        # Questi verranno calcolati dinamicamente
        'data_path': None,
        'bkg_path': None,
        'name': None,
        'save_path': None,
        'gif_path': None,
        'bn_rec_path': None,
        'col_rec_path': None,
        'holo_cut_path': None,
        'k': None,  # Calcolato da illum_wavelen e medium_index
        'M': None,  # Calcolato da f e q
        'p': None,  # Calcolato da f e q
        'satured_pixel_value': None,  # Calcolato da bit_camera
    }

def _calculate_derived_params(params, input_file=None):
    """
    Calcola i parametri derivati (path, costanti fisiche) a partire da input_file e altri parametri.
    """
    # Calcola k (numero d'onda)
    if params.get('illum_wavelen') and params.get('medium_index'):
        params['k'] = np.pi * 2 / (params['illum_wavelen'] / params['medium_index'])
    
    # Calcola M, p, q (magnificazione)
    f = params.get('f', float('nan'))
    # Gestisce sia None (da JSON/YAML) che float('nan')
    if f is None:
        f = float('nan')
    
    q = params.get('q', 0)
    
    # Verifica se f è un numero valido (non NaN)
    # Gestisce il caso in cui f è None o non è un numero
    try:
        # Converte f in float se possibile, altrimenti considera NaN
        f_float = float(f) if f is not None else float('nan')
        f_is_nan = np.isnan(f_float)
    except (TypeError, ValueError):
        f_is_nan = True
    
    if not f_is_nan:
        params['p'] = f * q / (np.abs(f - q)) if q != 0 else 0
        params['M'] = params['p'] / q if q != 0 else 1
    else:
        params['p'] = 0
        params['q'] = 0
        params['M'] = 1
    
    # Calcola pixel_size effettivo (solo se non è già stato calcolato)
    # Salviamo il pixel_size base se non esiste già
    if 'pixel_size_base' not in params:
        params['pixel_size_base'] = params.get('pixel_size', 5.5)
    
    # Moltiplica per M solo se M è stato calcolato e pixel_size non è già stato moltiplicato
    if params.get('M') and params.get('pixel_size_base'):
        # Usa pixel_size_base se pixel_size è uguale al base (non ancora moltiplicato)
        if params.get('pixel_size') == params['pixel_size_base']:
            params['pixel_size'] = params['pixel_size_base'] * params['M']
    
    # Calcola satured_pixel_value
    if params.get('bit_camera'):
        params['satured_pixel_value'] = 2 ** params['bit_camera'] - 1
    
    # Calcola path se input_file è fornito
    if input_file:
        input_file = os.path.normpath(input_file)
        params['input_file'] = input_file
        params['data_path'] = os.path.dirname(input_file)
        params['bkg_path'] = params['data_path'].replace("data", "background")
        
        _, head = os.path.split(input_file)
        params['name'] = head[:-4] if head.endswith('.tif') else head
        
        # Estrai nome_campione e f0 dal path
        # Struttura attesa: measurements/nome_campione/f0/data/file.tif
        path_parts = Path(params['data_path']).parts
        if 'measurements' in path_parts:
            meas_idx = path_parts.index('measurements')
            if meas_idx + 2 < len(path_parts):
                nome_campione = path_parts[meas_idx + 1]
                f0_dir = path_parts[meas_idx + 2]  # f0, f90, ecc.
            else:
                # Fallback se la struttura non è quella attesa
                nome_campione = Path(params['data_path']).parent.name
                f0_dir = Path(params['data_path']).name
        else:
            # Fallback: usa i nomi delle directory
            nome_campione = Path(params['data_path']).parent.name if Path(params['data_path']).parent.name != 'data' else Path(params['data_path']).parent.parent.name
            f0_dir = Path(params['data_path']).name if Path(params['data_path']).name != 'data' else 'f0'
        
        # Crea struttura: results/nome_campione/f0/
        results_base = params['data_path'].replace("measurements", "results")
        results_base_path = Path(results_base).parent.parent  # Sale a livello di nome_campione
        f0_results_path = os.path.join(results_base_path, f0_dir)
        params['f0_results_path'] = f0_results_path  # Path a results/nome_campione/f0/
        
        # Path per lo stack specifico: results/nome_campione/f0/nome_stack/
        stack_save_path = os.path.join(f0_results_path, params['name'])
        params['save_path'] = stack_save_path
        
        # xgif a livello di f0: results/nome_campione/f0/xgif/
        params['gif_path'] = os.path.join(f0_results_path, 'xgif')
        params['bn_rec_path'] = os.path.join(params['gif_path'], "rec_bn")
        params['col_rec_path'] = os.path.join(params['gif_path'], "rec_col")
        params['holo_cut_path'] = os.path.join(params['gif_path'], "holo_cut")
        params['variance_plots_path'] = os.path.join(params['gif_path'], "variance_plots")
        
        # results.txt va a livello di f0: results/nome_campione/f0/results.txt
        params['results_txt_path'] = os.path.join(f0_results_path, "results.txt")
        
        # Crea le directory se save_label è True
        if params.get('save_label', False):
            for path_key in ['save_path', 'gif_path', 'bn_rec_path', 'col_rec_path', 'holo_cut_path', 'variance_plots_path', 'f0_results_path']:
                path = params.get(path_key)
                if path and not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
    
    return params

def _load_config_file(config_file):
    """
    Carica un file di configurazione YAML o JSON.
    """
    config_file = os.path.abspath(config_file)
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"File di configurazione non trovato: {config_file}")
    
    ext = os.path.splitext(config_file)[1].lower()
    
    if ext in ['.yaml', '.yml']:
        if not HAS_YAML:
            raise ImportError("Per usare file YAML, installa PyYAML: pip install pyyaml")
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    elif ext == '.json':
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Formato file non supportato: {ext}. Usa .yaml, .yml o .json")
    
    return config

def get_params_from_init(config_file=None, is_cli_mode=False):
    """
    Carica i parametri da file di configurazione YAML/JSON.
    - Se config_file è specificato, carica da quel file
    - Altrimenti, carica config_default.yml dalla cartella main/
    
    Args:
        config_file: Path al file di configurazione YAML/JSON (opzionale)
    
    Returns:
        dict: Dizionario con tutti i parametri
    
    Raises:
        FileNotFoundError: Se il file di configurazione non esiste
    """
    params = {}
    
    # Determina il path di config_default.yml (sempre nella cartella main/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.join(script_dir, '../main')
    default_config_path = os.path.join(os.path.abspath(main_dir), 'config_default.yml')
    
    if config_file:
        # Carica da file di configurazione specificato
        config = _load_config_file(config_file)
        params = _get_default_params()
        # Aggiorna solo i valori non-None dal config (per evitare che None sovrascriva i default)
        # Ignora solo parametri deprecati
        ignored_keys = ['mode', 'N_file_da_analizzare', 'input_file', 'input_dir']  # input_file/input_dir deprecati, usa 'input'
        for key, value in config.items():
            if value is not None and key not in ignored_keys:
                # Normalizza N_frames_per_stack se presente (converte stringhe in None o int)
                if key == 'N_frames_per_stack':
                    if value == '' or str(value).lower() in ['none', 'null']:
                        params[key] = None
                    elif isinstance(value, str) and value.isdigit():
                        params[key] = int(value)
                    elif isinstance(value, (int, float)):
                        params[key] = int(value)
                    else:
                        params[key] = None
                else:
                    params[key] = value
        
        # Gestisci 'input' dal config: converte in input_file o input_dir in base al tipo
        if params.get('input') is not None:
            input_path = params['input']
            # Risolvi percorso relativo rispetto alla directory del config file
            if not os.path.isabs(input_path):
                config_dir = os.path.dirname(os.path.abspath(config_file))
                input_path = os.path.normpath(os.path.join(config_dir, input_path))
            if os.path.isfile(input_path):
                params['input_file'] = input_path
                params['input_dir'] = None
            elif os.path.isdir(input_path):
                params['input_file'] = None
                params['input_dir'] = input_path
            else:
                raise ValueError(f"ERRORE nel file di configurazione: Il path 'input' '{input_path}' non esiste o non è né un file né una directory.")
        # Rimuovi 'input' dai params (ora è stato convertito in input_file/input_dir)
        params.pop('input', None)
        
        # Valida che non ci siano sia input_file che input_dir (per compatibilità con vecchi config)
        if params.get('input_file') is not None and params.get('input_dir') is not None:
            raise ValueError("ERRORE nel file di configurazione: Non puoi specificare sia 'input_file' che 'input_dir' contemporaneamente.")
        
        # Calcola parametri derivati (senza input_file, verrà impostato da args)
        params = _calculate_derived_params(params, None)
        
    else:
        # Carica config_default.yml (sia da CLI che da IDE)
        if not os.path.exists(default_config_path):
            raise FileNotFoundError(
                f"File di configurazione non trovato: {default_config_path}\n"
                "Crea il file config_default.yml nella cartella main/ con i parametri necessari."
            )
        
        config = _load_config_file(default_config_path)
        params = _get_default_params()
        # Aggiorna solo i valori non-None dal config (per evitare che None sovrascriva i default)
        # Ignora solo parametri deprecati
        ignored_keys = ['mode', 'N_file_da_analizzare']
        for key, value in config.items():
            if value is not None and key not in ignored_keys:
                # Normalizza N_frames_per_stack se presente (converte stringhe in None o int)
                if key == 'N_frames_per_stack':
                    if value == '' or str(value).lower() in ['none', 'null']:
                        params[key] = None
                    elif isinstance(value, str) and value.isdigit():
                        params[key] = int(value)
                    elif isinstance(value, (int, float)):
                        params[key] = int(value)
                    else:
                        params[key] = None
                else:
                    params[key] = value
        
        # Gestisci 'input' dal config: converte in input_file o input_dir in base al tipo
        if params.get('input') is not None:
            input_path = params['input']
            # Risolvi percorso relativo rispetto alla directory del config file (config_default.yml è in main/)
            if not os.path.isabs(input_path):
                config_dir = os.path.dirname(os.path.abspath(default_config_path))
                input_path = os.path.normpath(os.path.join(config_dir, input_path))
            if os.path.isfile(input_path):
                params['input_file'] = input_path
                params['input_dir'] = None
            elif os.path.isdir(input_path):
                params['input_file'] = None
                params['input_dir'] = input_path
            else:
                raise ValueError(f"ERRORE nel file di configurazione: Il path 'input' '{input_path}' non esiste o non è né un file né una directory.")
        # Rimuovi 'input' dai params (ora è stato convertito in input_file/input_dir)
        params.pop('input', None)
        
        # Valida che non ci siano sia input_file che input_dir (per compatibilità con vecchi config)
        if params.get('input_file') is not None and params.get('input_dir') is not None:
            raise ValueError("ERRORE nel file di configurazione: Non puoi specificare sia 'input_file' che 'input_dir' contemporaneamente.")
        
        # I parametri derivati verranno calcolati dopo che validate_consistency
        # avrà impostato input_file da args
    
    return params

# --- 1. Il Generatore (molto piu efficiente rispetto ad un iteratore: un iteratore restituisce una lista di oggetti (panini) che occupa spazio in ram, un generatore è una ricetta on-demand, genera un oggetto alla volta e poi si mette in pausa, fino a quando non viene richiamato ---
  
def stack_iterator(input_file_path, N_frames, save_unwrapped=False, output_dir=None, verbose=True):
    """Legge lo stack frame per frame senza saturare la RAM."""
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"File non trovato: {input_file_path}")

    with tifffile.TiffFile(input_file_path) as tif:
        total = len(tif.pages)
        if verbose:
            print(colored(f"Stack caricato: {total} frame.", "cyan"))
        
        if save_unwrapped and output_dir:
            os.makedirs(output_dir, exist_ok=True)
             
        if N_frames is not None:
            if verbose:
                print(colored(f" Analizzo {N_frames} frame distribuiti uniformemente.", "cyan"))
            total_for_bar = min(total, N_frames)
        else:
            total_for_bar = total
        
        idx = np.linspace(0, total - 1, total_for_bar, dtype=int)

        bar_iterator_pages = [p for p in tif.pages[idx]]
        iterator = tqdm(bar_iterator_pages, desc="Processing Stack", unit="frame", disable=not verbose)
        for i, page in enumerate(iterator):
            frame = page.asarray()
            
            if save_unwrapped and output_dir:
                name = f"{os.path.splitext(os.path.basename(input_file_path))[0]}_{i:05d}.tif"
                tifffile.imwrite(os.path.join(output_dir, name), frame, photometric='minisblack')
            
            yield i, frame # 3. Restituisce il frame al ciclo principale e METTE IN PAUSA la funzione qui
    
# --- 2. Analisi di Singolo Frame: Sottrazione del background, ripropagazione ecc... ---
def process_single_image(img_matrix, frame_idx, bkg_matrix, params, stack_save_path, stack_id):
    """
    Analizza una SINGOLA matrice immagine (già in RAM).
    
    Args:
        img_matrix (numpy array): L'ologramma raw.
        frame_idx (int): Indice del frame (per log e nomi file).
        bkg_matrix (numpy array): Il background medio (già caricato).
        params (dict): Dizionario con tutti i parametri fisici (pixel_size, lambda, z_range, ecc.).
        stack_save_path (str): Path alla cartella dello stack (results/nome_campione/f0/nome_stack/).
        stack_id (str): ID dello stack per i log.
    
    Returns:
        dict: Risultati dell'analisi (dimA, dimB, z, area, timestamp_simulato).
    """
    
    # 1. Estrazione Parametri dal dizionario (per pulizia)
    p_size = params['pixel_size']
    wavelen = params['illum_wavelen']
    n_medium = params['medium_index']
    z_start = params['start_zrange']
    z_end = params['end_zrange']
    z_steps = params['steps']
    img_limits = params['img_limits']
    save_label = params['save_label']
    plot_label = params['plot_label']
    save_path = params['save_path']
    save_format = params['save_format']
    #I_max = params['satured_pixel_value']
    #bit = params['bit_camera']
    
    # Setup Cartelle output specifiche per questo frame
    frame_id = f"frame_{frame_idx:04d}"
    
    # Path per questo frame: results/nome_campione/f0/nome_stack/frame_XXXX/
    # stack_save_path è results/nome_campione/f0/nome_stack/
    frame_save_path = os.path.join(stack_save_path, frame_id)
    if save_label:
        os.makedirs(frame_save_path, exist_ok=True)
    # Usa frame_save_path invece di save_path per i salvataggi di questo frame
    
    # --- A. PRE-PROCESSING (Crop & Normalize) ---
    holo_raw = crop_image(img_matrix, img_limits, params['dim'], params['offset_x'], params['offset_y'])
    bkg_cropped = crop_image(bkg_matrix, img_limits, params['dim'], params['offset_x'], params['offset_y'])
    
    
    
    # Normalizzazione e Sottrazione Background
    holo_norm = holo_raw / np.sum(holo_raw)
    bkg_norm = bkg_cropped / np.sum(bkg_cropped)
    holo = 1 - holo_norm / bkg_norm # Ologramma pulito
    
         
    # --- B. RIPROPAGAZIONE ALL'INDIETRO DEL CAMPO: RICOSTRUZIONE ---
    best_recon, var_list, z, idx, max_idx = repropagate_fast(
        holo, z_start, z_end, z_steps, 
        p_size, n_medium, wavelen, 
        plot_label=plot_label, save_label=save_label, 
        save_path=save_path,
        save_format=save_format,
        incremental_plot_label=False,
        incremental_save_label=False,
        verbose=False
    )

    # --- C. ANALISI DIMENSIONALE (Solo sul piano a fuoco max_idx) ---
    # best_recon = propagation_list[max_idx]
    # Nota: z è già in cm da repropagate_fast, ma best_z serve in um per object_dimension16
    best_z_cm = z[max_idx]  # in cm
    best_z = best_z_cm * 10000  # in um per object_dimension16
    
    # 1. Filtro e Maschera
    obj = gaussian_filter(np.abs(best_recon)**2, sigma=1)
    obj = obj / np.amax(obj)
    mask = (obj > 0.02).astype(int) # Threshold hardcoded o parametrico
    
    # 2. Calcolo Area
    lw, num = label(mask)
    area_val = sum_labels(mask, lw, range(num + 1)) * p_size * p_size
    # Se ci sono più oggetti, prendiamo il max (assumiamo un oggetto principale)
    if np.ndim(area_val) > 0: area_max = area_val.max()
    else: area_max = area_val

    # 3. Calcolo Dimensioni (A e B)
    # Nota: object_dimension16 vuole un array uint8 (0-255)
    mask_uint8 = (mask * 255).astype('uint8')
    
    
    # Usa frame_save_path per salvare le dimensioni in questo frame
    frame_dim_path = frame_save_path if save_label else save_path
    dimA_arr, dimB_arr, _, _, _ = object_dimension16(
        mask_uint8, p_size, area_val, best_z, 
        save_label=save_label, save_path=frame_dim_path, save_format=save_format
    )
    
    dimA = round(np.amax(dimA_arr)/1000, 3) # Converti in mm
    dimB = round(np.amax(dimB_arr)/1000, 3)
    
    # 4. Salva plot delle varianze in xgif/variance_plots/
    if save_label:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        # z è già in cm da repropagate_fast
        plt.scatter(z, var_list, c=var_list, cmap='jet', marker='^', s=50)
        plt.axvline(best_z_cm, color='k', ls='--', linewidth=2, label=f'Best Z: {best_z_cm:.3f} cm')
        plt.xlabel('Z [cm]', fontsize=12)
        plt.ylabel('Variance', fontsize=12)
        plt.title(f'Variance Scan - Stack {stack_id} Frame {frame_idx}', fontsize=14)
        plt.colorbar(label='Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # Salva in xgif/variance_plots/ con nome che include stack_id, frame_id e best_z
        variance_plots_path = params.get('variance_plots_path', os.path.join(params['gif_path'], "variance_plots"))
        os.makedirs(variance_plots_path, exist_ok=True)
        variance_plot_filename = f"varianceplot_stack{stack_id}_frame{frame_idx:04d}_{best_z_cm:.3f}.png"
        variance_plot_path = os.path.join(variance_plots_path, variance_plot_filename)
        plt.savefig(variance_plot_path, dpi=150)
        plt.close()
        
        
        
    # --- D. SALVATAGGIO RISULTATI (Immagine Ricostruita) ---
    if save_label:
        # Salva le immagini ricostruite nella cartella del frame
        # e anche in xgif per le future gif
        bn_recon_filename = f"bn_stack{stack_id}_{frame_id}_z{best_z_cm:.3f}.png"
        col_recon_filename = f"col_stack{stack_id}_{frame_id}_z{best_z_cm:.3f}.png"
        
        # Salva nella cartella del frame come PNG (non TIFF per evitare stack)
        from PIL import Image as PILImage
        # mask_uint8 è già uint8, salvalo direttamente
        PILImage.fromarray(mask_uint8).save(os.path.join(frame_save_path, bn_recon_filename))
        # best_recon potrebbe essere complesso, prendi solo il modulo
        best_recon_abs = np.abs(best_recon)
        if best_recon_abs.max() > 0:
            best_recon_norm_frame = ((best_recon_abs.astype('float32') / best_recon_abs.max()) * 255).astype('uint8')
        else:
            best_recon_norm_frame = best_recon_abs.astype('uint8')
        PILImage.fromarray(best_recon_norm_frame).save(os.path.join(frame_save_path, col_recon_filename))
        
        # Salva anche in xgif (solo .png per le gif future) con nomi che includono stack_id
        params['bn_rec_path'] = params.get('bn_rec_path', os.path.join(params['gif_path'], "rec_bn"))
        params['col_rec_path'] = params.get('col_rec_path', os.path.join(params['gif_path'], "rec_col"))
        os.makedirs(params['bn_rec_path'], exist_ok=True)
        os.makedirs(params['col_rec_path'], exist_ok=True)
        
        # Converti in uint8 e salva come PNG in xgif
        # Normalizza le immagini per il salvataggio PNG
        if mask_uint8.max() > 0:
            mask_uint8_norm = ((mask_uint8.astype('float32') / mask_uint8.max()) * 255).astype('uint8')
        else:
            mask_uint8_norm = mask_uint8.astype('uint8')
        
        # best_recon_norm_frame è già stato calcolato sopra
        from PIL import Image
        Image.fromarray(mask_uint8_norm).save(os.path.join(params['bn_rec_path'], bn_recon_filename))
        Image.fromarray(best_recon_norm_frame).save(os.path.join(params['col_rec_path'], col_recon_filename))
        
        # Salva holo_cut (ologramma pulito dopo background subtraction)
        holo_norm = holo.copy()
        if holo_norm.max() > holo_norm.min():
            holo_norm = ((holo_norm.astype('float32') - holo_norm.min()) / (holo_norm.max() - holo_norm.min()) * 255).astype('uint8')
        else:
            holo_norm = holo_norm.astype('uint8')
        
        # Salva holo_cut in xgif/holo_cut/ con nome che include stack_id e prefisso holo_ (formato TIFF)
        holo_cut_filename = f"holo_stack{stack_id}_{frame_id}_z{best_z_cm:.3f}.tif"
        os.makedirs(params['holo_cut_path'], exist_ok=True)
        # Salva come TIFF invece di PNG
        tifffile.imwrite(os.path.join(params['holo_cut_path'], holo_cut_filename), holo_norm)

    # --- E. RETURN DATI ---
    # Restituiamo i dati numerici per farli raccogliere al Manager
    return {
        "frame_idx": frame_idx,
        "z_focus": best_z_cm,  # in cm per coerenza con results.txt
        "area": area_max,
        "dim_A": dimA,
        "dim_B": dimB
    }

# --- 3. Analisi di Singolo Stack di immagini, caricate in memoria una alla volta e poi dimenticate, grazie alla logica di stack_generator() ---
def process_stack_singlecore(stack_path, bkg_path, params, N_images_per_stack=None):
    
    # ... (caricamento background solito) ...
    # print("Caricamento Background Medio...")
    avg_background, _, _ = avg16(bkg_path + '/', plot_label=False)

    stack_name = os.path.splitext(os.path.basename(stack_path))[0]
    stack_id = stack_name.split("_")[-1].replace(".tif","")
    
    # Path per questo stack: results/nome_campione/f0/nome_stack/
    stack_save_path = os.path.join(params['f0_results_path'], stack_name)
    os.makedirs(stack_save_path, exist_ok=True)
    params['save_path'] = stack_save_path  # Aggiorna save_path per questo stack
    
    # results.txt va a livello di f0, non nello stack
    results_csv = params['results_txt_path']
    
    # --- LOGICA APPEND PURA ---
    # Controlliamo solo se serve l'header
    file_exists = os.path.exists(results_csv) and os.path.getsize(results_csv) > 0
    
    with open(results_csv, "a") as f:
        if not file_exists:
             f.write("StackName\tFrame\tZFocus[cm]\tArea[um2]\tDimA[mm]\tDimB[mm]\n")

    generator = stack_iterator(stack_path, N_frames = N_images_per_stack, verbose = True)
    print(f"--- Analisi {stack_name} -> Append su {os.path.basename(results_csv)} ---")

    
    for idx, img_matrix in generator:

        if N_images_per_stack is not None and idx >= N_images_per_stack: break

        res = process_single_image(img_matrix, idx, avg_background, params, stack_save_path, stack_id)
        
        # Riga di log
        log_str = f"{stack_name}\t{res['frame_idx']}\t{res['z_focus']:.4f}\t{res['area']:.4e}\t{res['dim_A']}\t{res['dim_B']}"
        
        with open(results_csv, "a") as f:
            f.write(log_str + "\n")
            
    print(f"--- Stack {stack_name} completato ---")

    return results_csv # Restituiamo il path del file creato

            
# --- 4. Analisi di N oggetti.tif (stack o immagini singole che siano) contenuti in una <img_dir>. ---
def scan_and_process_all(img_dir, N, params, N_images_per_stack):
    
    # Normalizzo il path di input
    img_path = os.path.normpath(img_dir)

    if not os.path.exists(img_path):
        print(f"Errore: La cartella {img_path} non esiste.")
        return

    files = sorted([f for f in os.listdir(img_path) if f.endswith(".tif")])
    files = sorted(files, key=lambda f: int(f.split('_')[-1].replace('.tif', '')))
    
    print(colored(f"Trovati {len(files)} elementi da analizzare in {img_path}. Ne analizzerò {N}.", "yellow"))
    # NON LI VORRAI ANALIZZARE TUTTI IMMAGINO...
    
    indices = np.linspace(0, len(files) - 1, N, dtype=int)
    for i,idx in enumerate(indices):
        f = files[idx]  
        print(f"\nProcessing {i+1}/{len(files)}: {f}")
        name_clean = f.replace(".tif", "") 
        
        try:
            process_stack_singlecore(stack_path = os.path.join(img_path,f), 
                                bkg_path = params['bkg_path'], 
                                params = params, 
                                N_images_per_stack = N_images_per_stack)
        
        except Exception as e:
            print(colored(f"!!! CRITICAL ERROR ON {name_clean}: {e}", "red"))
            import traceback
            traceback.print_exc()
            continue 
    
    # --- PULIZIA FINALE ---
    # Ora che abbiamo finito tutto, apriamo il file cumulativo e togliamo i doppioni
    # results.txt è già a livello di f0
    results_file = params['results_txt_path']
    clean_duplicate_rows_and_sort_results(results_file)
    
            
    return params

