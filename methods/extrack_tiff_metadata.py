#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tifffile
import os
import contextlib # Necessario per la redirezione
import numpy as np

def extract_tiff_metadata(file_path):
    """
    Estrae e stampa i metadati (tag) di ogni frame in un TIFF multipagina.
    """
    try:
        # 1. Apre il file TIFF
        with tifffile.TiffFile(file_path) as tif:
            num_pages = len(tif.pages)
            print(f"File: {os.path.basename(file_path)}")
            print(f"Identificate {num_pages} pagine (immagini).")

            # 2. Itera su ogni pagina
            for i, page in enumerate(tif.pages):
                
                print(f"\n--- Frame/Pagina {i+1} ---")
                
                # Accedi ai tag (metadati)
                # I tag sono spesso un dizionario con chiavi numeriche/stringa
                # I metadati specifici (es. timestamp, esposizione) sono
                # spesso inclusi nel tag ImageDescription (codice 270) in formato JSON o stringa.
                
                print(f"Dimensioni del Frame: {page.shape}")
                
                # Stampa alcuni tag standard
                print(f"Tipo di dati (dtype): {page.dtype}")
                
                # 3. Estrazione dei metadati specifici (ImageDescription, se presenti)
                try:
                    desc_tag = page.tags["ImageDescription"].value
                    print(f"ImageDescription (Metadati): {desc_tag[:100]}...")
                    # Se il desc_tag è JSON, puoi usare json.loads(desc_tag) per analizzarlo
                except KeyError:
                    print("Tag 'ImageDescription' (Metadati) non trovato.")
                    
                # Stampa tutti i tag (opzionale)
                for tag in page.tags:
                    print(f"  {tag.name}: {tag.value}")

    except FileNotFoundError:
        print(f"Errore: File non trovato a {file_path}")
    except Exception as e:
        print(f"Si è verificato un errore durante l'elaborazione del file: {e}")

def inspect_all_tiff_tags(file_path, max_frames=2):
    """
    Stampa tutti i tag (metadati) per i primi N frame per identificare
    dove sono salvati i dati personalizzati (es. Timestamp).
    """
    try:
        with tifffile.TiffFile(file_path) as tif:
            num_pages = len(tif.pages)
            print(f"File: {os.path.basename(file_path)} | Totale Pagine: {num_pages}")
            
            # Ispezioniamo solo i primi due frame per risparmiare tempo
            for i, page in enumerate(tif.pages):
                if i >= max_frames:
                    break
                    
                print(f"\n================ FRAME/PAGINA {i+1} ================")
                print(f"Dimensioni Immagine: {page.shape}")
                
                # Itera su tutti i tag della pagina
                for tag in page.tags:
                    value = tag.value
                    
                    # Formatta il valore per non stampare array giganti
                    if isinstance(value, (str, bytes)):
                        display_value = f"'{value[:100]}...' (lunghezza {len(value)})"
                    elif isinstance(value, np.ndarray):
                         display_value = f"<Numpy Array: shape {value.shape}, dtype {value.dtype}>"
                    else:
                        display_value = value
                        
                    print(f"  [Tag {tag.code}] {tag.name}: {display_value}")

    except FileNotFoundError:
        print(f"Errore: File non trovato a {file_path}")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")
        
def check_private_tags(file_path):
    """
    Estrae il valore dei tag privati 327xx e li mette in una tabella per confronto.
    """
    try:
        data = []
        with tifffile.TiffFile(file_path) as tif:
            
            # Trova tutti i codici di tag privati che iniziano con 327xx
            private_codes = sorted([t.code for t in tif.pages[0].tags if 32700 <= t.code <= 32799])
            
            # Raccogli i valori di quei tag per tutti i frame
            for i, page in enumerate(tif.pages):
                if i >= 65: # Limita a un numero gestibile di frame per la prima analisi
                    break
                    
                frame_data = {'Frame': i + 1}
                for code in private_codes:
                    try:
                        tag = page.tags[code]
                        # Tentativo di convertire valori non standard a numeri se possibile
                        value = tag.value[0] if isinstance(tag.value, tuple) else tag.value
                        frame_data[str(code)] = value
                    except KeyError:
                        frame_data[str(code)] = 'N/A'
                data.append(frame_data)
                
            # Stampa i dati in formato leggibile (CSV/Tabella)
            if data:
                keys = list(data[0].keys())
                # Scrivi su file per analisi esterna
                output_filename = os.path.splitext(os.path.basename(file_path))[0] + "_private_tags.csv"
                print(f"\nScrittura dei valori dei tag privati su: {output_filename}")
                
                with open(output_filename, 'w') as f:
                    f.write('\t'.join(keys) + '\n')
                    for row in data:
                        f.write('\t'.join(map(str, row.values())) + '\n')
                
                print(f"I primi 5 frame:")
                for i in range(min(5, len(data))):
                    print(data[i])
                    
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

def search_non_standard_tags(file_path):
    """
    Cerca tag con codici numerici alti o noti blocchi di metadati scientifici 
    che potrebbero contenere il timestamp.
    """
    print(f"Ricerca tag nascosti nel file: {os.path.basename(file_path)}")
    try:
        with tifffile.TiffFile(file_path) as tif:
            page = tif.pages[0] # Basta analizzare il primo frame
            found_metadata = False
            
            for tag in page.tags:
                tag_code = tag.code
                tag_name = tag.name
                value = tag.value
                
                # Cerca tag sopra la numerazione standard (es. 65000) o blocchi noti
                if tag_code >= 50000 or tag_name in ('Software', 'DocumentName', 'Manufacturer', 'ImageJMetaData'):
                    
                    found_metadata = True
                    display_value = ""

                    # Se è una stringa (JSON/XML) stampala per intero o tagliala
                    if isinstance(value, (str, bytes)):
                        display_value = value.decode('utf-8', errors='ignore') if isinstance(value, bytes) else value
                        # Stampiamo solo una parte se è troppo lungo
                        display_text = display_value if len(display_value) < 1000 else display_value[:500] + " [TRONCATO]..."
                        
                        print(f"\n[TAG SOSPETTO {tag_code} ({tag.name})]")
                        print(f"LUNGHEZZA: {len(display_value)} caratteri.")
                        print("CONTENUTO:\n" + "="*20)
                        print(display_text)
                        print("="*20)
                        
                    elif tag_code > 60000:
                        # Se è un tag numerico privato, stampiamo il valore
                        print(f"[TAG PRIVATO ALTO {tag_code} ({tag.name})]: {value}")

            if not found_metadata:
                print("\nNessun tag non standard significativo trovato. Il timestamp potrebbe non essere salvato per frame.")

    except Exception as e:
        print(f"Si è verificato un errore durante la ricerca: {e}")




# --- Esempio di utilizzo ---
file_path = "/media/leo/Pumpkin/ice-opt/measurements/0.5NaCl_4/f0/data/0.5gNaCl_2025-11-21T14-59-36.008_68.tif"
OUTPUT_FILENAME = "tiff_tags_output.txt"




print(f"L'output sta per essere reindirizzato a: {OUTPUT_FILENAME}")

# Apri il file e reindirizza l'output standard (stdout) al file
with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
    with contextlib.redirect_stdout(f):
        # Tutte le chiamate a print() all'interno di questo blocco
        # verranno scritte nel file 'f'.
        inspect_all_tiff_tags(file_path)

print("Redirezione completata. Controlla il file per l'output.")

# Apri il file e reindirizza l'output standard (stdout) al file
with open(OUTPUT_FILENAME, 'a', encoding='utf-8') as f:
    with contextlib.redirect_stdout(f):
        # Tutte le chiamate a print() all'interno di questo blocco
        # verranno scritte nel file 'f'.
        extract_tiff_metadata(file_path)


check_private_tags(file_path)
search_non_standard_tags(file_path)