# Guida all'uso di ice-opt-hologram-analysis.py

## Panoramica

`ice-opt-hologram-analysis.py` è uno script per la ricostruzione della silhouette di oggetti a partire dalla registrazione dei loro ologrammi.
Supporta sia l'analisi di singoli file .tif (che siano singoli frame o stack) che batch processing su directory.

## Sintassi Base

```bash
python ice-opt-hologram-analysis.py -i <path>
```

Il programma **inferisce automaticamente** la modalità:
- Se `-i` punta a un **file** → analizza quel singolo file
- Se `-i` punta a una **directory** → analizza tutti i file `.tif` nella directory

## Argomenti Principali

### Input

- **`-i, --input`**: Path al file `.tif` o directory con file `.tif` (opzionale se specificato nel config file)
  ```bash
  # Singolo file
  python ice-opt-hologram-analysis.py -i /path/to/file.tif
  
  # Directory
  python ice-opt-hologram-analysis.py -i /path/to/directory/
  ```
  
  **Nota**: Se `input` è specificato nel file di configurazione, `-i` diventa opzionale. Tale modalità, in cui viene specificato l'input nel file di configurazione, è pensata per un utilizzo da IDE (spyder, vscode, ecc...).

  In ogni caso, se `-i` viene specificato da terminale, ha **priorità** sul valore nel config file.

### Opzionali

- **`-N`**: Numero di file da analizzare (solo per directory)
  ```bash
  # Analizza solo i primi 10 file
  python ice-opt-hologram-analysis.py -i /path/to/dir/ -N 10
  ```
  Se non specificato, analizza **tutti i file** trovati nella directory.

- **`--frames, -f`**: Limite di frame da analizzare per ogni stack
  ```bash
  # Analizza solo i primi 5 frame di ogni stack
  python ice-opt-hologram-analysis.py -i file.tif --frames 5
  ```
  Se non specificato, analizza **tutti i frame**.

- **`--parallel, -mp`**: Abilita il multiprocessing (più veloce)
  ```bash
  python ice-opt-hologram-analysis.py -i file.tif --parallel
  ```

- **`--jobs, -j`**: Numero di core da utilizzare
  ```bash
  # Usa 4 core
  python ice-opt-hologram-analysis.py -i file.tif --parallel -j 4
  ```
  Se non specificato, usa automaticamente il 50% della CPU disponibile.

- **`--plot_label, -p`**: Mostra i plot durante l'elaborazione 
  ```bash
  python ice-opt-hologram-analysis.py -i file.tif -p
  ```

- **`--save_label, -s`**: Salva le immagini elaborate
  ```bash
  python ice-opt-hologram-analysis.py -i file.tif -s
  ```

- **`--output-dir, -o`**: Directory di output personalizzata
  ```bash
  # Specifica una directory di output personalizzata
  python ice-opt-hologram-analysis.py -i file.tif -o /path/to/output/
  ```
  Se non specificato, il programma sostituisce automaticamente "measurements" con "results" nel path di input.

- **`--config-file`**: File di configurazione YAML o JSON
  ```bash
  python ice-opt-hologram-analysis.py -i file.tif --config-file config.yml
  ```
  Vedi sezione [File di Configurazione](#file-di-configurazione) per dettagli.

- **`--yes, -y`**: Salta la conferma e procede automaticamente
  ```bash
  python ice-opt-hologram-analysis.py -i file.tif -y
  ```
  Utile per script automatizzati.


## Conferma Interattiva

Per default, il programma mostra un **riepilogo completo** dell'analisi e chiede conferma prima di procedere. Il riepilogo include:

- ✅ Modalità (singolo file o directory)
- ✅ File/directory di input
- ✅ Lista dei file che verranno analizzati
- ✅ Parametri di elaborazione (multiprocessing, frame, core)
- ✅ Parametri di output (salva immagini, mostra plot)
- ✅ File di configurazione usato
- ✅ Parametri fisici principali

Esempio di output:
```
======================================================================
                    RIEPILOGO ANALISI OLOGRAMMI
======================================================================

📋 MODALITÀ: DIRECTORY (BATCH)
📁 DIRECTORY INPUT: /path/to/directory/
   File trovati: 50
   File da analizzare: TUTTI (50)

📄 LISTA FILE DA ANALIZZARE:
     1. file_001.tif
     2. file_002.tif
     ...

⚙️  PARAMETRI ELABORAZIONE:
   Multiprocessing: ✅ SÌ
   Core utilizzati: 4 core
   Frame per stack: 10 frame per stack

💾 PARAMETRI OUTPUT:
   Salva immagini: ✅ SÌ
   Mostra plot: ✅ SÌ
   Path output: /path/to/results/xgif

📝 FILE CONFIGURAZIONE: config.yml

🔬 PARAMETRI FISICI:
   Pixel size: 5.50 µm
   Lunghezza d'onda: 0.63 µm
   Range Z: 260000 - 270000 µm
   Step Z: 20

======================================================================

❓ Procedere con l'analisi? [y/n]:
```

Per saltare la conferma (utile per script automatizzati):
```bash
python ice-opt-hologram-analysis.py -i file.tif --yes
```

## File di Configurazione

Il programma usa automaticamente `config_default.yml` dalla cartella `main/` se non specifichi un file personalizzato. Puoi creare un file di configurazione personalizzato per sovrascrivere i parametri di default.

### Formato YAML

Crea un file `config.yml` nella cartella `main/` o copia `config_example.yml`:
```yaml
# Parametri Input/Output
input: null                      # Path al file .tif o directory (null = usa --input da CLI)
output_dir: null                 # Directory di output (null = sostituisce "measurements" con "results")

# Parametri di Output
save_label: false
plot_label: true
save_format: ".png"

# Parametri Immagine
pixel_size: 5.5
bit_camera: 16
dim: 1400
img_limits: [500, 1900, 1250, 2650]

# Parametri Fisici
medium_index: 1.00027
illum_wavelen: 0.6335
start_zrange: 260000
end_zrange: 270000
steps: 20

# Parametri Elaborazione
N_frames_per_stack: null         # null = tutti i frame
use_parallel_multiprocessing: false
N_cores: null                    # null = automatico (75% CPU)
```

### Uso

```bash
# Usa il config file specificato
python ice-opt-hologram-analysis.py --config-file config.yml

# Oppure specifica input da CLI (ha priorità sul config)
python ice-opt-hologram-analysis.py -i file.tif --config-file config.yml
```

**Nota Importante**: 
- In **modalità CLI** (esecuzione da terminale), gli argomenti da terminale **sovrascrivono sempre** i valori nel file di configurazione.
- Se `input` è specificato nel config file, `-i` diventa opzionale, ma se viene specificato, ha priorità.
- Puoi sovrascrivere **qualsiasi parametro** del config file usando le flag CLI corrispondenti.

## File di Configurazione Default

Il programma carica automaticamente `config_default.yml` dalla cartella `main/` quando esegui in modalità CLI senza specificare `--config-file`. Questo file contiene tutti i parametri di default.

Puoi:
- **Modificare direttamente** `config_default.yml` per cambiare i default globali
- **Creare un file personalizzato** (es. `my_config.yml`) e usarlo con `--config-file`
- **Copiare** `config_example.yml` come punto di partenza per un nuovo file di configurazione

I file di configurazione devono essere in formato **YAML** e devono trovarsi nella cartella `main/` (o specificare il path completo con `--config-file`).

## Output

I risultati vengono salvati in una struttura gerarchica organizzata:

### Struttura Directory Output

```
results/
  └── nome_campione/
      └── f0/
          ├── results.txt                    # File risultati complessivo
          ├── xgif/                          # Immagini per GIF
          │   ├── rec_bn/                    # Ricostruzioni binarie
          │   ├── rec_col/                   # Ricostruzioni a colori
          │   ├── holo_cut/                  # Ologrammi puliti (formato TIFF)
          │   └── variance_plots/            # Plot delle varianze
          └── nome_stack/                    # Cartella per ogni stack analizzato
              └── frame_XXXX/                # Cartella per ogni frame
                  ├── bn_recon_*.png         # Ricostruzione binaria
                  ├── col_recon_*.png        # Ricostruzione a colori
                  ├── variance_scan_*.png     # Plot varianze
                  └── rec_{z}_cm/            # Sottocartella per dimensioni
                      └── dimensions_*.png   # Immagini con dimensioni
```

### Convenzione Path

Per default, il programma sostituisce "measurements" con "results" nel path di input:
```
measurements/sample/f0/data/          ← Input
results/sample/f0/                    ← Output (automatico)
```

Puoi specificare una directory di output personalizzata con `-o`:
```bash
python ice-opt-hologram-analysis.py -i input.tif -o /path/to/custom/output/
```

## Modalità di Esecuzione

Il programma distingue automaticamente tra due modalità:

### Modalità CLI (Command Line Interface)
- Attivata quando esegui lo script da terminale con argomenti (`len(sys.argv) > 1`)
- Gli argomenti da terminale hanno **priorità assoluta** sul config file
- Puoi sovrascrivere **qualsiasi parametro** del config file usando le flag CLI
- Se `input` è nel config file, `-i` diventa opzionale, ma se specificato ha priorità

### Modalità IDE
- Attivata quando esegui lo script da Spyder/IPython senza argomenti
- Usa i valori dal config file (`config_default.yml` o quello specificato)
- Gli argomenti da terminale sovrascrivono solo se esplicitamente specificati

### Esempi

```bash
# CLI mode: input da terminale, altri parametri dal config
python ice-opt-hologram-analysis.py -i file.tif

# CLI mode: input e tutti i parametri da terminale (sovrascrive config)
python ice-opt-hologram-analysis.py -i file.tif --parallel -f 10 -s

# CLI mode: input dal config, altri parametri da terminale
python ice-opt-hologram-analysis.py --parallel -f 10

# IDE mode: tutto dal config (esecuzione da Spyder senza argomenti, schiaccia run e via)
# (usa i valori da config_default.yml)
```
