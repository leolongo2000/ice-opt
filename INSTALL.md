# Guida all'Installazione - ICE-OPT Hologram Analysis

Questa guida fornisce istruzioni dettagliate per l'installazione su Windows e Linux usando **mamba** (consigliato) o **conda**.

## 📋 Prerequisiti

### Installazione di Miniforge3 (Mamba) o Anaconda3

**⚠️ IMPORTANTE**: Questo software richiede **mamba** o **conda** perché HoloPy è disponibile solo su conda-forge.

#### Opzione 1: Miniforge3 (Mamba) - ⭐ CONSIGLIATO

**Mamba è più veloce di conda** e consigliato per questo progetto.

**Windows:**
1. Scarica Miniforge3 da [github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge/releases)
2. Scarica il file `Miniforge3-Windows-x86_64.exe`
3. Esegui l'installer e segui le istruzioni
4. Apri **Miniforge Prompt** (non PowerShell normale)

**Linux:**
```bash
# Scarica e installa Miniforge3
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
# Segui le istruzioni e riavvia il terminale
```

#### Opzione 2: Anaconda3

**Windows:**
1. Scarica Anaconda da [anaconda.com](https://www.anaconda.com/products/distribution)
2. Esegui l'installer
3. Apri **Anaconda Prompt**

**Linux:**
```bash
# Scarica e installa Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-latest-Linux-x86_64.sh
bash Anaconda3-latest-Linux-x86_64.sh
# Segui le istruzioni e riavvia il terminale
```

### Verifica installazione

**Mamba:**
```bash
mamba --version
```

**Conda:**
```bash
conda --version
```

## 🚀 Installazione Automatica (Consigliata)

Il modo più semplice è usare lo script `setup.py` che automatizza tutto:

### Windows

1. **Apri Miniforge Prompt** (o Anaconda Prompt)
2. **Naviga nella cartella del progetto**:
   ```cmd
   cd C:\path\to\my_hologr
   ```
3. **Esegui lo script di setup**:
   ```cmd
   python setup.py
   ```

### Linux

1. **Apri un terminale** e naviga nella cartella del progetto:
   ```bash
   cd /path/to/my_hologr
   ```
2. **Esegui lo script di setup**:
   ```bash
   python3 setup.py
   ```

Lo script `setup.py`:
- ✅ Verifica che mamba/conda sia installato
- ✅ Crea l'ambiente virtuale `ice-opt` con Python 3.9. Perchè 3.9? Purtroppo Holopy vuole python 3.9, le versioni successive non gli piacciono.
- ✅ Installa HoloPy da conda-forge
- ✅ Installa tutte le altre dipendenze da requirements.txt
- ✅ Fornisce istruzioni per attivare l'ambiente

## 📝 Installazione Manuale

Se preferisci installare manualmente:

### 1. Crea l'ambiente

**Con mamba (consigliato):**
```bash
mamba create -n ice-opt python=3.9 -y
mamba activate ice-opt
```

**Con conda:**
```bash
conda create -n ice-opt python=3.9 -y
conda activate ice-opt
```

### 2. Installa HoloPy da conda-forge

**Con mamba:**
```bash
mamba install -c conda-forge holopy -y
```

**Con conda:**
```bash
conda install -c conda-forge holopy -y
```

### 3. Installa le altre dipendenze

```bash
pip install -r requirements.txt
```


## ✅ Verifica Installazione

Dopo l'installazione, verifica che tutto funzioni:

```bash
# Attiva l'ambiente (se non già attivo)
mamba activate ice-opt  # o conda activate ice-opt

# Mostra l'help
python main/ice-opt-hologram-analysis.py --help
```

Dovresti vedere la schermata di help con tutte le opzioni disponibili.


**Esegui un test demo**:
   ```bash
   python main/ice-opt-hologram-analysis.py -i /path/to/test/file.tif --yes
   ```

## 📚 Prossimi Passi

- Consulta [main/USAGE.md](main/USAGE.md) per la documentazione completa
- Esplora [main/config_example.yml](main/config_example.yml) per configurare i parametri
