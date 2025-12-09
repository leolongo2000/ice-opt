#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script di Setup Automatico per ICE-OPT Hologram Analysis

Questo script automatizza l'installazione completa del software:
- Verifica che mamba/conda sia installato
- Crea l'ambiente virtuale 'ice-opt-env'
- Installa HoloPy da conda-forge
- Installa tutte le altre dipendenze da requirements.txt

Uso:
    python setup.py
"""

import sys
import os
import subprocess
import platform

def print_header(text):
    """Stampa un'intestazione formattata"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def print_step(step_num, text):
    """Stampa un passo numerato"""
    print(f"\n[{step_num}] {text}")
    print("-" * 70)

def run_command(cmd, shell=False, check=True):
    """Esegue un comando e gestisce gli errori"""
    try:
        if isinstance(cmd, str):
            cmd = cmd.split()
        result = subprocess.run(cmd, shell=shell, check=check, 
                              capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr
    except FileNotFoundError:
        return False, "", "Comando non trovato"

def check_conda_mamba():
    """Verifica se mamba o conda sono installati"""
    print_step(1, "Verifica installazione mamba/conda")
    
    # Prova prima mamba
    success, stdout, stderr = run_command("mamba --version", shell=True)
    if success:
        print("✅ Mamba trovato!")
        print(f"   Versione: {stdout.strip()}")
        return "mamba"
    
    # Prova conda
    success, stdout, stderr = run_command("conda --version", shell=True)
    if success:
        print("✅ Conda trovato!")
        print(f"   Versione: {stdout.strip()}")
        print("⚠️  Nota: Mamba è più veloce. Considera di installare Miniforge3.")
        return "conda"
    
    print("❌ ERRORE: Né mamba né conda sono installati!")
    print("\nPer installare:")
    print("  - Miniforge3 (mamba): https://github.com/conda-forge/miniforge/releases")
    print("  - Anaconda (conda): https://www.anaconda.com/products/distribution")
    print("\nDopo l'installazione, riavvia il terminale e riesegui questo script.")
    return None

def check_environment_exists(manager):
    """Verifica se l'ambiente 'ice-opt-env' esiste già"""
    print_step(2, "Verifica ambiente esistente")
    
    success, stdout, stderr = run_command(f"{manager} env list", shell=True)
    if success:
        if "ice-opt-env" in stdout:
            print("⚠️  L'ambiente 'ice-opt-env' esiste già.")
            response = input("Vuoi ricrearlo? (s/n): ").strip().lower()
            if response in ['s', 'si', 'y', 'yes']:
                print(f"Rimuovo l'ambiente esistente...")
                run_command(f"{manager} env remove -n ice-opt-env -y", shell=True)
                return False
            else:
                print("Uso l'ambiente esistente.")
                return True
        else:
            print("✅ Nessun ambiente esistente trovato.")
            return False
    return False

def create_environment(manager):
    """Crea l'ambiente virtuale"""
    print_step(3, "Creazione ambiente virtuale 'ice-opt-env'")
    
    print(f"Creo l'ambiente con Python 3.9 usando {manager}...")
    success, stdout, stderr = run_command(
        f"{manager} create -n ice-opt-env python=3.9 -y", shell=True
    )
    
    if success:
        print("✅ Ambiente 'ice-opt-env' creato con successo!")
        return True
    else:
        print(f"❌ Errore nella creazione dell'ambiente:")
        print(stderr)
        return False

def install_holopy(manager):
    """Installa HoloPy da conda-forge"""
    print_step(4, "Installazione HoloPy da conda-forge")
    
    print("⚠️  IMPORTANTE: HoloPy deve essere installato da conda-forge, non da pip.")
    print(f"Installo HoloPy usando {manager}...")
    
    success, stdout, stderr = run_command(
        f"{manager} install -n ice-opt-env -c conda-forge holopy -y", shell=True
    )
    
    if success:
        print("✅ HoloPy installato con successo!")
        return True
    else:
        print(f"⚠️  Avviso durante l'installazione di HoloPy:")
        print(stderr)
        print("Puoi provare a installarlo manualmente dopo con:")
        print(f"  {manager} install -n ice-opt-env -c conda-forge holopy -y")
        return False

def install_requirements(manager):
    """Installa le dipendenze da requirements.txt"""
    print_step(5, "Installazione dipendenze da requirements.txt")
    
    if not os.path.exists("requirements.txt"):
        print("⚠️  File requirements.txt non trovato. Salto questo passo.")
        return True
    
    print("Installo le dipendenze usando pip nell'ambiente 'ice-opt-env'...")
    
    # Su Windows usa python, su Linux python3
    python_cmd = "python" if platform.system() == "Windows" else "python3"
    
    # Usa il python dell'ambiente conda
    if platform.system() == "Windows":
        pip_cmd = f"{manager} run -n ice-opt-env pip install -r requirements.txt"
    else:
        pip_cmd = f"{manager} run -n ice-opt-env pip install -r requirements.txt"
    
    success, stdout, stderr = run_command(pip_cmd, shell=True)
    
    if success:
        print("✅ Dipendenze installate con successo!")
        return True
    else:
        print(f"⚠️  Alcuni errori durante l'installazione:")
        print(stderr)
        print("\nPuoi provare a installarle manualmente dopo con:")
        print(f"  {manager} activate ice-opt-env")
        print("  pip install -r requirements.txt")
        return False

def print_final_instructions(manager):
    """Stampa le istruzioni finali"""
    print_header("🎉 Installazione Completata!")
    
    print("L'ambiente 'ice-opt-env' è stato creato e configurato.")
    print("\nPer usare il software:")
    print("\n1. Attiva l'ambiente:")
    print(f"   {manager} activate ice-opt-env")
    
    print("\n2. Esegui lo script:")
    if platform.system() == "Windows":
        print("   python main\\ice-opt-hologram-analysis.py --help")
    else:
        print("   python main/ice-opt-hologram-analysis.py --help")
    
    print("\n3. Per disattivare l'ambiente:")
    print("   conda deactivate")
    
    print("\n📚 Documentazione:")
    print("   - Guida installazione: INSTALL.md")
    print("   - Guida utilizzo: main/USAGE.md")
    print("   - Configurazione: main/config_example.yml")
    
    print("\n" + "="*70)

def main():
    """Funzione principale"""
    print_header("ICE-OPT Hologram Analysis - Setup Automatico")
    
    print("Questo script installerà:")
    print("  ✅ Ambiente virtuale 'ice-opt-env' con Python 3.9")
    print("  ✅ HoloPy da conda-forge")
    print("  ✅ Tutte le dipendenze da requirements.txt")
    print("\nPremi INVIO per continuare o Ctrl+C per annullare...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nInstallazione annullata.")
        sys.exit(0)
    
    # 1. Verifica mamba/conda
    manager = check_conda_mamba()
    if not manager:
        sys.exit(1)
    
    # 2. Verifica ambiente esistente
    env_exists = check_environment_exists(manager)
    
    # 3. Crea ambiente (se necessario)
    if not env_exists:
        if not create_environment(manager):
            print("\n❌ Errore nella creazione dell'ambiente. Installazione interrotta.")
            sys.exit(1)
    
    # 4. Installa HoloPy
    install_holopy(manager)
    
    # 5. Installa requirements
    install_requirements(manager)
    
    # 6. Istruzioni finali
    print_final_instructions(manager)

if __name__ == "__main__":
    main()

