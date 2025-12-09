#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
from datetime import datetime

class ExtractedDate:
    def __init__(self, match, datetime_obj):
        self.match = match
        self.datetime_obj = datetime_obj

    def __repr__(self):
        return f"<ExtractedDate match='{self.match}' datetime_obj={self.datetime_obj}>"

    def to_dict(self):
        return {
            'match': self.match,
            'datetime': self.datetime_obj.strftime("%Y-%m-%d %H:%M:%S"),
            'datetime_obj': self.datetime_obj
        }
    
def month_to_num(month_str):
    """Converte nomi/abbreviazioni di mesi (IT/EN) in numero."""
    month_map = {
        # Italiano (abbreviazioni)
        'gen': '01', 'feb': '02', 'mar': '03', 'apr': '04',
        'mag': '05', 'giu': '06', 'lug': '07', 'ago': '08',
        'set': '09', 'ott': '10', 'nov': '11', 'dic': '12',
        # Inglese (abbreviazioni)
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
        'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
        'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12',
        # Italiano (nomi completi)
        'gennaio': '01', 'febbraio': '02', 'marzo': '03', 'aprile': '04',
        'maggio': '05', 'giugno': '06', 'luglio': '07', 'agosto': '08',
        'settembre': '09', 'ottobre': '10', 'novembre': '11', 'dicembre': '12',
        # Inglese (nomi completi)
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'may': '05', 'june': '06', 'july': '07', 'august': '08',
        'september': '09', 'october': '10', 'november': '11', 'december': '12'
    }
    return month_map.get(month_str.lower(), "00")  # Cerca direttamente la stringa in lowercase


def extract_date(string, output_format="%Y-%m-%d %H:%M:%S", prompt="Inserisci la data (YYYY-MM-DD HH:MM:SS): ", interactive=False):
    basename = os.path.basename(string) #nel caso in cui la stringa fosse un path, basename estrae il basename
    basename = basename.strip()
    # Lista di tuple: (pattern, funzione per costruire la data)
    PATTERNS = [
        # Formati con timestamp
        (r'_?(\d{4})(\d{2})(\d{2})_(\d{2})_(\d{2})_(\d{2})', lambda m: f"{m[0]}-{m[1]}-{m[2]} {m[3]}:{m[4]}:{m[5]}"),  # YYYYMMDD_HH_MM_SS
        (r'_?(\d{4})(\d{2})(\d{2})[-_](\d{2})(\d{2})(\d{2})', lambda m: f"{m[0]}-{m[1]}-{m[2]} {m[3]}:{m[4]}:{m[5]}"),  # YYYYMMDD_(-)HHMMSS
        (r'_?(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})(\d{2})', lambda m: f"{m[0]}-{m[1]}-{m[2]} {m[3]}:{m[4]}:{m[5]}"),  # YYYY-MM-DD_HHMMSS
        (r'_?(\d{4})-(\d{2})-(\d{2})[-_ ](\d{2}):(\d{2}):(\d{2})', lambda m: f"{m[0]}-{m[1]}-{m[2]} {m[3]}:{m[4]}:{m[5]}"),  # YYYY-MM-DD_HH:MM:SS
        (r'_?(\d{4})-(\d{2})-(\d{2})_(\d{2})_(\d{2})_(\d{2})', lambda m: f"{m[0]}-{m[1]}-{m[2]} {m[3]}:{m[4]}:{m[5]}"),  # YYYY-MM-DD_HH_MM_SS
        (r'_?(\d{4})-(\d{2})-(\d{2})_(\d{2})[-_](\d{2})', lambda m: f"{m[0]}-{m[1]}-{m[2]} {m[3]}:{m[4]}:{00}"),  # YYYY-MM-DD_HH-MM
        (r'_?(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})', lambda m: f"{m[0]}-{m[1]}-{m[2]}T{m[3]}:{m[4]}:{m[5]}"),  # YYYYMMDDTHHMMSS
        (r'_?(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})', lambda m: f"{m[0]}-{m[1]}-{m[2]} {m[3]}:{m[4]}:{m[5]}"),  # YYYY-MM-DDTHH:MM:SS
        (r'_?(\d{4})-(\d{2})-(\d{2})[-_ ](\d{2}):(\d{2})', lambda m: f"{m[0]}-{m[1]}-{m[2]} {m[3]}:{m[4]}:00"), # YYYY-MM-DD_HH:MM
        
        # Formati con nomi/abbreviazioni di mesi (IT/EN)
        (r'_?(\d{1,2})-([A-Za-z]+)-(\d{4})', lambda m: f"{m[2]}-{month_to_num(m[1])}-{m[0]}"),  # Cattura parole intere 12 dicembre 2000
        (r'_?(\d{1,2})([A-Za-z]+)(\d{4})', lambda m: f"{m[2]}-{month_to_num(m[1])}-{m[0]}"),  # Cattura parole intere 12dicembre2000
        (r'_?(\d{1,2})[-_ ]([A-Za-z]{3})[-_ ](\d{4})', lambda m: f"{m[2]}-{month_to_num(m[1])}-{m[0]}"), # Cattura abbreviazioni
        (r'_?(\d{1,2})(?:[-_ ]?)([A-Za-z]{3})(?:[-_/\. ]?)(\d{4})', lambda m: f"{m[2]}-{month_to_num(m[1])}-{m[0]}"), # 3dic2024 ma anche 3_dic_2024 ma anche 2 dic 2024
        (r'_?(\d{1,2})(?:[-_ ]?)\s+([A-Za-z]+)\s+(?:[-_/\. ]?)(\d{4})', lambda m: f"{m[2]}-{month_to_num(m[1])}-{m[0]}"), # 14aprile2024
        
        # Formati solo data
        (r'_?(\d{4})(\d{2})(\d{2})', lambda m: f"{m[0]}-{m[1]}-{m[2]}"),  # YYYYMMDD
        (r'_?(\d{4})[-_ ](\d{2})[-_ ](\d{2})', lambda m: f"{m[0]}-{m[1]}-{m[2]}"),  # YYYY-MM-DD
        (r'_?(\d{2})[-_ ](\d{2})[-_ ](\d{4})', lambda m: f"{m[2]}-{m[1]}-{m[0]}"),  # DD-MM-YYYY
        (r'_?(\d{2})(\d{2})(\d{4})', lambda m: f"{m[2]}-{m[0]}-{m[1]}"),  # MMDDYYYY
        (r'_?(\d{8})', lambda m: f"{m[0][:4]}-{m[0][4:6]}-{m[0][6:8]}"),  # Qualsiasi 8 cifre (YYYYMMDD)
        (r'_?(\d{2})(\d{2})', lambda m: f"2024-{m[0]}-{m[1]}")
    ]
    
    for pattern, date_builder in PATTERNS:
        match = re.search(pattern, basename)
        if match:
            try:
                date_str = date_builder(match.groups())
                parsed_date = datetime.strptime(date_str, "%Y-%m-%d" if len(date_str) == 10 else"%Y-%m-%d %H:%M:%S")
                return ExtractedDate(match.group(), parsed_date)
            except (ValueError, IndexError) as e:
                continue  # prova il prossimo pattern
        
            
        
    # Se non trovato e non in modalità interattiva, raise ValueError, cosi posso catturarlo in un altro script
    if not interactive:
        raise ValueError(f"Nessun pattern valido trovato per la data in: {string}")
    
    # se invece è in modalità interattiva e non trova la data chiede di inserirla manualmente
    else:
        print(f"Nessun pattern valido trovato per la data in: {string}")
        while True:
            try:
                manual_date = input(prompt)
                parsed_date = datetime.strptime(manual_date, "%Y-%m-%d" if len(manual_date) == 10 else "%Y-%m-%d %H:%M:%S")
                return ExtractedDate(manual_date, parsed_date)
            except ValueError:
                print("Formato non valido. Usa YYYY-MM-DD (es: 2023-12-25). Riprova.")
            except KeyboardInterrupt:
                print("\nOperazione annullata")
                return None



# # Test
# print(extract_date("ARBOL0411.csv"))  # 20241104
# print(extract_date("dati_2024-11-06T11:03:10_report.csv"))  # 20241104
# print(extract_date("ARBOL_20241108_45477N_9231E.csv"))  # 20241104
# print(extract_date("04-11-2023_log.txt"))  # 20231104

# print(extract_date("04-dic-2023_log.txt"))  # 20231104
# print(extract_date("12-dicembre-2001_log.txt"))
# print(extract_date("file_senza_data.csv", interactive=False))  # Chiede input