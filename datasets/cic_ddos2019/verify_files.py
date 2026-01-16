#!/usr/bin/env python3
"""
Script de vérification des fichiers CIC-DDoS2019
Compare les fichiers présents avec la documentation officielle UNB
"""

from pathlib import Path
import re

# Définir le chemin du dataset
DATASET_PATH = Path(__file__).parent.parent.parent / "datasets" / "cic_ddos2019"

# Attaques attendues selon UNB (https://www.unb.ca/cic/datasets/ddos-2019.html)
# Training Day (First Day)
TRAINING_DAY_ATTACKS = {
    "Benign", "PortMap", "NetBIOS", "LDAP", "MSSQL", 
    "UDP", "UDP-Lag", "SYN"
}

# Test Day (Second Day)
TEST_DAY_ATTACKS = {
    "Benign", "NTP", "DNS", "LDAP", "MSSQL", "NetBIOS", 
    "SNMP", "SSDP", "UDP", "UDP-Lag", "WebDDoS", "SYN", 
    "TFTP", "PortScan"
}

def normalize_attack_name(filename):
    """Normalise le nom d'attaque en enlevant le préfixe DrDoS_ et autres variations"""
    name = Path(filename).stem
    
    # Enlever préfixe DrDoS_
    if name.startswith("DrDoS_"):
        name = name[6:]
    
    # Normaliser les variations de casse et tirets
    name = name.replace("_", "-").replace(" ", "-")
    
    # Normaliser les cas spécifiques
    normalizations = {
        "portmap": "PortMap",
        "syn": "SYN",
        "udplag": "UDP-Lag",
        "netbios": "NetBIOS",
        "ldap": "LDAP",
        "mssql": "MSSQL",
        "udp": "UDP",
        "ntp": "NTP",
        "dns": "DNS",
        "tftp": "TFTP",
        "ssdp": "SSDP",
        "snmp": "SNMP",
        "webddos": "WebDDoS",
        "portscan": "PortScan",
        "benign": "Benign"
    }
    
    name_lower = name.lower()
    if name_lower in normalizations:
        return normalizations[name_lower]
    
    # Capitaliser la première lettre
    return name.capitalize()

def find_csv_files(dataset_path):
    """Trouve tous les fichiers CSV valides (excluant les templates/exemples)"""
    all_csv = list(dataset_path.rglob("*.csv"))
    
    valid_csv = []
    for csv_file in all_csv:
        filename_lower = csv_file.name.lower()
        # Exclure les fichiers template/example
        if any(excluded in filename_lower for excluded in ['example', 'sample', 'template', 'structure']):
            continue
        valid_csv.append(csv_file)
    
    return valid_csv

def categorize_files(files):
    """Catégorise les fichiers par type d'attaque"""
    attacks_found = {}
    drdos_attacks = set()
    non_drdos_attacks = set()
    
    for file in files:
        filename = file.name
        attack_name = normalize_attack_name(filename)
        
        if attack_name not in attacks_found:
            attacks_found[attack_name] = []
        attacks_found[attack_name].append(file)
        
        # Identifier si DrDoS ou non
        if "DrDoS" in filename:
            drdos_attacks.add(attack_name)
        else:
            non_drdos_attacks.add(attack_name)
    
    return attacks_found, drdos_attacks, non_drdos_attacks

def main():
    print("=" * 80)
    print("VÉRIFICATION DES FICHIERS CIC-DDoS2019")
    print("=" * 80)
    print(f"\n📁 Chemin du dataset: {DATASET_PATH}")
    print(f"   Existe: {DATASET_PATH.exists()}")
    
    if not DATASET_PATH.exists():
        print("❌ Le répertoire du dataset n'existe pas!")
        return
    
    # Trouver les fichiers CSV
    csv_files = find_csv_files(DATASET_PATH)
    print(f"\n📊 Fichiers CSV valides trouvés: {len(csv_files)}")
    
    if len(csv_files) == 0:
        print("⚠️  Aucun fichier CSV valide trouvé!")
        print("   Vérifiez que les fichiers sont dans datasets/cic_ddos2019/")
        return
    
    # Catégoriser les fichiers
    attacks_found, drdos_attacks, non_drdos_attacks = categorize_files(csv_files)
    
    print(f"\n🔍 Types d'attaques détectés: {len(attacks_found)}")
    print("\n" + "-" * 80)
    print("FICHIERS PAR TYPE D'ATTAQUE")
    print("-" * 80)
    
    for attack in sorted(attacks_found.keys()):
        files = attacks_found[attack]
        has_drdos = attack in drdos_attacks
        has_non_drdos = attack in non_drdos_attacks
        
        print(f"\n{attack}:")
        if has_drdos and has_non_drdos:
            print(f"  ⚠️  Les deux variantes présentes (DrDoS_ et sans préfixe)")
        
        for f in files:
            is_drdos = "DrDoS" in f.name
            marker = "🔵 DrDoS_" if is_drdos else "⚪ Direct"
            rel_path = f.relative_to(DATASET_PATH)
            print(f"  {marker} {rel_path}")
    
    # Comparaison avec documentation UNB
    print("\n" + "=" * 80)
    print("COMPARAISON AVEC DOCUMENTATION OFFICIELLE UNB")
    print("=" * 80)
    
    all_expected = TRAINING_DAY_ATTACKS | TEST_DAY_ATTACKS
    found_attacks = set(attacks_found.keys())
    
    print(f"\n📅 Training Day attendu: {len(TRAINING_DAY_ATTACKS)} types")
    missing_training = TRAINING_DAY_ATTACKS - found_attacks
    if missing_training:
        print(f"   ⚠️  Manquants: {', '.join(sorted(missing_training))}")
    else:
        print(f"   ✅ Tous les types présents")
    
    print(f"\n📅 Test Day attendu: {len(TEST_DAY_ATTACKS)} types")
    missing_test = TEST_DAY_ATTACKS - found_attacks
    if missing_test:
        print(f"   ⚠️  Manquants: {', '.join(sorted(missing_test))}")
    else:
        print(f"   ✅ Tous les types présents")
    
    # Attaques supplémentaires
    extra = found_attacks - all_expected
    if extra:
        print(f"\nℹ️  Attaques supplémentaires détectées: {', '.join(sorted(extra))}")
    
    # Résumé DrDoS vs non-DrDoS
    print("\n" + "=" * 80)
    print("RÉSUMÉ DrDoS vs NON-DrDoS")
    print("=" * 80)
    
    print(f"\n🔵 Fichiers avec préfixe DrDoS_: {len([f for f in csv_files if 'DrDoS' in f.name])}")
    print(f"⚪ Fichiers sans préfixe DrDoS_: {len([f for f in csv_files if 'DrDoS' not in f.name])}")
    
    overlap = drdos_attacks & non_drdos_attacks
    if overlap:
        print(f"\n⚠️  Types avec les deux variantes: {', '.join(sorted(overlap))}")
        print("   → Ces fichiers représentent des variantes différentes (réflexion vs direct)")
    
    print("\n" + "=" * 80)
    print("✅ Vérification terminée")
    print("=" * 80)
    print(f"\n💡 Note: Comparez ces résultats avec le contenu du Google Drive")
    print(f"   Lien: https://drive.google.com/drive/folders/1oxem2Xj6MoFbe-OrmePq0zWBsgusyGO_")

if __name__ == "__main__":
    main()
