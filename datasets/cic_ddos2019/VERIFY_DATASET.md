# Vérification du Dataset CIC-DDoS2019

Cette documentation explique comment vérifier que votre dataset CIC-DDoS2019 est conforme à la documentation officielle de l'UNB.

## Source officielle

- **URL**: https://www.unb.ca/cic/datasets/ddos-2019.html
- **Article**: "Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy", IEEE 53rd International Carnahan Conference on Security Technology, Chennai, India, 2019

## Structure attendue selon UNB

### Training Day (First Day)

Le **premier jour (Training Day)** doit contenir **7 types d'attaques** plus le trafic bénin :

1. **Benign.csv** - Trafic réseau normal
2. **PortMap.csv** - Attaque Port map (9:43 - 9:51)
3. **NetBIOS.csv** - Attaque NetBIOS (10:00 - 10:09)
4. **LDAP.csv** - Attaque LDAP amplification (10:21 - 10:30)
5. **MSSQL.csv** - Attaque MSSQL (10:33 - 10:42)
6. **UDP.csv** - Attaque UDP flood (10:53 - 11:03)
7. **UDP-Lag.csv** - Attaque UDP-Lag (11:14 - 11:24)
8. **SYN.csv** - Attaque SYN flood (11:28 - 17:35)

**Total**: 8 fichiers CSV (7 attaques + Benign)

### Test Day (Second Day)

Le **deuxième jour (Test Day)** doit contenir **12 types d'attaques** plus le trafic bénin et PortScan :

1. **Benign.csv** - Trafic réseau normal
2. **NTP.csv** - Attaque NTP amplification (10:35 - 10:45)
3. **DNS.csv** - Attaque DNS amplification (10:52 - 11:05)
4. **LDAP.csv** - Attaque LDAP amplification (11:22 - 11:32)
5. **MSSQL.csv** - Attaque MSSQL (11:36 - 11:45)
6. **NetBIOS.csv** - Attaque NetBIOS (11:50 - 12:00)
7. **SNMP.csv** - Attaque SNMP (12:12 - 12:23)
8. **SSDP.csv** - Attaque SSDP (12:27 - 12:37)
9. **UDP.csv** - Attaque UDP flood (12:45 - 13:09)
10. **UDP-Lag.csv** - Attaque UDP-Lag (13:11 - 13:15)
11. **WebDDoS.csv** - Attaque Web-based DDoS (13:18 - 13:29)
12. **SYN.csv** - Attaque SYN flood (13:29 - 13:34)
13. **TFTP.csv** - Attaque TFTP (13:35 - 17:15)
14. **PortScan.csv** - Attaque Port scan (nouveau dans le test day, timing inconnu)

**Total**: 14 fichiers CSV (12+1 attaques + Benign)

## Notes importantes

### Variations de noms de fichiers

Le dataset peut contenir des variantes de noms :
- **DrDoS_*** : Certains fichiers peuvent avoir le préfixe "DrDoS_" (Distributed Reflection DoS)
  - Exemple: `DrDoS_NTP.csv` = `NTP.csv`
- **Casse**: Les noms peuvent être en minuscules, majuscules, ou camelCase
  - Exemple: `Syn.csv`, `SYN.csv`, `syn.csv`
  - Exemple: `Portmap.csv`, `PortMap.csv`, `Port_Map.csv`
  - Exemple: `UDPLag.csv`, `UDP-Lag.csv`, `UDP_Lag.csv`

### Organisation des fichiers

**⚠️ IMPORTANT: Aucune réorganisation nécessaire**

Le dataset loader est conçu pour **détecter et charger automatiquement les fichiers CSV** depuis n'importe quel emplacement dans `datasets/cic_ddos2019/`, y compris :
- Le répertoire racine (`datasets/cic_ddos2019/*.csv`)
- Les sous-répertoires (`datasets/cic_ddos2019/*/*.csv`)
- Les sous-répertoires imbriqués (`datasets/cic_ddos2019/*/*/*.csv`)

**Pourquoi vous n'avez pas besoin de réorganiser :**

1. **Chargement flexible** : Le loader recherche récursivement dans tous les sous-répertoires, donc les fichiers peuvent rester dans `examples/Training-Day01/` et `examples/Test-Day02/` tels quels.

2. **Filtrage automatique** : Le loader exclut automatiquement les fichiers template/exemple (fichiers contenant "example", "sample", "template" ou "structure" dans leur nom) tout en chargeant les fichiers de données réels.

3. **Aucune organisation manuelle requise** : Peu importe où se trouvent vos fichiers CSV :
   - `examples/Training-Day01/`
   - `examples/Test-Day02/`
   - Répertoire racine
   - Tout autre sous-répertoire
   
   Le loader les trouvera et les chargera automatiquement.

4. **Prévient la corruption des données** : En filtrant par motif de nom de fichier plutôt que par emplacement de répertoire, nous pouvons stocker en toute sécurité les fichiers template d'exemple aux côtés des données réelles sans risque de chargement accidentel.

**Organisation actuelle (telle quelle) :**
```
datasets/cic_ddos2019/
├── examples/
│   ├── Training-Day01/
│   │   ├── DrDoS_NTP.csv        ← Sera chargé
│   │   ├── UDP.csv              ← Sera chargé
│   │   ├── SYN.csv              ← Sera chargé
│   │   └── ...
│   ├── Test-Day02/
│   │   ├── DrDoS_DNS.csv        ← Sera chargé
│   │   ├── TFTP.csv             ← Sera chargé
│   │   └── ...
│   └── example_cicddos2019_structure.csv  ← Exclu (fichier template)
```

**Résultat** : Le dataset loader détecte automatiquement tous les fichiers CSV valides (~20 fichiers actuellement) quelle que soit leur localisation dans l'arborescence des répertoires.

## Comment vérifier votre dataset

Exécutez le pipeline principal. Le dataset loader affichera :
- Le nombre de fichiers CSV trouvés
- Les fichiers exclus (exemple/template)
- Les types d'attaques détectés

## Fichiers exclus automatiquement

Les fichiers suivants sont **exclus** du chargement pour éviter la corruption des données :
- Fichiers contenant "example", "sample", "template", ou "structure" dans leur nom
- Exemple: `example_cicddos2019_structure.csv`

Ces fichiers sont uniquement à des fins de documentation et ne doivent pas être utilisés pour l'entraînement.
