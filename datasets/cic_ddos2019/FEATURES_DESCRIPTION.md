# Description des Features CIC-DDoS2019

Ce document décrit la structure unifiée des features extraites par CICFlowMeter pour le dataset CIC-DDoS2019.

## Structure du CSV

Chaque fichier CSV contient **80 features** (environ) générées par CICFlowMeter, plus les colonnes d'identification et de label.

## Colonnes Principales

### Identification du Flux
- **Flow ID** : Identifiant unique du flux (format: `IP_source-IP_dest-Port_source-Port_dest-Protocol`)
- **Source IP** : Adresse IP source
- **Source Port** : Port source
- **Destination IP** : Adresse IP destination  
- **Destination Port** : Port destination
- **Protocol** : Protocole réseau (6=TCP, 17=UDP, etc.)
- **Timestamp** : Horodatage du flux

### Métriques Temporelles
- **Flow Duration** : Durée totale du flux (microsecondes)
- **Flow IAT Mean/Std/Max/Min** : Statistiques sur les intervalles entre paquets (Inter-Arrival Time)
- **Fwd IAT Total/Mean/Std/Max/Min** : Statistiques IAT pour les paquets forward
- **Bwd IAT Total/Mean/Std/Max/Min** : Statistiques IAT pour les paquets backward
- **Active Mean/Std/Max/Min** : Temps actif du flux
- **Idle Mean/Std/Max/Min** : Temps d'inactivité du flux

### Statistiques de Paquets (Forward)
- **Total Fwd Packets** : Nombre total de paquets envoyés
- **Total Length of Fwd Packets** : Taille totale (bytes) des paquets forward
- **Fwd Packet Length Max/Min/Mean/Std** : Statistiques sur la taille des paquets forward
- **Fwd Packets/s** : Débit de paquets forward (paquets/seconde)
- **Fwd Header Length** : Longueur totale des en-têtes forward
- **Subflow Fwd Packets/Bytes** : Statistiques de sous-flux forward

### Statistiques de Paquets (Backward)
- **Total Backward Packets** : Nombre total de paquets reçus
- **Total Length of Bwd Packets** : Taille totale (bytes) des paquets backward
- **Bwd Packet Length Max/Min/Mean/Std** : Statistiques sur la taille des paquets backward
- **Bwd Packets/s** : Débit de paquets backward (paquets/seconde)
- **Bwd Header Length** : Longueur totale des en-têtes backward
- **Subflow Bwd Packets/Bytes** : Statistiques de sous-flux backward

### Statistiques Globales du Flux
- **Flow Bytes/s** : Débit total du flux (bytes/seconde)
- **Flow Packets/s** : Débit total de paquets (paquets/seconde)
- **Min/Max Packet Length** : Taille minimale/maximale de paquet dans le flux
- **Packet Length Mean/Std/Variance** : Statistiques globales sur la taille des paquets
- **Average Packet Size** : Taille moyenne des paquets
- **Down/Up Ratio** : Ratio paquets down/up

### Flags TCP
- **FIN Flag Count** : Nombre de flags FIN
- **SYN Flag Count** : Nombre de flags SYN
- **RST Flag Count** : Nombre de flags RST
- **PSH Flag Count** : Nombre de flags PSH
- **ACK Flag Count** : Nombre de flags ACK
- **URG Flag Count** : Nombre de flags URG
- **CWE Flag Count** : Nombre de flags CWE
- **ECE Flag Count** : Nombre de flags ECE
- **Fwd/Bwd PSH Flags** : Flags PSH forward/backward
- **Fwd/Bwd URG Flags** : Flags URG forward/backward

### Métriques de Segments TCP
- **Avg Fwd Segment Size** : Taille moyenne des segments forward
- **Avg Bwd Segment Size** : Taille moyenne des segments backward
- **Fwd Avg Bytes/Bulk** : Bytes moyens par bulk forward
- **Fwd Avg Packets/Bulk** : Paquets moyens par bulk forward
- **Fwd Avg Bulk Rate** : Taux moyen de bulk forward
- **Bwd Avg Bytes/Bulk** : Bytes moyens par bulk backward
- **Bwd Avg Packets/Bulk** : Paquets moyens par bulk backward
- **Bwd Avg Bulk Rate** : Taux moyen de bulk backward
- **min_seg_size_forward** : Taille minimale de segment forward
- **act_data_pkt_fwd** : Paquets de données actifs forward

### Fenêtre TCP
- **Init_Win_bytes_forward** : Taille de fenêtre initiale forward (-1 si UDP)
- **Init_Win_bytes_backward** : Taille de fenêtre initiale backward (-1 si UDP)

### Autres Features
- **SimillarHTTP** : Indicateur de similarité HTTP (0/1)
- **Inbound** : Indicateur de trafic entrant (0/1)

### Labels
- **Label** : Type d'attaque ou "Benign" (chaîne de caractères)
- **Attack** : Type d'attaque (alias de Label, peut être utilisé pour multi-classe)

## Valeurs Typiques

### Trafic Benign (Normal)
- Flow Duration : variable (quelques secondes à minutes)
- Flow Packets/s : modéré (10-1000 paquets/s)
- SYN Flags : 1-2 (établissement de connexion)
- Paquets forward/backward : équilibrés
- Down/Up Ratio : proche de 1

### Attaques DDoS
- **UDP Flood** : Beaucoup de paquets UDP, pas de backward packets, durée courte
- **SYN Flood** : Beaucoup de SYN flags, peu de ACK, connexions incomplètes
- **LDAP/MSSQL** : Amplification, gros paquets, ratio down/up élevé
- **DNS Amplification** : Gros paquets DNS, beaucoup de forward packets
- **NetBIOS/SNMP/SSDP** : Similar à UDP flood, caractéristiques spécifiques au protocole

## Normalisation et Préprocessing

Avant utilisation dans les modèles ML, il est recommandé de :
1. **Gérer les valeurs infinies** : Remplacer `Inf` par `NaN` ou une grande valeur
2. **Gérer les valeurs manquantes** : Imputer avec median ou mean
3. **Normaliser les features** : Utiliser RobustScaler ou StandardScaler
4. **Encoder les catégories** : LabelEncoder pour Label/Attack
5. **Sélection de features** : Certaines features peuvent être corrélées ou redondantes

## Exemple de Fichier

Voir `example_cicddos2019_structure.csv` pour un exemple complet avec plusieurs types d'attaques et du trafic benign.

## Références

- CICFlowMeter: https://github.com/ahlashkari/CICFlowMeter
- CIC-DDoS2019 Dataset: https://www.unb.ca/cic/datasets/ddos-2019.html
- Paper: Sharafaldin et al. (2019). "Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy"
