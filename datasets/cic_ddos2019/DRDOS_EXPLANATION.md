# DrDoS vs Non-DrDoS Files - Explication

## Différence entre les fichiers avec et sans préfixe "DrDoS_"

Dans le dataset CIC-DDoS2019, vous pouvez trouver deux types de noms de fichiers pour certaines attaques :

### Fichiers avec préfixe "DrDoS_"
- **Exemples**: `DrDoS_NTP.csv`, `DrDoS_LDAP.csv`, `DrDoS_DNS.csv`, `DrDoS_MSSQL.csv`
- **Signification**: **DrDoS = Distributed Reflection Denial of Service**
- **Type d'attaque**: Attaques de **réflexion et amplification** DDoS

### Fichiers sans préfixe "DrDoS_"
- **Exemples**: `NTP.csv`, `LDAP.csv`, `DNS.csv`, `MSSQL.csv`, `UDP.csv`, `SYN.csv`
- **Signification**: Attaques DDoS **directes** ou variantes **non-amplifiées**
- **Type d'attaque**: Attaques DDoS standard (flood, exploitation, etc.)

## Qu'est-ce que DrDoS (Distributed Reflection DoS) ?

**DrDoS** est un type spécifique d'attaque DDoS qui utilise la technique de **réflexion** :

1. **Amplification**: L'attaquant envoie des requêtes avec l'adresse IP source **forgée** (celle de la victime) à des serveurs légitimes (réflecteurs)
2. **Réflexion**: Les serveurs légitimes répondent à la victime au lieu de l'attaquant
3. **Amplification**: La taille de la réponse est souvent beaucoup plus grande que la requête, amplifiant l'attaque

### Exemples d'attaques DrDoS

- **DrDoS_NTP**: Utilise le protocole NTP (Network Time Protocol) comme réflecteur
- **DrDoS_DNS**: Utilise des serveurs DNS publics comme réflecteurs
- **DrDoS_LDAP**: Utilise des serveurs LDAP comme réflecteurs
- **DrDoS_MSSQL**: Utilise Microsoft SQL Server comme réflecteur
- **DrDoS_SSDP**: Utilise SSDP (Simple Service Discovery Protocol) comme réflecteur

### Exemples d'attaques non-DrDoS

- **UDP**: Flood UDP direct (pas de réflexion)
- **SYN**: Flood SYN TCP (exploitation de la poignée de main TCP)
- **UDP-Lag**: Attaque UDP avec latence artificielle
- **PortScan**: Scan de ports (reconnaissance, pas DDoS)
- **PortMap**: Mapping de ports

## Dans le Dataset CIC-DDoS2019

Selon la [documentation officielle UNB](https://www.unb.ca/cic/datasets/ddos-2019.html), le dataset contient :

### Taxonomy des Attaques DDoS

**1. Reflection-based DDoS (DrDoS)**:
- **TCP-based**: MSSQL, SSDP
- **UDP-based**: CharGen, NTP, TFTP
- **TCP or UDP**: DNS, LDAP, NETBIOS, SNMP

**2. Exploitation-based DDoS**:
- **TCP-based**: SYN flood
- **UDP-based**: UDP flood, UDP-Lag

### Pourquoi les deux nomenclatures ?

Le dataset CIC-DDoS2019 peut contenir :
- Des fichiers avec préfixe `DrDoS_` pour distinguer explicitement les attaques de réflexion/amplification
- Des fichiers sans préfixe pour les mêmes types d'attaques mais avec des variantes différentes (directes, non-amplifiées, ou autre méthodologie)

**Note importante**: Les deux types de fichiers représentent des **variantes d'attaques similaires** mais peuvent avoir :
- Des caractéristiques de trafic différentes
- Des volumes d'attaque différents
- Des méthodes d'exécution différentes

## Impact sur le Dataset Loader

Le dataset loader traite **les deux types de fichiers** de la même manière :
- Les fichiers `DrDoS_NTP.csv` et `NTP.csv` sont **tous les deux chargés** comme représentant des attaques NTP
- Le loader normalise les noms en enlevant le préfixe "DrDoS_" pour la cohérence
- Les deux types contribuent à enrichir le dataset avec différentes variantes d'attaques

## Recommandation

Si vous avez les deux variantes (par exemple `DrDoS_NTP.csv` et `NTP.csv`), **gardez les deux** car ils représentent :
- Des scénarios d'attaque différents
- Des patterns de trafic complémentaires
- Une meilleure couverture du spectre d'attaques DDoS

Le loader les traitera automatiquement et les combinera dans le dataset final.
