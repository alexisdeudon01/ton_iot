# Détection DDoS pour PME

Framework reproductible pour charger, prétraiter, entraîner et évaluer des modèles de détection DDoS sur CIC‑DDoS2019 et TON_IoT, puis appliquer l’analyse multicritères (AHP‑TOPSIS) et générer les graphiques du mémoire.

## Prérequis
- Python 3.8+ (recommandé 3.10)
- Docker + NVIDIA Container Toolkit (pour GPU)

## Installation (local)
```bash
pip install -r requirements.txt
```

## Configuration des datasets
Les datasets doivent être placés dans :
```
data/raw/cic_ddos2019/
data/raw/ton_iot/
```

Les échantillons (ZIP) se trouvent dans :
```
data/sample/cic_ddos2019_sample.csv
data/sample/ton_iot_sample.csv
```

Si les datasets ne sont pas présents, le programme indiquera qu’ils sont disponibles sur Google Drive.
Lien Google Drive (dataset utilisé) :
- https://drive.google.com/file/d/1CAdK9IgIr74RvtR60OdJBuiXKy37egWg/view?usp=sharing

Le fichier `config.yaml` fixe l’échantillonnage à 5% (`sampling.fraction: 0.05`) pour alléger l’exécution.

## Exécution (local)
```bash
python main.py --config config.yaml
```
Options :
- `--skip-training`
- `--only-viz`

## Exécution (Docker)
```bash
docker-compose build
docker-compose run --rm ddos-framework python main.py --config config.yaml
docker-compose run --rm ddos-framework python main.py --config config.yaml --only-viz
docker-compose run --rm ddos-framework bash
```

## Structure des résultats
- `results/metrics.csv`
- `results/decision_matrix.csv`
- `results/topsis_ranking.csv`
- `graphs/` (figures générées)

## Structure du projet
```
project/
├── README.md
├── requirements.txt
├── config.yaml
├── main.py
├── Dockerfile
├── docker-compose.yml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   ├── models/
│   ├── mcdm/
│   └── visualization/
├── graphs/
└── results/
```
