# worldcup-predictor

Predicción de resultados de fútbol internacional usando el dataset de Kaggle
[martj42/international-football-results-from-1872-to-2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
(actualizado hasta 2025/2026).

## Estructura

```
worldcup-predictor/
├── data/
│   ├── raw/          # CSVs originales de Kaggle (no versionados)
│   └── processed/    # datasets derivados
├── src/
│   └── data_loader.py
├── notebooks/
├── models/           # modelos entrenados (no versionados)
└── requirements.txt
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Descargar datos

```powershell
python -c "import truststore; truststore.inject_into_ssl(); import kagglehub, shutil, pathlib; p = kagglehub.dataset_download('martj42/international-football-results-from-1872-to-2017'); [shutil.copy2(f, 'data/raw') for f in pathlib.Path(p).glob('*.csv')]"
```

(`truststore` evita errores SSL en Windows usando el almacén de certificados del sistema.)

## Uso

```powershell
python src\data_loader.py
```

`load_dataset(since=2010)` devuelve los partidos desde 2010 con features básicas:
equipos, sede neutral, tipo de torneo, diferencia de goles y resultado (`outcome`).
