# Guía de Generación de Música

Esta guía detalla cómo generar música con el modelo entrenado, controlando emoción y tonalidad.

## Requisitos Previos

### Software Necesario
- Python 3.11+ con PyTorch instalado
- CUDA compatible (recomendado para velocidad)
- Fast Transformers instalado
- Dependencias del proyecto (`requirements.txt`)

### Archivos Requeridos (Descargar de fases previas)

Antes de generar música, debes descargar los resultados de las fases anteriores:

1. **Modelo entrenado**: https://drive.google.com/file/d/12O4MM2_R7gA88OtAJSqSepUH-ETip5M4/view?usp=drive_link
   - Descargar y colocar en: `dataset/models/checkpoints/`
   - Archivo principal: `loss_8_params.pt`

2. **Representación del dataset**: https://drive.google.com/file/d/1HvPVjTb8I4BergqWIi_vTNzvPsuEZJg0/view?usp=sharing
   - Descargar y extraer en: `dataset/representation/`
   - Archivos necesarios: `train_dictionary.pkl`, `train_data.npz`, `train_idx.npz`

## Descripción General

El sistema genera música MIDI de piano condicionada por:
- **Emoción**: 4 cuadrantes (Q1: Feliz, Q2: Enojado, Q3: Triste, Q4: Relajado)
- **Tonalidad**: 24 tonalidades mayores y menores (C:maj, A:min, etc.)

Y condiciones adicionales como tempo, duration, velocity

## Generación Básica

### Comando de Ejemplo (desde demo.bat)

```bash
python workspace\transformer\main_cp.py ^
  --data_root dataset\representation ^
  --exp_path dataset\models ^
  --mode inference-deterministic ^
  --task_type 4-cls ^
  --num_songs 1 ^
  --emo_tag 4 ^
  --save_tokens_json 1 ^
  --load_ckt checkpoints ^
  --load_ckt_loss 8 ^
  --load_dict train_dictionary.pkl ^
  --conditions "{\"duration\": \"Note_Duration_240\"}" ^
  --key_tag E:min ^
  --out_dir output\files
```

### Comando Simplificado

Para generar una pieza rápidamente:

```bash
cd workspace/transformer/

python main_cp.py --mode inference --task_type 4-cls --num_songs 1 --emo_tag 1 --load_ckt checkpoints --load_ckt_loss 8 --load_dict ../../dataset/representation/train_dictionary.pkl
```

## Parámetros de Control

### Parámetros Esenciales

| Parámetro | Descripción | Valores |
|-----------|-------------|---------|
| `--mode` | Modo de operación | `inference`, `inference-deterministic` |
| `--task_type` | Tipo de tarea | `4-cls` (4 emociones) |
| `--emo_tag` | Emoción objetivo | 1 (Feliz), 2 (Enojado), 3 (Triste), 4 (Relajado) |
| `--num_songs` | Número de piezas a generar | Entero positivo |
| `--load_ckt` | Carpeta del checkpoint | `checkpoints` |
| `--load_ckt_loss` | Época del modelo | Número de época (ej: 8, 25) |
| `--load_dict` | Diccionario de vocabulario | Ruta al archivo .pkl |

### Parámetros de Condicionamiento

| Parámetro | Descripción | Ejemplo |
|-----------|-------------|---------|
| `--key_tag` | Tonalidad musical | `C:maj`, `A:min`, `G:maj`, `E:min` |
| `--conditions` | Condiciones adicionales | JSON string con tempo, duration, velocity |
| `--save_tokens_json` | Guardar tokens generados | 0 o 1 |

### Parámetros de Ruta

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--data_root` | Carpeta de datos | `dataset/representation` |
| `--exp_path` | Carpeta de experimentos | `dataset/models` |
| `--out_dir` | Directorio de salida | `output/files` |

## Ejemplos de Uso

### Generar música feliz en Do mayor
```bash
python workspace/transformer/main_cp.py --mode inference --emo_tag 1 --key_tag C:maj --num_songs 1 --load_ckt checkpoints --load_ckt_loss 8 --load_dict dataset/representation/train_dictionary.pkl
```

### Generar música triste en La menor
```bash
python workspace/transformer/main_cp.py --mode inference --emo_tag 3 --key_tag A:min --num_songs 1 --load_ckt checkpoints --load_ckt_loss 8 --load_dict dataset/representation/train_dictionary.pkl
```

### Generar música relajada con condiciones específicas
```bash
python workspace/transformer/main_cp.py --mode inference-deterministic --emo_tag 4 --key_tag E:min --conditions "{\"duration\": \"Note_Duration_240\"}" --num_songs 1 --load_ckt checkpoints --load_ckt_loss 8 --load_dict dataset/representation/train_dictionary.pkl
```

## Archivos de Salida

Los archivos generados se guardan en `output/files/` con el formato:
- `emo_[emocion]_[id_aleatorio].mid` - Archivo MIDI
- `emo_[emocion]_[id_aleatorio].npy` - Representación numpy
- `emo_[emocion]_[id_aleatorio].tokens.json` - Tokens (si `--save_tokens_json 1`)
- `emo_[emocion]_[id_aleatorio].meta.json` - Metadata

## Ejecución del Demo

Para ejecutar el demo completo con configuración predefinida:

```bash
# Windows
demo.bat

# Linux/Mac (ajustar rutas en el comando)
bash demo.sh
```

## Siguiente Paso

Con las piezas generadas, puedes:
- Reproducirlas en cualquier software MIDI
- Convertirlas a audio con herramientas externas
- Analizar los tokens generados en los archivos JSON
- Evaluar la adherencia usando la [guía de evaluación](evaluation.md)
