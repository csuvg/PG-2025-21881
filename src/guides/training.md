# Guía de Entrenamiento del Modelo

Esta guía detalla cómo entrenar el modelo Transformer para generación de música condicionada.

## Descripción General

El modelo es un Transformer autoregresivo de 39.2M parámetros que aprende a generar secuencias musicales condicionadas por emoción y tonalidad. El entrenamiento utiliza el dataset EMOPIA tokenizado con 9 dimensiones de características.

## Arquitectura del Modelo

- **Tipo**: Transformer autoregresivo con Fast Attention
- **Parámetros**: ~39.2 millones
- **Embedding**: 512 dimensiones
- **Capas**: 12 bloques transformer
- **Attention heads**: 8
- **Vocabulario**: Variable por dimensión (~25-135 tokens)
- **Secuencia**: 1024 tokens máximo

## Requisitos de Hardware

### Mínimo
- GPU: NVIDIA con 8GB VRAM (RTX 2070+)
- RAM: 16GB sistema
- Almacenamiento: 10GB libres

### Recomendado
- GPU: NVIDIA con 24GB VRAM (RTX 3090/4090)
- RAM: 32GB sistema
- Almacenamiento: SSD con 50GB libres (Recordar que se guardan múltiples)

## Preparación

1. Verificar instalación de PyTorch con CUDA:
   ```python
   import torch
   print(torch.cuda.is_available())  # Debe retornar True
   print(torch.cuda.get_device_name(0))
   ```

2. Verificar datos preprocesados:
   ```bash
   ls dataset/representation/
   # Debe contener: train_data.npz, train_dictionary.pkl, train_idx.npz
   ```

## Comando Básico de Entrenamiento

```bash
cd workspace/transformer/

python main_cp.py \
  --mode train \
  --path_train_data train \
  --exp_name my_experiment \
  --load_ckt none \
  --load_dict ../../dataset/representation/train_dictionary.pkl
```

## Parámetros de Entrenamiento

### Esenciales

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--mode` | train | Modo de operación (train/inference) |
| `--path_train_data` | train | Prefijo de archivos de datos |
| `--exp_name` | default | Nombre del experimento |
| `--load_dict` | - | Ruta al diccionario .pkl |
| `--load_ckt` | none | Checkpoint para continuar ('none' para nuevo) |

### Mejoras

| Parámetro | Descripción |
|-----------|-------------|
| `--data_parallel` | 0/1 - Usar múltiples GPUs |

## Configuraciones por Tipo de Tarea

### 1. Clasificación 4-Emociones (Recomendado)

```bash
python main_cp.py \
  --mode train \
  --task_type 4-cls \
  --path_train_data train \
  --exp_name emotion_4cls \
  --batch_size 8 \
  --epochs 800 \
  --load_dict ../../dataset/representation/train_dictionary.pkl
```

### 3. Entrenamiento con Multi-GPU

```bash
python main_cp.py \
  --mode train \
  --path_train_data train \
  --exp_name multi_gpu \
  --data_parallel 1 \
  --batch_size 16 \
  --load_dict ../../dataset/representation/train_dictionary.pkl
```

## Monitoreo del Entrenamiento

### Logs en Tiempo Real

El entrenamiento genera logs cada 100 batches:
```
Epoch 1/800 | Batch 100/206 | Loss: 3.2451
  - Tempo: 0.8923
  - Chord: 1.2341
  - Emotion: 0.5678
  - Key: 0.3421
```

### Checkpoints

Se guardan automáticamente en `dataset/models/checkpoints/`:
- Cada 5 épocas: `loss_X_params.pt`
- Mejor modelo: `best_params.pt`
- Último: `latest_params.pt`

## Entrenamiento en Cloud (RunPod)

### Setup en RunPod

1. Seleccionar instancia con GPU (A100 recomendado)
2. Clonar repositorio:
   ```bash
   git clone tu_repo
   cd emotune-transformer/src
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   pip install pytorch-fast-transformers
   ```

4. Subir datos:
   ```bash
   # Usar SCP o rsync
   rsync -av dataset/ root@runpod_ip:/workspace/dataset/
   ```

5. Ejecutar entrenamiento:
   ```bash
   nohup python workspace/transformer/main_cp.py \
     --mode train --batch_size 32 \
     --load_dict dataset/representation/train_dictionary.pkl \
     > training.log 2>&1 &
   ```

### Recuperar Modelo Entrenado

Se provee del siguiente script para un retrieving ordenado y rápido del modelo. 

```bash
# Descargar checkpoints
python workspace/others/scripts/upload_to_drive.py

# O usar SCP
scp -r root@runpod_ip:/workspace/dataset/models/checkpoints ./
```

## Resultados descargables
A continuación se brinda el enlace de descarga al mejor model obtenido durante el entrenamiento:
https://drive.google.com/file/d/12O4MM2_R7gA88OtAJSqSepUH-ETip5M4/view?usp=drive_link

> Este puede ser usado en conjunto con los resultados descargables del preprocesamiento.

## Siguiente Paso

Una vez entrenado el modelo, proceder a la [evaluación](evaluation.md) o directamente a la [generación](generation.md) de música.
