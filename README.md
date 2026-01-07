# Control multiparamétrico en generación automática de música: Extensión del sistema EMOPIA mediante información de tonalidad y métodos de inferencia especializados (EmoTune)

## Descripción

Este proyecto implementa un sistema de generación automática de música de piano condicionada por emociones y tonalidad musical utilizando arquitecturas Transformer. El sistema extiende el dataset EMOPIA original incorporando análisis de tonalidad mediante el algoritmo Krumhansl-Schmuckler, permitiendo un control multidimensional sobre las características musicales generadas: estado emocional (4 cuadrantes: feliz, enojado, triste, relajado) y tonalidad musical (24 tonalidades mayores y menores) y demás características musicales.

El modelo procesa secuencias musicales tokenizadas con 9 dimensiones incluyendo tempo, acordes, compás, tipo de evento, altura, duración, velocidad, emoción y tonalidad, logrando generar composiciones coherentes con tasas de adherencia categórica de hasta 50% para emoción y 41.67% para tonalidad.

## Tecnologías Utilizadas

- Python 3.9+ / Python 3.11 (para diferentes módulos)
- PyTorch 2.8+
- Fast Transformers
- MidiToolkit
- Librosa & Madmom (procesamiento de audio)
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (visualización)
- CUDA (aceleración GPU)

## Requisitos Previos

- Python 3.9 y Python 3.11 instalados
- CUDA 11.8+ con GPU NVIDIA compatible
- Visual Studio Build Tools (Windows)
- FFmpeg (para procesamiento de audio)
- yt-dlp (descarga de clips de YouTube)
- Mínimo 8GB RAM (recomendado 16GB+)
- ~10GB espacio en disco para datasets

## Instalación Rápida

1. Clonar el repositorio
2. Instalar PyTorch con CUDA (https://pytorch.org/get-started/locally/)
3. Instalar dependencias: `pip install -r src/requirements.txt`
4. Instalar Fast Transformers: `pip install pytorch-fast-transformers`

Para instrucciones detalladas de instalación y configuración, consultar las guías específicas en la carpeta `src/guides/`.

## Uso Rápido

Para usar el sistema, seguir las guías en orden:

1. **[Preprocesamiento](src/guides/preprocessing.md)**: Preparar el dataset EMOPIA
2. **[Entrenamiento](src/guides/training.md)**: Entrenar el modelo Transformer
3. **[Generación](src/guides/generation.md)**: Generar música con control emocional y tonal
4. **[Evaluación](src/guides/evaluation.md)**: Analizar la calidad de las generaciones

### Inicio Rápido

Si ya tienes el modelo entrenado descargado, puedes generar música directamente ejecutando:

```bash
python src/demo.py  # Windows
```

> Recuerda modificar la variable del path.

## Demo

El video demostrativo del sistema se encuentra en [`/demo/demo.mp4`](demo/demo.mp4)

## Documentación

- **Informe final del proyecto**: [`/docs/informe_final.pdf`](docs/informe_final.pdf)
- **Guías detalladas**:
  - [Preprocesamiento de datos](src/guides/preprocessing.md) - Pipeline completo de preparación del dataset
  - [Entrenamiento del modelo](src/guides/training.md) - Configuración y entrenamiento del Transformer
  - [Generación de música](src/guides/generation.md) - Cómo generar música con el modelo entrenado
  - [Evaluación de resultados](src/guides/evaluation.md) - Análisis de calidad y adherencia

### Archivos Descargables

Para evitar el preprocesamiento y entrenamiento, puedes descargar directamente:
- **Modelo entrenado**: Ver [guía de generación](src/guides/generation.md#archivos-requeridos-descargar-de-fases-previas)
- **Dataset procesado**: Enlaces disponibles en [guía de preprocesamiento](src/guides/preprocessing.md#resultados-descargables)

## Estructura del Proyecto

```
PG-2025-21881/
├── demo/                  # Video demostrativo
├── docs/                  # Documentación del proyecto
│   └── informe_final.pdf  # Informe final
└── src/                   # Código fuente
    ├── dataset/           # Pipeline de preprocesamiento
    │   ├── preprocessing/ # Scripts de conversión MIDI
    │   ├── representation/# Datos tokenizados
    │   └── models/       # Checkpoints del modelo
    ├── workspace/
    │   ├── transformer/  # Implementación del modelo
    │   └── evaluation.ipynb  # Notebooks de evaluación
    ├── guides/           # Guías de uso
    └── output/           # Carpeta para resultados
```

> Los archivos midi y de metadata generados utilizados en el proyecto los puedes descargar en https://drive.google.com/file/d/15AmhpWkNWad7rQvBYHRnQ75pD9tWXszZ/view?usp=sharing

## Descargables

Todos los archivos descargables se encuentran el siguiente directorio: https://drive.google.com/drive/folders/1rV5y_OCeXRdz8pMgpI4U2SZiSsB58yOp?usp=sharing

## Resultados Clave

- **Modelo**: Transformer de 39.2M parámetros
- **Entrenamiento**: 875 épocas, convergencia 98.46% en tonalidad
- **Adherencia emocional**: Hasta 50% precisión categórica
- **Adherencia tonal**: Hasta 41.67% precisión, correlación 0.76
- **Modos de inferencia**: Normal, determinístico, primed, forced

## Autores

Samuel Chamalé - 21881

## Licencia

MIT License - Ver archivo [LICENSE](LICENSE) para más detalles