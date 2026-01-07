# Guía de Preprocesamiento de Datos

Esta guía detalla el pipeline completo para preparar el dataset EMOPIA para entrenamiento.

## Resumen del Pipeline

El preprocesamiento convierte archivos MIDI del dataset EMOPIA en representaciones tokenizadas que el modelo Transformer puede procesar. El pipeline completo incluye:

1. **Obtención del dataset original** (MIDI + metadatos)
2. **Descarga de audio MP3** para sincronización
3. **Sincronización y análisis** de archivos MIDI
4. **Extracción de características** (notas, acordes, tonalidad)
5. **Tokenización** en eventos compound-word
6. **Compilación** en formato NumPy para entrenamiento

## Requisitos Específicos

- Python 3.9 (requerido para madmom)
- FFmpeg instalado y en PATH
- yt-dlp para descargas de YouTube
- ~5GB de espacio libre

## Paso 1: Obtener Dataset EMOPIA

1. Descargar EMOPIA v2.2 desde [Zenodo](https://zenodo.org/records/5257995)
2. Extraer los archivos MIDI:
   ```bash
   # Estructura esperada:
   # S:\EMOPIA_2.2\EMOPIA_2.2\midis\
   # Deberías ver: "Found 1071 MIDI clips"
   ```

## Paso 2: Configurar Entorno Python 3.9

```bash
# Crear entorno virtual con Python 3.9
py -3.9 -m venv venv_py39
venv_py39\Scripts\activate

# Instalar dependencias específicas
pip install --upgrade setuptools wheel
pip install numpy==1.19.5  # Versión específica para madmom
pip install scipy cython
pip install librosa madmom 
pip install miditoolkit
pip install chorder
```

## Paso 3: Descargar Audio MP3 (para sincronización)

1. Instalar herramientas:
   ```bash
   pip install yt-dlp
   winget install ffmpeg  # Windows
   ```

2. Ejecutar script de descarga:
   ```bash
   python workspace/others/scripts/download_timestamp_clips.py \
     --ffmpeg-location "ruta/a/ffmpeg/bin" \
     --ytdlp yt-dlp
   ```
   
   Resultado esperado:
   - Downloads: ~320 exitosos de 387 videos
   - Clips: 854 generados exitosamente

## Paso 4: Sincronización y Análisis

1. Clonar repositorio compound-word-transformer:
   ```bash
   git clone https://github.com/YatingMusic/compound-word-transformer
   cd compound-word-transformer/dataset
   ```

2. Organizar archivos:
   ```bash
   # Mover MIDIs a:
   midi_transcribed/
   
   # Mover MP3s a:
   mp3/
   ```

3. Ejecutar sincronización (usa versión modificada):
   ```bash
   python workspace/others/modifications/compound-word-transformer/synchronizer.py
   ```

4. Ejecutar análisis (versión modificada, omite tempo 0):
   ```bash
   python workspace/others/modifications/compound-word-transformer/analyzer.py
   ```

5. Mover resultados:
   ```bash
   # Mover MIDIs analizados a:
   dataset/pre/midis/
   ```

## Paso 5: Pipeline de Tokenización

Cambiar al directorio `dataset/preprocessing/` y ejecutar en orden:

### 5.1. Convertir MIDI a Corpus

```bash
python midi2corpus.py ./pre/midis ./pre/corpus
```

Esto extrae:
- Notas cuantizadas (resolución 480 ticks)
- Acordes detectados
- Tempo y compás
- **Tonalidad global** vía Krumhansl-Schmuckler
- Metadatos de emoción

Salida: archivos `.pkl` en `./pre/corpus/`

### 5.2. Corpus a Eventos

```bash
python corpus2events.py
```

Convierte el corpus en secuencias de eventos compound-word con 9 dimensiones:
- tempo, chord, bar-beat, type
- pitch, duration, velocity
- emotion, **key** (nueva dimensión)

Salida: archivos `.pkl` en `./pre/events/`

### 5.3. Eventos a Palabras

```bash
python event2words.py
```

Crea el diccionario de vocabulario y mapea eventos a índices:
- Genera `train_dictionary.pkl`
- Crea archivos de palabras tokenizadas

Salida: `./pre/words/` y diccionario

### 5.4. Compilar a NumPy

```bash
python compile.py
```

Genera arrays NumPy para entrenamiento eficiente:
- `train_data.npz`: secuencias tokenizadas
- `train_idx.npz`: índices de entrenamiento
- Verifica longitud 1024 tokens por secuencia

> Lazy option: usar el pipeline.ipynb

## Estructura de Salida

```
dataset/
├── pre/
│   ├── midis/        # MIDIs analizados (entrada)
│   ├── corpus/       # Representación intermedia
│   ├── events/       # Eventos tokenizados
│   └── words/        # Palabras mapeadas
└── representation/
    ├── train_data.npz       # Datos de entrenamiento
    ├── train_dictionary.pkl # Diccionario vocabulario
    └── train_idx.npz        # Índices de secuencias
```

## Resultados descargables

Los resultados de estos pasos de ejecucion en el proyecto se pueden ser descargados en: 

MP3+MIDIS: https://drive.google.com/file/d/15AmhpWkNWad7rQvBYHRnQ75pD9tWXszZ/view?usp=sharing
> Evita la necesidad de correr el proceso de descarga y cualquier parte intermedia de pre-procesamiento

Representation: https://drive.google.com/file/d/1HvPVjTb8I4BergqWIi_vTNzvPsuEZJg0/view?usp=sharing
> Los resultados especfíficos de esta fase. Evita la ejecución completa del preprocesamiento.

## Siguiente Paso

Con los datos preprocesados, continuar con el [entrenamiento del modelo](training.md).
