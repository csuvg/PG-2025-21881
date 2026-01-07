# Guía de Evaluación del Modelo

## Descripción General

La evaluación mide qué tan bien el modelo genera música que respeta las condiciones de entrada (emoción y tonalidad). Se analizan las muestras generadas para calcular métricas de adherencia y calidad musical.

## Métricas Principales

- **Adherencia Emocional**: % de piezas que mantienen la emoción objetivo (esperado: 40-50%)
- **Adherencia Tonal**: % de piezas en la tonalidad correcta (esperado: 35-45%)
- **Correlación Tonal**: Similitud entre tonalidad generada y objetivo (objetivo > 0.7)
- **Métricas Musicales**: Densidad de notas, rango dinámico, coherencia armónica

## Scripts Disponibles

El proyecto incluye los siguientes scripts para evaluación en `workspace/transformer/`:

- **analyze_emotion_key.py**: Analiza adherencia emocional y tonal de las muestras generadas
- **analyze_adherence.py**: Evaluación detallada de adherencia a condiciones
- **main_cp.py**: Script principal que también puede generar muestras para evaluación

## Notebook de Evaluación

El análisis principal se realiza mediante el notebook interactivo:

```bash
cd workspace/
jupyter notebook evaluation.ipynb
```

### ¿Qué hace el notebook?

1. **Carga de datos**: Lee las muestras generadas (.mid, .npy, .json)
2. **Extracción de características**: Analiza propiedades musicales de cada pieza
3. **Cálculo de métricas**: Computa adherencia emocional, tonal y correlaciones
4. **Visualización**: Genera gráficos de distribuciones y matrices de confusión
5. **Reporte de resultados**: Tablas comparativas con todas las métricas

## Proceso de Evaluación

1. Primero, generar un conjunto de muestras de prueba usando el modelo (ver [guía de generación](generation.md))
2. Ejecutar el notebook `evaluation.ipynb` para análisis completo
3. Revisar las visualizaciones y métricas generadas
4. Interpretar resultados según los benchmarks esperados

## Resultados descargables
Los archivos generados durante la fase de evaluación los puedes encontrar en:
https://drive.google.com/file/d/15AmhpWkNWad7rQvBYHRnQ75pD9tWXszZ/view?usp=sharing

## Resultados Esperados

- **Modelo base**: 35-40% adherencia emocional, 30-35% adherencia tonal
- **Modelo optimizado**: 45-50% adherencia emocional, 38-42% adherencia tonal
- **Correlación tonal objetivo**: > 0.7

## Requisitos

Los requisitos son los mismos que para la [guía de generación](generation.md):
- Modelo entrenado y archivos de representación descargados
- Python 3.11+ con dependencias instaladas
- Muestras generadas en `output/` para evaluar

