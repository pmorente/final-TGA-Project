# Proyecto TGA - Comparación de Vectores One-to-Many

Proyecto de la asignatura **Tarjetas Gráficas y Aceleradores (TGA)** que implementa y compara diferentes estrategias de paralelización para el cálculo de similitud entre vectores en CPU y GPU.

## 📋 Descripción del Proyecto

Este proyecto implementa y compara diferentes métricas de similitud vectorial utilizando múltiples estrategias de paralelización tanto en CPU como en GPU. El objetivo es comparar el rendimiento de diferentes enfoques de paralelización para operaciones de comparación one-to-many (un vector de consulta contra múltiples vectores de base de datos).

### Algoritmos Implementados

- **Similitud del Coseno (Cosine Similarity)**: Mide el ángulo entre dos vectores
- **Distancia Euclidiana (Euclidean Distance)**: Mide la distancia geométrica entre dos vectores
- **Correlación de Pearson (Pearson Correlation)**: Mide la correlación lineal entre dos vectores

## 🏗️ Estructura del Proyecto

```
final-TGA-Project/
├── CPU/                          # Implementaciones en CPU
│   ├── compare_one_to_many.cpp  # Estrategia secuencial
│   ├── parallel_compare_one_to_many.cpp  # Paralelización a nivel de vector
│   ├── parallel_metric_compare_one_to_many.cpp  # Paralelización a nivel de dimensión
│   ├── cosine/, euclidean/, pearson/  # Implementaciones de métricas
│
├── GPU/                          # Implementaciones en GPU (CUDA)
│   ├── strategy_1/               # Grid sobre pares, secuencial por thread
│   ├── strategy_2/               # Grid sobre grupo pequeño, loop sobre grupo grande
│   ├── strategy_3/               # Híbrido: 2D tiling + reducción paralela
│   ├── strategy_5/               # Grid sobre pequeño, reducción sobre dimensión (templates)
│   ├── strategy_6/               # Grid sobre grande, loop sobre pequeño (templates)
│   └── strategy_7/               # 2D tiles con reutilización de datos
│
├── inputs/                       # Archivos de entrada (vectores binarios)
│   ├── 384/, 768/, 1024/        # Diferentes dimensiones
│
├── trust_files/                  # Resultados de referencia (CPU secuencial)
├── scripts_get_results_cpu/      # Scripts para benchmarks en CPU
└── scripts_get_results_gpu/      # Scripts para benchmarks en GPU
```

## 🚀 Instalación y Configuración

### Requisitos Previos

- **Sistema Operativo**: Linux (Ubuntu/WSL recomendado)
- **CPU**: Procesador multi-core con soporte OpenMP
- **GPU**: NVIDIA GPU con soporte CUDA (Compute Capability 7.5+)
- **Compiladores**: `g++` con OpenMP, `nvcc` (CUDA Toolkit)
- **Python 3.10+** (para scripts de análisis)

### Instalación

1. **Configurar entorno virtual de Python**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Instalar CUDA Toolkit** (si no está instalado):
```bash
sudo apt update
sudo apt install -y nvidia-cuda-toolkit
# O instalar desde NVIDIA (ver GPU/README.md)
```

3. **Verificar instalación**:
```bash
nvcc --version
nvidia-smi
```

### Configuración de Compute Capability

**Importante**: Debes compilar con la capacidad de cómputo correcta para tu GPU.

**Encontrar la capacidad de cómputo**:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

**Mapeo de Compute Capability**:
- **7.5** → `sm_75` (Turing: RTX 20-series, GTX 16-series)
- **8.0** → `sm_80` (Ampere: A100)
- **8.6** → `sm_86` (Ampere: RTX 30-series, **RTX 3050 Mobile**)
- **8.9** → `sm_89` (Ada: RTX 40-series)

**Para RTX 3050 Mobile**: Usa `sm_86` (ya configurado en el Makefile)

## 🔧 Compilación

### Compilar Implementaciones CPU

```bash
cd CPU

# Estrategia 1: Secuencial
g++ -O2 compare_one_to_many.cpp cosine/cosine.cpp pearson/pearson.cpp euclidean/euclidean.cpp -o compare_one_to_many

# Estrategia 2: Paralelo a nivel de vector
g++ -fopenmp -O2 parallel_compare_one_to_many.cpp cosine/cosine.cpp pearson/pearson.cpp euclidean/euclidean.cpp -o parallel_compare_one_to_many

# Estrategia 3: Paralelo a nivel de dimensión
g++ -fopenmp -O2 parallel_metric_compare_one_to_many.cpp cosine/cosine_parallel.cpp pearson/pearson_parallel.cpp euclidean/euclidean_parallel.cpp -o parallel_metric_compare_one_to_many
```

### Compilar Implementaciones GPU

**Usar el Makefile (recomendado)**:
```bash
cd GPU
make all          # Compila todas las estrategias
make s1           # Solo estrategia 1
make s2           # Solo estrategia 2
make s3           # Solo estrategia 3
make s5           # Solo estrategia 5
make s6           # Solo estrategia 6
make s7           # Solo estrategia 7
make clean        # Limpiar ejecutables
```

**Compilar manualmente**:
```bash
cd GPU/strategy_1
nvcc -O3 -arch=sm_xx estrategia1_cosine.cu -o estrategia1_cosine
# ... (repetir para otras estrategias)
```

**Nota**: Después de compilar, dar permisos de ejecución:
```bash
chmod +x GPU/strategy_*/estrategia*
```

## 📊 Estrategias de Implementación

### Estrategias CPU

1. **Estrategia 1 - Secuencial**: Sin paralelización, baseline para comparación
2. **Estrategia 2 - Paralelo a nivel de vector**: Paraleliza el bucle externo sobre vectores de consulta
3. **Estrategia 3 - Paralelo a nivel de dimensión**: Paraleliza el cálculo dentro de cada comparación

### Estrategias GPU (CUDA)

1. **Strategy 1**: Grid sobre pares, 1 thread por comparación (baseline)
2. **Strategy 2**: Grid sobre grupo pequeño (queries), loop sobre grupo grande (database)
3. **Strategy 3**: Híbrido con 2D tiling + reducción paralela dentro de cada comparación
4. **Strategy 5**: Grid sobre pequeño, reducción sobre dimensión (usa templates C++)
5. **Strategy 6**: Grid sobre grande, loop sobre pequeño (usa templates C++)
6. **Strategy 7**: 2D tiles con reutilización de datos en memoria compartida

Para más detalles, consulta:
- `CPU/README.md` - Estrategias CPU
- `GPU/README.md` - Estrategias GPU detalladas
- `GPU/STRATEGIES_PARAMETERS.md` - Parámetros y configuración

## 🎯 Ejecución

### Ejecución Manual

**CPU**:
```bash
cd CPU
./compare_one_to_many inputs/384/file_a.bin inputs/384/file_b.bin cosine
```

**GPU**:
```bash
cd GPU/strategy_2
./estrategia2_cosine ../../inputs/384/file_a.bin ../../inputs/384/file_b.bin 128
```

### Scripts de Benchmarking

#### Benchmarks Individuales (GPU)

```bash
cd scripts_get_results_gpu

# Strategy 1 (sin parámetros)
./run_strategy1_benchmark.sh

# Strategy 2, 3, 5, 6 (requieren block_size: 32, 64, 128, 256, 512, 1024)
./run_strategy2_benchmark.sh 128
./run_strategy3_benchmark.sh 128
./run_strategy5_benchmark.sh 128
./run_strategy6_benchmark.sh 128

# Strategy 7 (requiere tile_size: 4, 8, 16, 30, 31, 32)
./run_strategy7_benchmark.sh 30
```

#### Comparación de Block Sizes (GPU)

Estos scripts prueban múltiples tamaños de bloque y seleccionan automáticamente el mejor:

```bash
cd scripts_get_results_gpu

# Compara block sizes: 32, 64, 128, 256, 512, 1024
./run_strategy2_compare_blocksizes.sh
./run_strategy3_compare_blocksizes.sh
./run_strategy5_compare_blocksizes.sh
./run_strategy6_compare_blocksizes.sh

# Compara tile sizes: 4, 8, 16, 30, 31, 32
./run_strategy7_compare_tiles.sh
```

**Salida**: Al final verás un resumen mostrando el mejor block/tile size:
```
=== BEST BLOCK SIZE SUMMARY ===

Best block size (lowest average time) for each metric and dimension:

  Cosine:
    Dimension 384: Block Size 128 (Time: 0.123456s)
    Dimension 768: Block Size 256 (Time: 0.234567s)
    Dimension 1024: Block Size 128 (Time: 0.345678s)
  ...
```

#### Verificación de Resultados

Compara los resultados de GPU con los resultados de referencia (CPU):

```bash
cd scripts_get_results_gpu

# Comparar resultados de una estrategia con archivos de confianza
./compare_with_trust.sh ../GPU/strategy_1
./compare_with_trust.sh ../GPU/strategy_2
# ... (para otras estrategias)

# Con tolerancia personalizada
./compare_with_trust.sh ../GPU/strategy_1 --tolerance 1e-6
```

## 📈 Formato de Datos

### Archivos de Entrada

- **Tipo de dato**: `float32` (4 bytes por valor)
- **Estructura**: 
  - Número de registros (int32)
  - Dimensión de vectores (int32)
  - Datos de vectores (float32 array)

### Archivos de Resultados

- **Tipo de dato**: `float32` (4 bytes por valor)
- **Orden**: Row-major (índice = `query_idx * num_database + db_idx`)
- **Estructura**: Array plano de valores float32

## 📁 Archivos de Resultados

Los scripts generan archivos Markdown con los resultados:

- `scripts_get_results_gpu/strategy*_results*.md` - Resultados de benchmarks individuales
- `scripts_get_results_gpu/strategy*_blocksize_comparison.md` - Comparaciones de block sizes
- `scripts_get_results_gpu/strategy7_tilesize_comparison.md` - Comparación de tile sizes

## 🔍 Troubleshooting

### Error: "compute capability mismatch"
- Verifica: `nvidia-smi --query-gpu=compute_cap --format=csv`
- Actualiza `-arch=sm_XX` en el Makefile

### Error: "executable not found" o script se cuelga
- Dar permisos: `chmod +x GPU/strategy_*/estrategia*`
- Verificar que los ejecutables están compilados

### Error: "GLIBC version mismatch"
- Recompila los ejecutables en tu sistema actual

### Error: "file not found"
- Verifica que los archivos existen en `inputs/`
- Usa rutas absolutas si es necesario

## 📚 Documentación Adicional

- `CPU/README.md` - Documentación detallada de estrategias CPU
- `GPU/README.md` - Documentación detallada de estrategias GPU
- `GPU/STRATEGIES_PARAMETERS.md` - Guía de parámetros y configuración
- `scripts_get_results_gpu/README.md` - Guía de scripts de benchmarking

## 📝 Notas

- Los resultados de CPU secuencial se guardan en `trust_files/` como referencia
- Todos los scripts generan tablas de rendimiento en formato Markdown
- Los scripts de comparación automáticamente seleccionan el mejor block/tile size

