# ============================================================================
# MASTER MAKEFILE - TGA GPU PROJECT
# ============================================================================

NVCC = nvcc
FLAGS = -O3 -arch=sm_86

# --- OBJETIVOS ---

# Estrategias Simples / Estándar (Sin Templates)
S1_TARGETS = strategy_1/estrategia1_euclidean.exe strategy_1/estrategia1_cosine.exe strategy_1/estrategia1_pearson.exe
S2_TARGETS = strategy_2/estrategia2_euclidean.exe strategy_2/estrategia2_cosine.exe strategy_2/estrategia2_pearson.exe
S3_TARGETS = strategy_3/estrategia3_euclidean.exe strategy_3/estrategia3_cosine.exe strategy_3/estrategia3_pearson.exe
S5_TARGETS = strategy_5/estrategia5_euclidean.exe strategy_5/estrategia5_cosine.exe strategy_5/estrategia5_pearson.exe strategy_5/estrategia5_pearson_opt.exe
S6_TARGETS = strategy_6/estrategia6_euclidean.exe strategy_6/estrategia6_cosine.exe strategy_6/estrategia6_pearson.exe strategy_6/estrategia6_pearson_opt.exe

# Estrategia 7 (Estándar + Templates)
S7_DYN = strategy_7/estrategia7_euclidean.exe strategy_7/estrategia7_cosine.exe strategy_7/estrategia7_pearson.exe
S7_TMP = strategy_7/estrategia7_euclidean_temp.exe strategy_7/estrategia7_cosine_temp.exe strategy_7/estrategia7_pearson_temp.exe
S7_TARGETS = $(S7_DYN) $(S7_TMP)

# --- REGLAS ---

.PHONY: all clean s1 s2 s3 s5 s6 s7 help

all: s1 s2 s3 s5 s6 s7

s1: $(S1_TARGETS)
s2: $(S2_TARGETS)
s3: $(S3_TARGETS)
s5: $(S5_TARGETS)
s6: $(S6_TARGETS)
s7: $(S7_TARGETS)

# Regla genérica de compilación
%.exe: %.cu
	$(NVCC) $(FLAGS) $< -o $@

clean:
	del /S *.exe *.exp *.lib 2>NUL