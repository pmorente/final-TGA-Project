import os
import numpy as np

def compare_and_print(cpu, gpu, cpu_label, gpu_label, tolerance=1e-7):
    print(f"\nComparando {cpu_label} vs {gpu_label}:")
    if cpu.shape != gpu.shape:
        print(f"¡Tamaños distintos! {cpu_label}: {cpu.shape}, {gpu_label}: {gpu.shape}")
        return
    if np.allclose(cpu, gpu, atol=tolerance):
        print("¡Los resultados coinciden dentro de la tolerancia numérica!")
    else:
        dif = np.abs(cpu - gpu)
        print("Hay diferencias.")
        print(f"Máxima diferencia: {dif.max()}")
        print(f"Índice de máxima diferencia: {dif.argmax()}")
        print(f"Valor {cpu_label}: {cpu[dif.argmax()]}, Valor {gpu_label}: {gpu[dif.argmax()]}")
        print("\nPrimeros 10 pares diferentes:")
        for i in np.where(dif > tolerance)[0][:10]:
            print(f"Índice {i}: {cpu_label}={cpu[i]}, {gpu_label}={gpu[i]}, Dif={dif[i]}")

cpu_results_cosine = np.fromfile('CPU/cpu_results_cosine.bin', dtype=np.float32)
cpu_results_euclidean = np.fromfile('CPU/cpu_results_euclidean.bin', dtype=np.float32)
cpu_results_pearson = np.fromfile('CPU/cpu_results_pearson.bin', dtype=np.float32)
gpu_results_cosine = np.fromfile('GPU/gpu_results_cosine.bin', dtype=np.float32)
gpu_results_euclidean = np.fromfile('GPU/gpu_results_euclidean.bin', dtype=np.float32)
gpu_results_pearson = np.fromfile('GPU/gpu_results_pearson.bin', dtype=np.float32)
gpu_results_pearson_opt = None
if os.path.exists('GPU/gpu_results_pearson_opt.bin'):
    gpu_results_pearson_opt = np.fromfile('GPU/gpu_results_pearson_opt.bin', dtype=np.float32)

compare_and_print(cpu_results_cosine, gpu_results_cosine, 'CPU Cosine', 'GPU Cosine')
compare_and_print(cpu_results_euclidean, gpu_results_euclidean, 'CPU Euclidean', 'GPU Euclidean')
compare_and_print(cpu_results_pearson, gpu_results_pearson, 'CPU Pearson', 'GPU Pearson')
if gpu_results_pearson_opt is not None:
    compare_and_print(cpu_results_pearson, gpu_results_pearson_opt, 'CPU Pearson', 'GPU Pearson Opt')