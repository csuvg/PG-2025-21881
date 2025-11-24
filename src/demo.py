import os
import subprocess
import sys

"""
Script de demostración para ejecutar múltiples configuraciones del generador de música.

Uso:
    python demo.py              # Ejecuta todos los comandos
    python demo.py 1            # Ejecuta solo el comando 1
    python demo.py 1 3          # Ejecuta los comandos 1 y 3
    
Configuraciones:
    1. Note_Duration_240, E:min
    2. Note_Duration_1800, E:min
    3. Note_Duration_1800, G:min
    4. Note_Duration_1800, C:maj
"""

# Define la ruta base como variable
BASE_PATH = r"path/to/your/project"

# Define los comandos usando la variable BASE_PATH
commands = [
    # Comando 1: Note_Duration_240, E:min
    f'python {BASE_PATH}\\workspace\\transformer\\main_cp.py '
    f'--data_root {BASE_PATH}\\dataset\\representation '
    f'--exp_path {BASE_PATH}\\dataset\\models '
    f'--mode inference-deterministic '
    f'--task_type 4-cls '
    f'--num_songs 1 '
    f'--emo_tag 4 '
    f'--save_tokens_json 1 '
    f'--load_ckt checkpoints '
    f'--load_ckt_loss 8 '
    f'--load_dict train_dictionary.pkl '
    f'--conditions "{{\\\"duration\\\": \\\"Note_Duration_240\\\"}}" '
    f'--key_tag E:min '
    f'--out_dir {BASE_PATH}\\output\\files\\1',
    
    # Comando 2: Note_Duration_1800, E:min
    f'python {BASE_PATH}\\workspace\\transformer\\main_cp.py '
    f'--data_root {BASE_PATH}\\dataset\\representation '
    f'--exp_path {BASE_PATH}\\dataset\\models '
    f'--mode inference-deterministic '
    f'--task_type 4-cls '
    f'--num_songs 1 '
    f'--emo_tag 4 '
    f'--save_tokens_json 1 '
    f'--load_ckt checkpoints '
    f'--load_ckt_loss 8 '
    f'--load_dict train_dictionary.pkl '
    f'--conditions "{{\\\"duration\\\": \\\"Note_Duration_1800\\\"}}" '
    f'--key_tag E:min '
    f'--out_dir {BASE_PATH}\\output\\files\\2',
    
    # Comando 3: Note_Duration_1800, G:min
    f'python {BASE_PATH}\\workspace\\transformer\\main_cp.py '
    f'--data_root {BASE_PATH}\\dataset\\representation '
    f'--exp_path {BASE_PATH}\\dataset\\models '
    f'--mode inference-deterministic '
    f'--task_type 4-cls '
    f'--num_songs 1 '
    f'--emo_tag 4 '
    f'--save_tokens_json 1 '
    f'--load_ckt checkpoints '
    f'--load_ckt_loss 8 '
    f'--load_dict train_dictionary.pkl '
    f'--conditions "{{\\\"duration\\\": \\\"Note_Duration_1800\\\"}}" '
    f'--key_tag G:min '
    f'--out_dir {BASE_PATH}\\output\\files\\3',
    
    # Comando 4: Note_Duration_1800, C:maj
    f'python {BASE_PATH}\\workspace\\transformer\\main_cp.py '
    f'--data_root {BASE_PATH}\\dataset\\representation '
    f'--exp_path {BASE_PATH}\\dataset\\models '
    f'--mode inference-deterministic '
    f'--task_type 4-cls '
    f'--num_songs 1 '
    f'--emo_tag 4 '
    f'--save_tokens_json 1 '
    f'--load_ckt checkpoints '
    f'--load_ckt_loss 8 '
    f'--load_dict train_dictionary.pkl '
    f'--conditions "{{\\\"duration\\\": \\\"Note_Duration_1800\\\"}}" '
    f'--key_tag C:maj '
    f'--out_dir {BASE_PATH}\\output\\files\\4'
]

def run_command(cmd_index, cmd):
    """Ejecuta un comando individual con streaming de salida"""
    print(f"\n{'='*60}")
    print(f"Ejecutando comando {cmd_index}...")
    print(f"{'='*60}\n")
    
    # Crea el directorio de salida si no existe
    out_dir = f"{BASE_PATH}\\output\\files\\{cmd_index}"
    os.makedirs(out_dir, exist_ok=True)
    
    # Ejecuta el comando con streaming de salida en tiempo real
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True
    )
    
    # Lee y muestra la salida línea por línea en tiempo real
    for line in process.stdout:
        print(line, end='')  # Ya incluye el salto de línea
        sys.stdout.flush()  # Fuerza la salida inmediata
    
    # Espera a que termine el proceso
    process.wait()
    
    if process.returncode == 0:
        print(f"\n✓ Comando {cmd_index} ejecutado exitosamente")
        return True
    else:
        print(f"\n✗ Error en comando {cmd_index} (código de salida: {process.returncode})")
        return False

# Ejecuta los comandos
if __name__ == "__main__":
    # Opcional: permite ejecutar comandos específicos pasando argumentos
    if len(sys.argv) > 1:
        # Ejecuta solo los comandos especificados
        indices = [int(arg) for arg in sys.argv[1:] if arg.isdigit()]
        if indices:
            print(f"Ejecutando comandos: {indices}")
            for idx in indices:
                if 1 <= idx <= len(commands):
                    run_command(idx, commands[idx-1])
                else:
                    print(f"⚠ Comando {idx} no existe (rango válido: 1-{len(commands)})")
        else:
            print("Uso: python demo.py [número_comando...]")
            print("Ejemplo: python demo.py 1 3  # Ejecuta comandos 1 y 3")
            print("Sin argumentos ejecuta todos los comandos")
    else:
        # Ejecuta todos los comandos
        print("Ejecutando todos los comandos...")
        success_count = 0
        
        for i, cmd in enumerate(commands, 1):
            success = run_command(i, cmd)
            if success:
                success_count += 1
            # Opcional: detener si hay un error
            # else:
            #     break
        
        print(f"\n{'='*60}")
        print(f"Proceso completado: {success_count}/{len(commands)} comandos exitosos")
        print(f"{'='*60}")