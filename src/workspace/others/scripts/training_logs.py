import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_training_logs(log_file_path):
    """
    Parsea el archivo de logs de entrenamiento y extrae:
    - epoch loss
    - epoch each loss (componentes: Tempo, Chord, BarBeat, Type, Pitch, Duration, Velocity, Emotion, Key [opcional])
    - número de época (contado secuencialmente, no el batch number)
    - tiempo
    """
    epoch_losses = []
    epoch_each_losses = []
    epoch_numbers = []
    times = []
    
    # Contador de épocas reales (1, 2, 3, ...)
    epoch_counter = 0
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Parsear epoch loss
            if line.startswith('epoch loss |'):
                parts = line.split('|')
                if len(parts) >= 4:
                    try:
                        loss = float(parts[1].strip())
                        # El segundo número es batch number, NO usarlo como época
                        time = float(parts[3].strip())
                        
                        # Incrementar contador de épocas (empieza en 1)
                        epoch_counter += 1
                        
                        epoch_losses.append(loss)
                        epoch_numbers.append(epoch_counter)  # Usar contador secuencial
                        times.append(time)
                    except ValueError:
                        continue
            
            # Parsear epoch each loss (sigue después de epoch loss)
            elif line.startswith('epoch each loss |'):
                parts = line.split('|')
                if len(parts) >= 2:
                    try:
                        loss_str = parts[1].strip()
                        # Separar los valores por coma
                        losses = [float(x.strip()) for x in loss_str.split(',')]
                        # Aceptar 8 o 9 componentes (8 sin key, 9 con key)
                        if len(losses) == 8 or len(losses) == 9:
                            epoch_each_losses.append(losses)
                    except ValueError:
                        continue
    
    return {
        'epoch_losses': np.array(epoch_losses),
        'epoch_each_losses': np.array(epoch_each_losses),
        'epoch_numbers': np.array(epoch_numbers),
        'times': np.array(times)
    }

def create_visualizations(data, output_dir='results'):
    """
    Crea múltiples gráficos con insights del entrenamiento
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    epoch_losses = data['epoch_losses']
    epoch_each_losses = data['epoch_each_losses']
    epoch_numbers = data['epoch_numbers']
    times = data['times']
    
    # Determinar número de componentes dinámicamente
    n_components = epoch_each_losses.shape[1] if len(epoch_each_losses) > 0 else 8
    
    # Nombres de los componentes del loss (en orden)
    component_names = ['Tempo', 'Chord', 'BarBeat', 'Type', 'Pitch', 'Duration', 'Velocity', 'Emotion']
    if n_components == 9:
        component_names.append('Key')
    
    # Crear figura con múltiples subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Evolución del epoch loss total
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(epoch_numbers, epoch_losses, 'b-', linewidth=2, alpha=0.7, label='Epoch Loss')
    # Línea de tendencia
    z = np.polyfit(epoch_numbers, epoch_losses, 2)
    p = np.poly1d(z)
    ax1.plot(epoch_numbers, p(epoch_numbers), "r--", alpha=0.5, linewidth=1, label='Tendencia')
    ax1.set_xlabel('Época', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Evolución del Epoch Loss Total', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Evolución del epoch loss en escala logarítmica
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogy(epoch_numbers, epoch_losses, 'b-', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Época', fontsize=10)
    ax2.set_ylabel('Loss (log scale)', fontsize=10)
    ax2.set_title('Epoch Loss (Escala Log)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Evolución de cada componente del epoch each loss
    ax3 = fig.add_subplot(gs[1, 1:])
    colors = plt.cm.tab10(np.linspace(0, 1, n_components))
    for i in range(n_components):
        if len(epoch_each_losses) > 0:
            ax3.plot(epoch_numbers[:len(epoch_each_losses)], 
                    epoch_each_losses[:, i], 
                    color=colors[i], 
                    linewidth=1.5, 
                    alpha=0.7, 
                    label=component_names[i])
    ax3.set_xlabel('Época', fontsize=10)
    ax3.set_ylabel('Loss', fontsize=10)
    ax3.set_title('Evolución de Cada Componente del Loss', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8, ncol=2)
    
    # 4. Comparación de componentes (promedio)
    ax4 = fig.add_subplot(gs[2, 0])
    if len(epoch_each_losses) > 0:
        component_means = np.mean(epoch_each_losses, axis=0)
        component_std = np.std(epoch_each_losses, axis=0)
        bars = ax4.bar(range(n_components), component_means, yerr=component_std, 
                      color=colors, alpha=0.7, capsize=5)
        ax4.set_xlabel('Componente', fontsize=10)
        ax4.set_ylabel('Loss Promedio', fontsize=10)
        ax4.set_title('Promedio y Desviación de Componentes', fontsize=11, fontweight='bold')
        ax4.set_xticks(range(n_components))
        ax4.set_xticklabels(component_names, rotation=45, ha='right', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Reducción relativa del loss
    ax5 = fig.add_subplot(gs[2, 1])
    if len(epoch_losses) > 1:
        initial_loss = epoch_losses[0]
        relative_reduction = ((initial_loss - epoch_losses) / initial_loss) * 100
        ax5.plot(epoch_numbers, relative_reduction, 'g-', linewidth=2, alpha=0.7)
        ax5.set_xlabel('Época', fontsize=10)
        ax5.set_ylabel('Reducción Relativa (%)', fontsize=10)
        ax5.set_title('Reducción Relativa del Loss', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 6. Velocidad de entrenamiento (loss por tiempo)
    ax6 = fig.add_subplot(gs[2, 2])
    if len(times) > 1:
        time_diff = np.diff(times)
        loss_diff = np.diff(epoch_losses)
        # Loss por segundo (negativo porque queremos que el loss disminuya)
        rate = -loss_diff / time_diff
        ax6.plot(epoch_numbers[1:], rate, 'm-', linewidth=1.5, alpha=0.7)
        ax6.set_xlabel('Época', fontsize=10)
        ax6.set_ylabel('Tasa de Reducción\n(Loss/s)', fontsize=10)
        ax6.set_title('Velocidad de Mejora del Loss', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Análisis de Entrenamiento - Visualización Completa', 
                 fontsize=15, fontweight='bold', y=0.995)
    
    output_path = output_dir / 'training_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Grafico principal guardado en: {output_path}")
    
    # Crear gráfico adicional con evolución detallada de componentes
    # Ajustar layout dinámicamente según número de componentes
    if n_components <= 8:
        n_rows, n_cols = 2, 4
        figsize = (24, 14)  # Mayor tamaño para más espacio por gráfica
    else:
        n_rows, n_cols = 3, 3
        figsize = (24, 16)  # Mayor tamaño para más espacio por gráfica
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig2.subplots_adjust(hspace=0.4, wspace=0.35)  # Más espaciado entre subplots
    axes = axes.flatten()
    
    for i in range(n_components):
        if len(epoch_each_losses) > 0:
            ax = axes[i]
            ax.plot(epoch_numbers[:len(epoch_each_losses)], 
                   epoch_each_losses[:, i], 
                   color=colors[i], 
                   linewidth=2)
            ax.set_xlabel('Época', fontsize=11)
            ax.set_ylabel('Loss', fontsize=11)
            ax.set_title(component_names[i], fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=10)  # Aumentar tamaño de ticks
            
            # Agregar estadísticas en el gráfico
            min_val = np.min(epoch_each_losses[:, i])
            max_val = np.max(epoch_each_losses[:, i])
            mean_val = np.mean(epoch_each_losses[:, i])
            final_val = epoch_each_losses[-1, i]
            
            stats_text = f'Min: {min_val:.4f}\nMax: {max_val:.4f}\nProm: {mean_val:.4f}\nFinal: {final_val:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Ocultar subplots vacíos si hay menos componentes que subplots
    for i in range(n_components, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Evolución Detallada de Cada Componente del Loss', 
                 fontsize=16, fontweight='bold')
    
    output_path2 = output_dir / 'training_components_detail.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"[OK] Grafico detallado de componentes guardado en: {output_path2}")
    
    # Crear gráfico de estadísticas generales
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Estadísticas del epoch loss
    ax1 = axes[0, 0]
    ax1.plot(epoch_numbers, epoch_losses, 'b-', linewidth=2, alpha=0.7)
    ax1.axhline(y=np.mean(epoch_losses), color='r', linestyle='--', 
               label=f'Promedio: {np.mean(epoch_losses):.4f}')
    ax1.axhline(y=np.min(epoch_losses), color='g', linestyle='--', 
               label=f'Mínimo: {np.min(epoch_losses):.4f}')
    ax1.fill_between(epoch_numbers, 
                     np.mean(epoch_losses) - np.std(epoch_losses),
                     np.mean(epoch_losses) + np.std(epoch_losses),
                     alpha=0.2, color='blue', label='±1 Desv. Est.')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.set_title('Epoch Loss con Estadísticas', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribución de mejoras
    ax2 = axes[0, 1]
    if len(epoch_losses) > 1:
        improvements = -np.diff(epoch_losses)  # Cambios positivos = mejoras
        ax2.hist(improvements, bins=30, color='green', alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.mean(improvements), color='r', linestyle='--', 
                   label=f'Promedio: {np.mean(improvements):.6f}')
        ax2.set_xlabel('Mejora por Época')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Mejoras del Loss', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Comparación inicial vs final de componentes
    ax3 = axes[1, 0]
    if len(epoch_each_losses) > 0:
        initial_components = epoch_each_losses[0]
        final_components = epoch_each_losses[-1]
        x = np.arange(n_components)
        width = 0.35
        ax3.bar(x - width/2, initial_components, width, label='Época Inicial', 
               color='red', alpha=0.7)
        ax3.bar(x + width/2, final_components, width, label='Época Final', 
               color='green', alpha=0.7)
        ax3.set_xlabel('Componente')
        ax3.set_ylabel('Loss')
        ax3.set_title('Comparación Inicial vs Final de Componentes', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(component_names, rotation=45, ha='right', fontsize=9)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Tiempo de entrenamiento
    ax4 = axes[1, 1]
    if len(times) > 1:
        cumulative_time = times - times[0]  # Tiempo desde el inicio
        ax4.plot(epoch_numbers, cumulative_time, 'purple', linewidth=2, alpha=0.7)
        ax4.set_xlabel('Época')
        ax4.set_ylabel('Tiempo Acumulado (s)')
        ax4.set_title('Tiempo de Entrenamiento', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Calcular tiempo promedio por época
        avg_time_per_epoch = np.mean(np.diff(cumulative_time))
        ax4.text(0.02, 0.98, f'Tiempo promedio/época: {avg_time_per_epoch:.2f}s', 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle('Análisis Estadístico del Entrenamiento', 
                 fontsize=16, fontweight='bold')
    
    output_path3 = output_dir / 'training_statistics.png'
    plt.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"[OK] Grafico estadistico guardado en: {output_path3}")
    
    plt.close('all')
    
    # Generar reporte completo en texto
    generate_text_report(data, component_names, output_dir)
    
    # Imprimir resumen estadístico
    print("\n" + "="*60)
    print("RESUMEN ESTADÍSTICO DEL ENTRENAMIENTO")
    print("="*60)
    print(f"Total de épocas: {len(epoch_numbers)}")
    print(f"Época inicial: {epoch_numbers[0]}")
    print(f"Época final: {epoch_numbers[-1]}")
    print(f"\nEpoch Loss:")
    print(f"  Inicial: {epoch_losses[0]:.6f}")
    print(f"  Final: {epoch_losses[-1]:.6f}")
    print(f"  Reducción: {((epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100):.2f}%")
    print(f"  Mínimo: {np.min(epoch_losses):.6f}")
    print(f"  Promedio: {np.mean(epoch_losses):.6f}")
    print(f"  Desviación estándar: {np.std(epoch_losses):.6f}")
    
    if len(epoch_each_losses) > 0:
        print(f"\nEpoch Each Loss (Componentes):")
        for i in range(n_components):
            initial = epoch_each_losses[0, i]
            final = epoch_each_losses[-1, i]
            reduction = ((initial - final) / initial * 100) if initial > 0 else 0
            print(f"  {component_names[i]}: {initial:.6f} -> {final:.6f} ({reduction:.2f}% reduccion)")
    
    if len(times) > 1:
        total_time = times[-1] - times[0]
        print(f"\nTiempo de entrenamiento: {total_time:.2f} segundos ({total_time/3600:.2f} horas)")
        print(f"Tiempo promedio por época: {np.mean(np.diff(times)):.2f} segundos")
    
    print("="*60)

def generate_text_report(data, component_names, output_dir='results'):
    """
    Genera un reporte completo en texto con todas las estadísticas mostradas en las gráficas
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    epoch_losses = data['epoch_losses']
    epoch_each_losses = data['epoch_each_losses']
    epoch_numbers = data['epoch_numbers']
    times = data['times']
    n_components = len(component_names)
    
    report_lines = []
    
    # Encabezado
    report_lines.append("="*80)
    report_lines.append("REPORTE COMPLETO DE ANÁLISIS DE ENTRENAMIENTO")
    report_lines.append("="*80)
    report_lines.append("")
    
    # 1. Información General
    report_lines.append("1. INFORMACIÓN GENERAL DEL ENTRENAMIENTO")
    report_lines.append("-" * 80)
    report_lines.append(f"Total de épocas: {len(epoch_numbers)}")
    report_lines.append(f"Época inicial: {epoch_numbers[0]}")
    report_lines.append(f"Época final: {epoch_numbers[-1]}")
    
    if len(times) > 1:
        total_time = times[-1] - times[0]
        avg_time_per_epoch = np.mean(np.diff(times))
        report_lines.append(f"Tiempo total de entrenamiento: {total_time:.2f} segundos ({total_time/3600:.2f} horas)")
        report_lines.append(f"Tiempo promedio por época: {avg_time_per_epoch:.2f} segundos")
        report_lines.append(f"Tiempo por época - Mínimo: {np.min(np.diff(times)):.2f} segundos")
        report_lines.append(f"Tiempo por época - Máximo: {np.max(np.diff(times)):.2f} segundos")
        report_lines.append(f"Tiempo por época - Desviación estándar: {np.std(np.diff(times)):.2f} segundos")
    
    report_lines.append("")
    
    # 2. Epoch Loss Total - Estadísticas Completas
    report_lines.append("2. EPOCH LOSS TOTAL - ESTADÍSTICAS COMPLETAS")
    report_lines.append("-" * 80)
    initial_loss = epoch_losses[0]
    final_loss = epoch_losses[-1]
    reduction_pct = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
    
    report_lines.append(f"Inicial (Época {epoch_numbers[0]}): {initial_loss:.6f}")
    report_lines.append(f"Final (Época {epoch_numbers[-1]}): {final_loss:.6f}")
    report_lines.append(f"Reducción absoluta: {initial_loss - final_loss:.6f}")
    report_lines.append(f"Reducción relativa: {reduction_pct:.2f}%")
    report_lines.append(f"Mínimo: {np.min(epoch_losses):.6f} (Época {epoch_numbers[np.argmin(epoch_losses)]})")
    report_lines.append(f"Máximo: {np.max(epoch_losses):.6f} (Época {epoch_numbers[np.argmax(epoch_losses)]})")
    report_lines.append(f"Promedio: {np.mean(epoch_losses):.6f}")
    report_lines.append(f"Mediana: {np.median(epoch_losses):.6f}")
    report_lines.append(f"Desviación estándar: {np.std(epoch_losses):.6f}")
    report_lines.append(f"Rango: {np.max(epoch_losses) - np.min(epoch_losses):.6f}")
    
    # Reducción relativa por época
    if len(epoch_losses) > 1:
        relative_reduction = ((initial_loss - epoch_losses) / initial_loss) * 100
        report_lines.append(f"\nReducción relativa por época:")
        report_lines.append(f"  Final: {relative_reduction[-1]:.2f}%")
        report_lines.append(f"  Promedio: {np.mean(relative_reduction):.2f}%")
        report_lines.append(f"  Máxima: {np.max(relative_reduction):.2f}%")
    
    report_lines.append("")
    
    # 3. Estadísticas de Mejoras del Loss Total
    if len(epoch_losses) > 1:
        report_lines.append("3. ESTADÍSTICAS DE MEJORAS DEL LOSS TOTAL (por época)")
        report_lines.append("-" * 80)
        improvements = -np.diff(epoch_losses)  # Cambios positivos = mejoras
        positive_improvements = improvements[improvements > 0]
        negative_improvements = improvements[improvements <= 0]  # Empeoramientos
        
        report_lines.append(f"Total de mejoras (cambios positivos): {len(positive_improvements)} épocas")
        report_lines.append(f"Total de empeoramientos (cambios negativos o cero): {len(negative_improvements)} épocas")
        report_lines.append(f"\nMejora promedio: {np.mean(improvements):.6f}")
        report_lines.append(f"Mejora máxima: {np.max(improvements):.6f} (entre épocas {epoch_numbers[np.argmax(improvements)]} y {epoch_numbers[np.argmax(improvements)+1]})")
        report_lines.append(f"Mejora mínima: {np.min(improvements):.6f} (entre épocas {epoch_numbers[np.argmin(improvements)]} y {epoch_numbers[np.argmin(improvements)+1]})")
        report_lines.append(f"Desviación estándar de mejoras: {np.std(improvements):.6f}")
        if len(positive_improvements) > 0:
            report_lines.append(f"\nMejora promedio (solo épocas con mejora): {np.mean(positive_improvements):.6f}")
        if len(negative_improvements) > 0:
            report_lines.append(f"Empeoramiento promedio: {np.mean(negative_improvements):.6f}")
        report_lines.append("")
    
    # 4. Velocidad de Mejora del Loss
    if len(times) > 1 and len(epoch_losses) > 1:
        report_lines.append("4. VELOCIDAD DE MEJORA DEL LOSS")
        report_lines.append("-" * 80)
        time_diff = np.diff(times)
        loss_diff = np.diff(epoch_losses)
        rate = -loss_diff / time_diff  # Tasa de reducción (loss/segundo)
        
        report_lines.append(f"Tasa promedio de reducción: {np.mean(rate):.6f} loss/segundo")
        report_lines.append(f"Tasa máxima de reducción: {np.max(rate):.6f} loss/segundo")
        report_lines.append(f"Tasa mínima de reducción: {np.min(rate):.6f} loss/segundo")
        report_lines.append(f"Desviación estándar de tasa: {np.std(rate):.6f} loss/segundo")
        report_lines.append("")
    
    # 5. Estadísticas Detalladas por Componente
    if len(epoch_each_losses) > 0:
        report_lines.append("5. ESTADÍSTICAS DETALLADAS POR COMPONENTE")
        report_lines.append("-" * 80)
        
        for i in range(n_components):
            component_losses = epoch_each_losses[:, i]
            initial_val = component_losses[0]
            final_val = component_losses[-1]
            reduction_pct = ((initial_val - final_val) / initial_val * 100) if initial_val > 0 else 0
            
            # Encontrar épocas de min y max
            min_epoch = epoch_numbers[np.argmin(component_losses)]
            max_epoch = epoch_numbers[np.argmax(component_losses)]
            
            report_lines.append(f"\n5.{i+1} Componente: {component_names[i]}")
            report_lines.append(f"  {'-' * 76}")
            report_lines.append(f"  Inicial (Época {epoch_numbers[0]}): {initial_val:.6f}")
            report_lines.append(f"  Final (Época {epoch_numbers[-1]}): {final_val:.6f}")
            report_lines.append(f"  Reducción absoluta: {initial_val - final_val:.6f}")
            report_lines.append(f"  Reducción relativa: {reduction_pct:.2f}%")
            report_lines.append(f"  Mínimo: {np.min(component_losses):.6f} (Época {min_epoch})")
            report_lines.append(f"  Máximo: {np.max(component_losses):.6f} (Época {max_epoch})")
            report_lines.append(f"  Promedio: {np.mean(component_losses):.6f}")
            report_lines.append(f"  Mediana: {np.median(component_losses):.6f}")
            report_lines.append(f"  Desviación estándar: {np.std(component_losses):.6f}")
            report_lines.append(f"  Rango: {np.max(component_losses) - np.min(component_losses):.6f}")
            
            # Mejoras por componente
            if len(component_losses) > 1:
                comp_improvements = -np.diff(component_losses)
                report_lines.append(f"  Mejora promedio por época: {np.mean(comp_improvements):.6f}")
                report_lines.append(f"  Mejora máxima: {np.max(comp_improvements):.6f}")
                report_lines.append(f"  Mejora mínima: {np.min(comp_improvements):.6f}")
        
        report_lines.append("")
        
        # 6. Comparación Inicial vs Final por Componente
        report_lines.append("6. COMPARACIÓN INICIAL VS FINAL POR COMPONENTE")
        report_lines.append("-" * 80)
        initial_components = epoch_each_losses[0]
        final_components = epoch_each_losses[-1]
        
        report_lines.append(f"{'Componente':<15} {'Inicial':<12} {'Final':<12} {'Reducción':<12} {'% Reducción':<12}")
        report_lines.append("-" * 80)
        for i in range(n_components):
            reduction_abs = initial_components[i] - final_components[i]
            reduction_pct = ((initial_components[i] - final_components[i]) / initial_components[i] * 100) if initial_components[i] > 0 else 0
            report_lines.append(f"{component_names[i]:<15} {initial_components[i]:<12.6f} {final_components[i]:<12.6f} {reduction_abs:<12.6f} {reduction_pct:<12.2f}%")
        
        report_lines.append("")
        
        # 7. Ranking de Componentes
        report_lines.append("7. RANKING DE COMPONENTES")
        report_lines.append("-" * 80)
        
        # Por valor final
        final_values = epoch_each_losses[-1]
        sorted_indices_final = np.argsort(final_values)
        report_lines.append("\nPor valor final (menor a mayor):")
        for rank, idx in enumerate(sorted_indices_final, 1):
            report_lines.append(f"  {rank}. {component_names[idx]}: {final_values[idx]:.6f}")
        
        # Por reducción absoluta
        reductions_abs = initial_components - final_components
        sorted_indices_reduction = np.argsort(reductions_abs)[::-1]  # Mayor a menor
        report_lines.append("\nPor reducción absoluta (mayor a menor):")
        for rank, idx in enumerate(sorted_indices_reduction, 1):
            reduction_pct = ((initial_components[idx] - final_components[idx]) / initial_components[idx] * 100) if initial_components[idx] > 0 else 0
            report_lines.append(f"  {rank}. {component_names[idx]}: {reductions_abs[idx]:.6f} ({reduction_pct:.2f}%)")
        
        # Por reducción relativa
        reductions_pct = np.array([((initial_components[i] - final_components[i]) / initial_components[i] * 100) if initial_components[i] > 0 else 0 for i in range(n_components)])
        sorted_indices_pct = np.argsort(reductions_pct)[::-1]  # Mayor a menor
        report_lines.append("\nPor reducción relativa (mayor a menor):")
        for rank, idx in enumerate(sorted_indices_pct, 1):
            report_lines.append(f"  {rank}. {component_names[idx]}: {reductions_pct[idx]:.2f}%")
        
        # Por valor promedio durante entrenamiento
        component_means = np.mean(epoch_each_losses, axis=0)
        sorted_indices_mean = np.argsort(component_means)
        report_lines.append("\nPor valor promedio durante entrenamiento (menor a mayor):")
        for rank, idx in enumerate(sorted_indices_mean, 1):
            report_lines.append(f"  {rank}. {component_names[idx]}: {component_means[idx]:.6f}")
        
        report_lines.append("")
    
    # 8. Resumen Ejecutivo
    report_lines.append("8. RESUMEN EJECUTIVO")
    report_lines.append("-" * 80)
    report_lines.append(f"El entrenamiento completó {len(epoch_numbers)} épocas.")
    report_lines.append(f"El loss total se redujo de {initial_loss:.6f} a {final_loss:.6f}, una reducción del {reduction_pct:.2f}%.")
    
    if len(epoch_each_losses) > 0:
        reductions_pct = np.array([((epoch_each_losses[0, i] - epoch_each_losses[-1, i]) / epoch_each_losses[0, i] * 100) if epoch_each_losses[0, i] > 0 else 0 for i in range(n_components)])
        best_reduction_idx = np.argmax(reductions_pct)
        worst_reduction_idx = np.argmin(reductions_pct)
        report_lines.append(f"El componente con mayor mejora relativa fue {component_names[best_reduction_idx]} ({reductions_pct[best_reduction_idx]:.2f}%).")
        report_lines.append(f"El componente con menor mejora relativa fue {component_names[worst_reduction_idx]} ({reductions_pct[worst_reduction_idx]:.2f}%).")
    
    if len(times) > 1:
        total_time = times[-1] - times[0]
        report_lines.append(f"El entrenamiento tomó {total_time/3600:.2f} horas ({total_time:.2f} segundos).")
    
    report_lines.append("")
    report_lines.append("="*80)
    
    # Guardar reporte en archivo
    report_path = output_dir / 'training_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"[OK] Reporte completo guardado en: {report_path}")

def main():
    # Ruta al archivo de logs
    log_file = Path('results/logs_training.txt')
    
    if not log_file.exists():
        print(f"Error: No se encontró el archivo {log_file}")
        return
    
    print(f"Leyendo logs de entrenamiento desde: {log_file}")
    data = parse_training_logs(log_file)
    
    if len(data['epoch_losses']) == 0:
        print("Error: No se encontraron datos de epoch loss en el archivo")
        return
    
    print(f"[OK] Datos parseados: {len(data['epoch_losses'])} epocas encontradas")
    
    print("\nGenerando visualizaciones...")
    create_visualizations(data, output_dir='results')
    
    print("\n[OK] Analisis completado!")

if __name__ == '__main__':
    main()

