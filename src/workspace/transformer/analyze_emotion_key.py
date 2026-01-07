"""
Analyze emotion and key adherence for generated MIDI files.
"""
import os
import subprocess
import re
import json
import numpy as np
import miditoolkit
import collections
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Krumhansl–Schmuckler key detection helpers (from midi2corpus.py)
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def compute_pitch_class_histogram(midi_path: str) -> np.ndarray:
    """Compute pitch class histogram from MIDI file."""
    midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)
    
    hist = np.zeros(12, dtype=float)
    for instr in midi_obj.instruments:
        for note in instr.notes:
            pc = note.pitch % 12
            dur = max(1, note.end - note.start)
            hist[pc] += dur
    
    if hist.sum() > 0:
        hist /= hist.sum()
    
    return hist


def detect_key_from_midi(midi_path: str, return_score: bool = False) -> str:
    """Detect key from MIDI file using Krumhansl–Schmuckler algorithm."""
    hist = compute_pitch_class_histogram(midi_path)
    
    best_score = -1e9
    best = ('C', 'maj')
    
    for i, tonic in enumerate(PITCH_CLASSES):
        maj_prof = np.roll(KK_MAJOR, i)
        min_prof = np.roll(KK_MINOR, i)
        
        denom_maj = (np.linalg.norm(hist) * np.linalg.norm(maj_prof) + 1e-8)
        denom_min = (np.linalg.norm(hist) * np.linalg.norm(min_prof) + 1e-8)
        
        smaj = float(np.dot(hist, maj_prof) / denom_maj)
        smin = float(np.dot(hist, min_prof) / denom_min)
        
        if smaj > best_score:
            best_score = smaj
            best = (tonic, 'maj')
        if smin > best_score:
            best_score = smin
            best = (tonic, 'min')
    
    if return_score:
        return f"{best[0]}:{best[1]}", best_score
    return f"{best[0]}:{best[1]}"


def compute_key_correlation(midi_path: str, expected_key: str) -> float:
    """
    Compute correlation score between MIDI histogram and expected key profile.
    Higher score = closer to expected key (max ~1.0).
    
    Args:
        midi_path: Path to MIDI file
        expected_key: Expected key in format "C:maj" or "A:min"
    
    Returns:
        Correlation score (higher = better match)
    """
    hist = compute_pitch_class_histogram(midi_path)
    
    # Parse expected key
    try:
        tonic, mode = expected_key.split(':')
        tonic_idx = PITCH_CLASSES.index(tonic)
    except (ValueError, IndexError):
        return 0.0
    
    # Get the appropriate key profile
    if mode == 'maj':
        profile = np.roll(KK_MAJOR, tonic_idx)
    elif mode == 'min':
        profile = np.roll(KK_MINOR, tonic_idx)
    else:
        return 0.0
    
    # Compute correlation score
    denom = (np.linalg.norm(hist) * np.linalg.norm(profile) + 1e-8)
    score = float(np.dot(hist, profile) / denom)
    
    return score


def run_emotion_classifier(midi_path: str, wsl_emopia_path: str = '/mnt/s/UVG/LOCKIN/EMOPIA_cls') -> Dict:
    """
    Run emotion classifier on MIDI file through WSL.
    
    Returns:
        dict with 'emotion' (Q1-Q4), 'values' array, and 'success' boolean
    """
    # Convert Windows path to WSL path
    if midi_path.startswith('S:\\') or midi_path.startswith('S:/'):
        wsl_midi_path = midi_path.replace('S:\\', '/mnt/s/').replace('S:/', '/mnt/s/').replace('\\', '/')
    else:
        # Assume it's already a WSL path or relative
        wsl_midi_path = midi_path.replace('\\', '/')
    
    # Build WSL command
    wsl_cmd = [
        'wsl', '-e', 'bash', '-c',
        f'cd {wsl_emopia_path} && source my_venv_38/bin/activate && python inference.py --types midi_like --task ar_va --file_path "{wsl_midi_path}" --cuda 0'
    ]
    
    try:
        result = subprocess.run(wsl_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return {'success': False, 'error': result.stderr}
        
        # Parse output
        output = result.stdout
        
        # Extract emotion (Q1-Q4)
        emotion_match = re.search(r'is emotion (Q\d)', output)
        if not emotion_match:
            return {'success': False, 'error': 'Could not parse emotion from output'}
        
        emotion = emotion_match.group(1)
        
        # Extract values array
        values_match = re.search(r'Inference values:\s*\[([\s\-\d.]+)\]', output)
        if not values_match:
            return {'success': False, 'error': 'Could not parse values from output'}
        
        values_str = values_match.group(1)
        values = np.array([float(x) for x in values_str.split()])
        
        return {
            'success': True,
            'emotion': emotion,
            'emotion_idx': int(emotion[1]),  # Q1->1, Q2->2, etc.
            'values': values.tolist(),
            'raw_output': output
        }
        
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Timeout running emotion classifier'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def analyze_emotion_adherence(midi_files: List[str], expected_emotion: int) -> Dict:
    """
    Analyze emotion adherence for a list of MIDI files.
    
    Args:
        midi_files: List of MIDI file paths
        expected_emotion: Expected emotion tag (1-4)
    
    Returns:
        Dictionary with adherence statistics
    """
    results = []
    
    for midi_path in midi_files:
        if not os.path.exists(midi_path):
            continue
        
        result = run_emotion_classifier(midi_path)
        if result['success']:
            results.append({
                'file': midi_path,
                'detected_emotion': result['emotion_idx'],
                'expected_emotion': expected_emotion,
                'match': result['emotion_idx'] == expected_emotion,
                'values': result['values']
            })
    
    if not results:
        return {'error': 'No successful emotion classifications'}
    
    # Calculate statistics
    matches = sum(1 for r in results if r['match'])
    total = len(results)
    
    return {
        'adherence_ratio': matches / total if total > 0 else 0.0,
        'matches': matches,
        'total': total,
        'details': results
    }


def analyze_key_adherence(midi_files: List[str], expected_key: str) -> Dict:
    """
    Analyze key adherence for a list of MIDI files.
    
    Args:
        midi_files: List of MIDI file paths
        expected_key: Expected key (e.g., 'C:maj', 'A:min')
    
    Returns:
        Dictionary with adherence statistics including correlation scores
    """
    results = []
    correlation_scores = []
    
    for midi_path in midi_files:
        if not os.path.exists(midi_path):
            continue
        
        try:
            detected_key = detect_key_from_midi(midi_path)
            correlation = compute_key_correlation(midi_path, expected_key)
            
            results.append({
                'file': midi_path,
                'detected_key': detected_key,
                'expected_key': expected_key,
                'match': detected_key == expected_key,
                'correlation': correlation
            })
            correlation_scores.append(correlation)
        except Exception as e:
            print(f"Error detecting key for {midi_path}: {e}")
            continue
    
    if not results:
        return {'error': 'No successful key detections'}
    
    # Calculate statistics
    matches = sum(1 for r in results if r['match'])
    total = len(results)
    
    # Also track what keys were detected
    key_distribution = collections.defaultdict(int)
    for r in results:
        key_distribution[r['detected_key']] += 1
    
    # Compute average correlation (distance metric)
    # Higher correlation = closer to expected key
    # We report as "mean_correlation" (higher is better, range ~0-1)
    mean_correlation = np.mean(correlation_scores) if correlation_scores else 0.0
    
    return {
        'adherence_ratio': matches / total if total > 0 else 0.0,
        'matches': matches,
        'total': total,
        'mean_correlation': float(mean_correlation),
        'key_distribution': dict(key_distribution),
        'details': results
    }


def compute_emotion_key_adherence(run_dir: str, expected_emotion: int, expected_key: Optional[str] = None) -> Dict:
    """
    Compute both emotion and key adherence for a generation run.
    
    Args:
        run_dir: Directory containing generated MIDI files
        expected_emotion: Expected emotion tag (1-4)
        expected_key: Expected key (optional, only for scratch model)
    
    Returns:
        Dictionary with both emotion and key adherence results
    """
    # Find all MIDI files in the directory
    midi_files = []
    for f in os.listdir(run_dir):
        if f.endswith('.mid') or f.endswith('.midi'):
            midi_files.append(os.path.join(run_dir, f))
    
    if not midi_files:
        return {'error': 'No MIDI files found in directory'}
    
    results = {}
    
    # Emotion adherence
    results['emotion'] = analyze_emotion_adherence(midi_files, expected_emotion)
    
    # Key adherence (only if expected_key is provided)
    if expected_key:
        results['key'] = analyze_key_adherence(midi_files, expected_key)
    
    results['num_files'] = len(midi_files)
    
    return results


if __name__ == '__main__':
    # Test with a sample run
    import sys
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
        emotion = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        key = sys.argv[3] if len(sys.argv) > 3 else None
        
        results = compute_emotion_key_adherence(run_dir, emotion, key)
        print(json.dumps(results, indent=2))
