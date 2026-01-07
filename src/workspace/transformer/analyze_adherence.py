import os
import json
import argparse
import numpy as np
import re


def load_meta(meta_path):
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_tokens(npy_path):
    return np.load(npy_path)


def extract_numeric_value(label):
    """
    Extract numeric value from token labels like 'Note_Pitch_60', 'Note_Duration_1920', etc.
    Returns None if not a numeric token or if extraction fails.
    """
    if not isinstance(label, str):
        return None
    
    # Match patterns like Note_Pitch_60, Note_Duration_1920, Note_Velocity_100, Tempo_120
    match = re.search(r'_(\d+)$', label)
    if match:
        return int(match.group(1))
    return None


def compute_adherence(tokens, conditions, token_order, word2event=None):
    """
    Compute adherence of generated tokens to conditioning.
    
    Args:
        tokens: np.array of shape [T, F] (T timesteps, F features)
        conditions: dict {token_name: target_index}
        token_order: list of token names in order (from vocab_order in metadata)
        word2event: optional dict for label lookup to compute distance metrics
    
    Returns:
        dict {token_name: {'target_index', 'matches', 'total', 'adherence_ratio', 
                           'mean_distance', 'median_distance', 'target_value', 'is_numeric'}}
    """
    results = {}
    token_name_to_idx = {name: i for i, name in enumerate(token_order)}

    for name, target_idx in (conditions or {}).items():
        if name not in token_name_to_idx:
            continue
        pos = token_name_to_idx[name]
        col = tokens[:, pos]
        total = int(col.shape[0])
        matches = int(np.sum(col == int(target_idx)))
        adherence = float(matches) / float(total) if total > 0 else 0.0
        
        result = {
            'target_index': int(target_idx),
            'matches': matches,
            'total': total,
            'adherence_ratio': adherence,
            'is_numeric': False
        }
        
        # Compute distance metrics for numerical tokens
        if word2event and name in word2event:
            target_label = word2event[name].get(str(target_idx))
            target_value = extract_numeric_value(target_label)
            
            if target_value is not None:
                result['is_numeric'] = True
                result['target_value'] = target_value
                
                # Extract numeric values for all generated tokens
                generated_values = []
                for idx in col:
                    label = word2event[name].get(str(int(idx)))
                    val = extract_numeric_value(label)
                    if val is not None:
                        generated_values.append(val)
                
                if generated_values:
                    distances = [abs(v - target_value) for v in generated_values]
                    result['mean_distance'] = float(np.mean(distances))
                    result['median_distance'] = float(np.median(distances))
                    result['min_distance'] = float(np.min(distances))
                    result['max_distance'] = float(np.max(distances))
                    result['std_distance'] = float(np.std(distances))
        
        results[name] = result
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='Directory with generated *.meta.json and *.npy files')
    parser.add_argument('--out', default=None, help='Optional path to save aggregate report JSON')
    parser.add_argument('--word2event', default=None, help='Optional path to word2event.json for distance metrics')
    args = parser.parse_args()

    # Load word2event if provided
    word2event = None
    if args.word2event and os.path.exists(args.word2event):
        with open(args.word2event, 'r', encoding='utf-8') as f:
            word2event = json.load(f)

    metas = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith('.meta.json')]
    aggregate = {
        'files': [],
        'summary': {},
        'summary_distance': {}
    }

    per_token_accumulator = {}
    per_token_counts = {}
    per_token_distance = {}

    for meta_path in metas:
        try:
            meta = load_meta(meta_path)
            tokens = load_tokens(meta['paths']['npy'])
            conditions = meta.get('conditions_indices') or {}
            token_order = meta.get('vocab_order') or []
            adherence = compute_adherence(tokens, conditions, token_order, word2event)

            aggregate['files'].append({
                'meta': meta_path,
                'npy': meta['paths']['npy'],
                'midi': meta['paths']['midi'],
                'inference_mode': meta.get('inference_mode'),
                'emotion_tag': meta.get('emotion_tag'),
                'key_tag': meta.get('key_tag'),
                'adherence': adherence
            })

            # accumulate
            for name, stats in adherence.items():
                per_token_accumulator.setdefault(name, 0.0)
                per_token_counts.setdefault(name, 0)
                per_token_accumulator[name] += stats['adherence_ratio']
                per_token_counts[name] += 1
                
                # accumulate distance metrics if available
                if stats.get('is_numeric') and 'mean_distance' in stats:
                    per_token_distance.setdefault(name, [])
                    per_token_distance[name].append(stats['mean_distance'])
        except Exception as e:
            # skip malformed
            print(f"[WARN] Skipping {meta_path}: {e}")
            continue

    # compute averages
    for name, total_ratio in per_token_accumulator.items():
        count = per_token_counts.get(name, 1)
        aggregate['summary'][name] = {
            'adherence_ratio': total_ratio / float(count)
        }
        
        if name in per_token_distance and per_token_distance[name]:
            aggregate['summary'][name]['mean_distance'] = float(np.mean(per_token_distance[name]))
            aggregate['summary'][name]['median_distance'] = float(np.median(per_token_distance[name]))

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(aggregate, f, indent=2)
    else:
        print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()


