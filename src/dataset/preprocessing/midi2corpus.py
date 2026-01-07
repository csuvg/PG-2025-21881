import os
import re
import glob
import pickle
import numpy as np
import miditoolkit
import collections

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================================================== #  
#  Configuration                                     #
# ================================================== #  
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4
INSTR_NAME_MAP = {'piano': 0}
MIN_BPM = 40
MIN_VELOCITY = 40
NOTE_SORTING = 1 #  0: ascending / 1: descending

DEFAULT_VELOCITY_BINS = np.linspace(0,  128, 64+1, dtype=np.int_)
DEFAULT_BPM_BINS      = np.linspace(32, 224, 64+1, dtype=np.int_)
DEFAULT_SHIFT_BINS    = np.linspace(-60, 60, 60+1, dtype=np.int_)
DEFAULT_DURATION_BINS = np.arange(
        BEAT_RESOL/8, BEAT_RESOL*8+1, BEAT_RESOL/8)

# ================================================== #  


# -------------------------------------------------- #
# Krumhansl–Schmuckler key detection helpers
# -------------------------------------------------- #
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def compute_pitch_class_hist(instr_grid):
    hist = np.zeros(12, dtype=float)
    for _, note_grid in instr_grid.items():
        for _, notes in note_grid.items():
            for note in notes:
                pc = note.pitch % 12
                dur = max(1, note.end - note.start)
                hist[pc] += dur
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def detect_key_krumhansl(instr_grid):
    hist = compute_pitch_class_hist(instr_grid)
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
    return f"{best[0]}:{best[1]}"


def traverse_dir(
        root_dir,
        extension=('mid', 'MID', 'midi'),
        amount=None,
        str_=None,
        is_pure=False,
        verbose=False,
        is_sort=False,
        is_ext=True):
    if verbose:
        print('[*] Scanning...')
    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                if (amount is not None) and (cnt == amount):
                    break
                if str_ is not None:
                    if str_ not in file:
                        continue
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                if verbose:
                    print(pure_path)
                file_list.append(pure_path)
                cnt += 1
    if verbose:
        print('Total: %d files' % len(file_list))
        print('Done!!!')
    if is_sort:
        file_list.sort()
    return file_list


def proc_one(path_midi, path_outfile):
    # --- load --- #
    midi_obj = miditoolkit.midi.parser.MidiFile(path_midi)

    # collect emotion tag (robust to OS separators and varied filenames)
    base_name = os.path.basename(path_midi)
    match = re.search(r"Q[1-4]", base_name.upper())
    emo_tag = match.group(0) if match else 'Q1'
    
    # load notes
    instr_notes = collections.defaultdict(list)
    for instr in midi_obj.instruments:
        # skip 
        if instr.name not in INSTR_NAME_MAP.keys():
            continue

        # process
        instr_idx = INSTR_NAME_MAP[instr.name]
        for note in instr.notes:
            note.instr_idx=instr_idx
            instr_notes[instr_idx].append(note)
        if NOTE_SORTING == 0:
            instr_notes[instr_idx].sort(
                key=lambda x: (x.start, x.pitch))
        elif NOTE_SORTING == 1:
            instr_notes[instr_idx].sort(
                key=lambda x: (x.start, -x.pitch))
        else:
            raise ValueError(' [x] Unknown type of sorting.')

    # load chords
    chords = []
    for marker in midi_obj.markers:
        if marker.text.split('_')[0] != 'global' and \
        'Boundary' not in marker.text.split('_')[0]:
            chords.append(marker)
    chords.sort(key=lambda x: x.time)

    # load tempos
    tempos = midi_obj.tempo_changes
    tempos.sort(key=lambda x: x.time)

    # load labels
    labels = []
    for marker in midi_obj.markers:
        if 'Boundary' in marker.text.split('_')[0]:
            labels.append(marker)
    labels.sort(key=lambda x: x.time)

    # load global bpm
    gobal_bpm = 120
    for marker in midi_obj.markers:
        if marker.text.split('_')[0] == 'global' and \
            marker.text.split('_')[1] == 'bpm':
            gobal_bpm = int(marker.text.split('_')[2])
        
    # --- process items to grid --- #
    # compute empty bar offset at head
    first_note_time = min([instr_notes[k][0].start for k in instr_notes.keys()])
    last_note_time = max([instr_notes[k][-1].start for k in instr_notes.keys()])

    quant_time_first = int(np.round(first_note_time  / TICK_RESOL) * TICK_RESOL)
    offset = quant_time_first // BAR_RESOL # empty bar
    last_bar = int(np.ceil(last_note_time / BAR_RESOL)) - offset
    print(' > offset:', offset)
    print(' > last_bar:', last_bar)

    # process notes
    intsr_gird = dict()
    for key in instr_notes.keys():
        notes = instr_notes[key]
        note_grid = collections.defaultdict(list)
        for note in notes:
            note.start = note.start - offset * BAR_RESOL
            note.end = note.end - offset * BAR_RESOL

            # quantize start
            quant_time = int(np.round(note.start / TICK_RESOL) * TICK_RESOL)

            # velocity
            note.velocity = DEFAULT_VELOCITY_BINS[
                np.argmin(abs(DEFAULT_VELOCITY_BINS-note.velocity))]
            note.velocity = max(MIN_VELOCITY, note.velocity)

            # shift of start
            note.shift = note.start - quant_time 
            note.shift = DEFAULT_SHIFT_BINS[np.argmin(abs(DEFAULT_SHIFT_BINS-note.shift))]

            # duration
            note_duration = note.end - note.start
            if note_duration > BAR_RESOL:
                note_duration = BAR_RESOL
            ntick_duration = int(np.round(note_duration / TICK_RESOL) * TICK_RESOL)
            # miditoolkit's Note.duration is a read-only property (end - start).
            # Quantize by updating end to preserve duration semantics.
            note.end = note.start + ntick_duration

            # append
            note_grid[quant_time].append(note)
        
        # set to track
        intsr_gird[key] = note_grid.copy()

    # process chords
    chord_grid = collections.defaultdict(list)
    for chord in chords:
        # quantize
        chord.time = chord.time - offset * BAR_RESOL
        chord.time  = 0 if chord.time < 0 else chord.time 
        quant_time = int(np.round(chord.time / TICK_RESOL) * TICK_RESOL)

        # append
        chord_grid[quant_time].append(chord)

    # process tempo
    tempo_grid = collections.defaultdict(list)
    for tempo in tempos:
        # quantize
        tempo.time = tempo.time - offset * BAR_RESOL
        tempo.time = 0 if tempo.time < 0 else tempo.time
        quant_time = int(np.round(tempo.time / TICK_RESOL) * TICK_RESOL)
        tempo.tempo = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS-tempo.tempo))]

        # append
        tempo_grid[quant_time].append(tempo)

    # process boundary
    label_grid = collections.defaultdict(list)
    for label in labels:
        # quantize
        label.time = label.time - offset * BAR_RESOL
        label.time = 0 if label.time < 0 else label.time
        quant_time = int(np.round(label.time / TICK_RESOL) * TICK_RESOL)

        # append
        label_grid[quant_time] = [label]
        
    # process global bpm
    gobal_bpm = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS-gobal_bpm))]

    # detect global key using Krumhansl–Schmuckler over all instruments
    key_text = detect_key_krumhansl(intsr_gird)

    # collect
    song_data = {
        'notes': intsr_gird,
        'chords': chord_grid,
        'tempos': tempo_grid,
        'labels': label_grid,
        'metadata': {
            'global_bpm': gobal_bpm,
            'last_bar': last_bar,
            'emotion': emo_tag,
            'key': key_text
        }
    }

    # save
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[:-len(fn)], exist_ok=True)
    pickle.dump(song_data, open(path_outfile, 'wb'))
    
    return song_data

if __name__ == '__main__':
     # paths
    import sys
    # Usage: python midi2corpus.py <input_dir> <output_dir>
    # Example: python midi2corpus.py ./midi_analyzed/fixed ./corpus/fixed
    path_indir = sys.argv[1]
    path_outdir = sys.argv[2]
    os.makedirs(path_outdir, exist_ok=True)

    # list files
    midifiles = traverse_dir(
        path_indir,
        is_pure=True,
        is_sort=True)
    n_files = len(midifiles)
    print('num fiels:', n_files)

    # run all
    for fidx in range(n_files):
        path_midi = midifiles[fidx]
        print('{}/{}'.format(fidx, n_files))

        # paths
        path_infile = os.path.join(path_indir, path_midi)
        path_outfile = os.path.join(path_outdir, path_midi+'.pkl')

        # proc
        _ = proc_one(path_infile, path_outfile)