import sys
import os
import ipdb
from tqdm import tqdm
import math
import time
import glob
import datetime
import random
import pickle
import json
import numpy as np
from collections import OrderedDict
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader


import saver
from models import TransformerModel, network_paras
from utils import write_midi, get_random_string


################################################################################
# config
################################################################################

parser = ArgumentParser()
parser.add_argument("--mode",default="train",type=str,choices=[
    "train",
    "inference",
    "inference-normal",
    "inference-deterministic",
    "inference-primed",
    "inference-forced"
])
parser.add_argument("--task_type",default="4-cls",type=str,choices=['4-cls', 'Arousal', 'Valence', 'ignore'])
parser.add_argument("--gid", default= 0, type=int)
parser.add_argument("--data_parallel", default= 0, type=int)

parser.add_argument("--exp_name", default='output' , type=str)
parser.add_argument("--load_ckt", default="none", type=str)   #pre-train model
parser.add_argument("--load_ckt_loss", default="25", type=str)     #pre-train model
parser.add_argument("--path_train_data", default='train', type=str)  
parser.add_argument("--data_root", default='../dataset/co-representation/', type=str)
parser.add_argument("--load_dict", default="dictionary.pkl", type=str)
parser.add_argument("--init_lr", default= 0.00001, type=float)
# inference config

parser.add_argument("--num_songs", default=5, type=int)
parser.add_argument("--emo_tag", default=1, type=int)
parser.add_argument("--out_dir", default='none', type=str)
# inference advanced options
parser.add_argument("--conditions", default=None, type=str,
                    help="JSON string mapping token names to labels or indices, e.g. '{\"duration\": \"Note_Duration_1920\"}'")
parser.add_argument("--force_tokens", default=None, type=str,
                    help="JSON string mapping token names or positions to fixed indices during generation")
parser.add_argument("--temperature", default=None, type=float,
                    help="Global temperature for sampling (overrides defaults)")
parser.add_argument("--top_p", default=None, type=float,
                    help="Global nucleus sampling p (overrides defaults)")
parser.add_argument("--key_tag", default=None, type=str,
                    help="Key tag string to condition when using 9-token models (e.g., 'A:maj', 'C#:min')")
parser.add_argument("--save_tokens_json", default=0, type=int,
                    help="If 1, also save tokens as JSON for readability")
parser.add_argument("--generation_timeout", default=60, type=int,
                    help="Maximum time in seconds for generation before auto-stopping (default: 60)")

# exp path
parser.add_argument("--exp_path", default='exp', type=str)

args = parser.parse_args()

print('=== args ===')
for arg in args.__dict__:
    print(arg, args.__dict__[arg])
print('=== args ===')
# time.sleep(10)    #sleep to check again if args are right


MODE = args.mode
task_type = args.task_type


###--- data ---###
path_data_root = args.data_root

path_train_data = os.path.join(path_data_root, args.path_train_data + '_data.npz')
path_dictionary =  os.path.join(path_data_root, args.load_dict)
path_train_idx = os.path.join(path_data_root, args.path_train_data + '_fn2idx_map.json')
path_train_data_cls_idx = os.path.join(path_data_root, args.path_train_data + '_idx.npz')

print('path_train_data:', path_train_data)
print('path_dictionary:', path_dictionary)
print('path_train_idx:', path_train_idx)
print('path_train_data_cls_idx:', path_train_data_cls_idx)

assert os.path.exists(path_train_data)
assert os.path.exists(path_dictionary)
assert os.path.exists(path_train_idx)

# if the dataset has the emotion label, get the cls_idx for the dataloader
if args.path_train_data == 'emopia':    
    assert os.path.exists(path_train_data_cls_idx)

###--- training config ---###
 
if MODE == 'train':
    path_exp = args.exp_path + '/' + args.exp_name

if args.data_parallel > 0:
    batch_size = 8
else:
    batch_size = 4      #4

gid = args.gid
init_lr = args.init_lr   #0.0001


###--- fine-tuning & inference config ---###
if args.load_ckt == 'none':
    info_load_model = None
    print('NO pre-trained model used')

else:
    info_load_model = (
        # path to ckpt for loading
        args.exp_path + '/' + args.load_ckt,
        # loss
        args.load_ckt_loss                               
        )


if args.out_dir == 'none':
    path_gendir = os.path.join(args.exp_path + '/' + args.load_ckt, 'gen_midis', 'loss_'+ args.load_ckt_loss)
else:
    path_gendir = args.out_dir

num_songs = args.num_songs
emotion_tag = args.emo_tag


################################################################################
# File IO
################################################################################

if args.data_parallel == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)


##########################################################################################################################
# Script
##########################################################################################################################


class PEmoDataset(Dataset):
    def __init__(self,
                 
                 task_type,
                 emotion_col):

        self.train_data = np.load(path_train_data)
        self.train_x = self.train_data['x']
        self.train_y = self.train_data['y']
        self.train_mask = self.train_data['mask']
        
        self.emotion_col = emotion_col

        if task_type != 'ignore':
            # Load class indices if available; otherwise compute from emotion column
            if os.path.exists(path_train_data_cls_idx):
                self.cls_idx = np.load(path_train_data_cls_idx)
                self.cls_1_idx = self.cls_idx['cls_1_idx']
                self.cls_2_idx = self.cls_idx['cls_2_idx']
                self.cls_3_idx = self.cls_idx['cls_3_idx']
                self.cls_4_idx = self.cls_idx['cls_4_idx']
            else:
                # Compute indices from the first token's emotion value per sequence
                if emotion_col is None or emotion_col < 0:
                    n = self.train_x.shape[0]
                    self.cls_1_idx = np.arange(n, dtype=int)
                    self.cls_2_idx = np.array([], dtype=int)
                    self.cls_3_idx = np.array([], dtype=int)
                    self.cls_4_idx = np.array([], dtype=int)
                else:
                    labels = self.train_x[:, 0, emotion_col]
                    self.cls_1_idx = np.where(labels == 1)[0]
                    self.cls_2_idx = np.where(labels == 2)[0]
                    self.cls_3_idx = np.where(labels == 3)[0]
                    self.cls_4_idx = np.where(labels == 4)[0]
                # Save for reuse
                try:
                    np.savez(
                        path_train_data_cls_idx,
                        cls_1_idx=self.cls_1_idx,
                        cls_2_idx=self.cls_2_idx,
                        cls_3_idx=self.cls_3_idx,
                        cls_4_idx=self.cls_4_idx
                    )
                except Exception:
                    pass
        
            if task_type == 'Arousal':
                print('preparing data for training "Arousal"')
                self.label_transfer('Arousal')

            elif task_type == 'Valence':
                print('preparing data for training "Valence"')
                self.label_transfer('Valence')


        self.train_x = torch.from_numpy(self.train_x).long()
        self.train_y = torch.from_numpy(self.train_y).long()
        self.train_mask = torch.from_numpy(self.train_mask).float()


        self.seq_len = self.train_x.shape[1]
        self.dim = self.train_x.shape[2]
        
        print('train_x: ', self.train_x.shape)

    def label_transfer(self, TYPE):
        col = self.emotion_col if self.emotion_col is not None else -1
        if TYPE == 'Arousal':
            for i in range(self.train_x.shape[0]):
                if self.train_x[i][0][col] in [1,2]:
                    self.train_x[i][0][col] = 1
                elif self.train_x[i][0][col] in [3,4]:
                    self.train_x[i][0][col] = 2
        
        elif TYPE == 'Valence':
            for i in range(self.train_x.shape[0]):
                if self.train_x[i][0][col] in [1,4]:
                    self.train_x[i][0][col] = 1
                elif self.train_x[i][0][col] in [2,3]:
                    self.train_x[i][0][col] = 2   

        
        
    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index], self.train_mask[index]
    

    def __len__(self):
        return len(self.train_x)


def prep_dataloader(task_type, batch_size, emotion_col, n_jobs=0):
    
    dataset = PEmoDataset(task_type, emotion_col) 
    
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=False, drop_last=False,
        num_workers=n_jobs, pin_memory=True)                          
    return dataloader




def train():

    myseed = 42069
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():    
        torch.cuda.manual_seed_all(myseed)


    # hyper params
    n_epoch = 4000
    max_grad_norm = 3

    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary
    
    # derive emotion column dynamically from dictionary
    class_order = list(event2word.keys())
    emotion_col = class_order.index('emotion') if 'emotion' in class_order else -1

    train_loader = prep_dataloader(args.task_type, batch_size, emotion_col)

    # create saver
    saver_agent = saver.Saver(path_exp)

    # config
    n_class = []  # number of classes of each token. [56, 127, 18, 4, 85, 18, 41, 5]  with key: [... , 25]
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))
    
    

    n_token = len(n_class)
    # log
    print('num of classes:', n_class)
    
    # init
    

    if args.data_parallel > 0 and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = TransformerModel(n_class, data_parallel=True)
        net = nn.DataParallel(net)

    else:
        net = TransformerModel(n_class)

    net.cuda()
    net.train()
    n_parameters = network_paras(net)
    print('n_parameters: {:,}'.format(n_parameters))
    saver_agent.add_summary_msg(
        ' > params amount: {:,d}'.format(n_parameters))

    
    # load model
    if info_load_model:
        path_ckpt = info_load_model[0] # path to ckpt dir
        loss = info_load_model[1] # loss
        name = 'loss_' + str(loss)
        path_saved_ckpt = os.path.join(path_ckpt, name + '_params.pt')
        print('[*] load model from:',  path_saved_ckpt)
        
        try:
            net.load_state_dict(torch.load(path_saved_ckpt))
        except:
            # print('WARNING!!!!! Not the whole pre-train model is loaded, only load partial')
            # print('WARNING!!!!! Not the whole pre-train model is loaded, only load partial')
            # print('WARNING!!!!! Not the whole pre-train model is loaded, only load partial')
            # net.load_state_dict(torch.load(path_saved_ckpt), strict=False)
        
            state_dict = torch.load(path_saved_ckpt)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] 
                new_state_dict[name] = v
            
            net.load_state_dict(new_state_dict)


    # optimizers
    optimizer = optim.Adam(net.parameters(), lr=init_lr)


    # run
    start_time = time.time()
    for epoch in range(n_epoch):
        acc_loss = 0
        acc_losses = np.zeros(n_token)


        num_batch = len(train_loader)
        print('    num_batch:', num_batch)

        for bidx, (batch_x, batch_y, batch_mask)  in enumerate(train_loader): # num_batch 
            saver_agent.global_step_increment()

            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_mask = batch_mask.cuda()
            
            
            losses = net(batch_x, batch_y, batch_mask)

            if args.data_parallel > 0:
                loss = 0
                calculated_loss = []
                for i in range(n_token):
                    
                    loss += ((losses[i][0][0] + losses[i][0][1]) / (losses[i][1][0] + losses[i][1][1]))
                    calculated_loss.append((losses[i][0][0] + losses[i][0][1]) / (losses[i][1][0] + losses[i][1][1]))
                loss = loss / n_token
                
                
            else:
                loss = sum(losses) / n_token


            # Update
            net.zero_grad()
            loss.backward()
                
                
            if max_grad_norm is not None:
                clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            if args.data_parallel > 0:
                # Formato dinámico basado en n_token
                loss_format_str = ', '.join(['{:04f}'] * n_token)
                sys.stdout.write('{}/{} | Loss: {:06f} | {}\r'.format(
                        bidx, num_batch, loss, loss_format_str.format(*calculated_loss[:n_token])))
                sys.stdout.flush()


                # acc
                acc_losses += np.array([l.item() for l in calculated_loss])



            else:
                # Formato dinámico basado en n_token
                loss_format_str = ', '.join(['{:04f}'] * n_token)
                sys.stdout.write('{}/{} | Loss: {:06f} | {}\r'.format(
                        bidx, num_batch, loss, loss_format_str.format(*[l.item() for l in losses[:n_token]])))
                sys.stdout.flush()


                # acc
                acc_losses += np.array([l.item() for l in losses])



            acc_loss += loss.item()

            # log
            saver_agent.add_summary('batch loss', loss.item())

        
        # epoch loss
        runtime = time.time() - start_time
        epoch_loss = acc_loss / num_batch
        acc_losses = acc_losses / num_batch
        print('------------------------------------')
        print('epoch: {}/{} | Loss: {} | time: {}'.format(
            epoch, n_epoch, epoch_loss, str(datetime.timedelta(seconds=runtime))))
        
        # Formato dinámico basado en n_token (puede ser 8 o 9 componentes)
        loss_format_str = ', '.join(['{:04f}'] * n_token)
        each_loss_str = loss_format_str.format(*acc_losses[:n_token])
        
        # Imprimir con \r para consola (sobreescribir línea)
        print('    >', each_loss_str + '\r', end='', flush=True)
        print()  # Nueva línea después

        # Guardar sin \r en el archivo (solo el string limpio con todos los componentes)
        saver_agent.add_summary('epoch loss', epoch_loss)
        saver_agent.add_summary('epoch each loss', each_loss_str)

        # save model, with policy
        loss = epoch_loss
        if 0.4 < loss <= 0.8:
            fn = int(loss * 10) * 10
            saver_agent.save_model(net, name='loss_' + str(fn))
        elif 0.08 < loss <= 0.40:
            fn = int(loss * 100)
            saver_agent.save_model(net, name='loss_' + str(fn))
        elif loss <= 0.08:
            print('Finished')
            return  
        else:
            saver_agent.save_model(net, name='loss_high')


def generate():

    # path
    path_ckpt = info_load_model[0] # path to ckpt dir
    loss = info_load_model[1] # loss
    name = 'loss_' + str(loss)
    path_saved_ckpt = os.path.join(path_ckpt, name + '_params.pt')

    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary

    # Save event2word and word2event as JSON files for inspection
    # with open(os.path.join('event2word.json'), 'w') as f:
    #     json.dump(event2word, f, indent=4)
    # with open(os.path.join('word2event.json'), 'w') as f:
    #     json.dump(word2event, f, indent=4)

    # outdir
    os.makedirs(path_gendir, exist_ok=True)

    # helpers for conditioning
    name2idx = {'tempo': 0, 'chord': 1, 'bar-beat': 2, 'type': 3, 'pitch': 4, 'duration': 5, 'velocity': 6, 'emotion': 7}
    if len(event2word.keys()) == 9 or ('key' in event2word):
        name2idx['key'] = 8

    # allow alias for bar-beat
    aliases = {
        'barbeat': 'bar-beat',
        'bar_beat': 'bar-beat',
        'bar-beat': 'bar-beat'
    }

    def normalize_name(n):
        if n in aliases:
            return aliases[n]
        return n

    def label_to_index(token_name, label):
        token_name = normalize_name(token_name)
        if isinstance(label, int):
            return int(label)
        # try parse int from string
        try:
            return int(label)
        except Exception:
            pass
        # lookup by vocabulary label
        try:
            return int(event2word[token_name][label])
        except Exception:
            return None

    def parse_conditions():
        if not args.conditions:
            return {}
        try:
            raw = json.loads(args.conditions)
        except Exception:
            print('[!] Could not parse --conditions JSON; ignoring')
            return {}
        cond_idx = {}
        for k, v in raw.items():
            k_norm = normalize_name(k)
            if k_norm not in name2idx:
                continue
            idx_val = label_to_index(k_norm, v)
            if idx_val is not None:
                cond_idx[k_norm] = idx_val
        return cond_idx

    def parse_force_indices():
        if not args.force_tokens:
            return {}, {}
        try:
            raw = json.loads(args.force_tokens)
        except Exception:
            print('[!] Could not parse --force_tokens JSON; ignoring')
            return {}, {}
        force_by_pos = {}
        force_raw_resolved = {}
        for k, v in raw.items():
            # resolve name or numeric position
            pos = None
            k_norm = normalize_name(k)
            if k_norm in name2idx:
                pos = name2idx[k_norm]
            else:
                try:
                    pos = int(k)
                except Exception:
                    pos = None
            if pos is None:
                continue
            idx_val = label_to_index(k_norm if k_norm in name2idx else k, v)
            if idx_val is None:
                continue
            force_by_pos[int(pos)] = int(idx_val)
            force_raw_resolved[str(k)] = int(idx_val)
        return force_by_pos, force_raw_resolved

    conditions_indices = parse_conditions()
    force_indices, force_resolved = parse_force_indices()

    # build sampling configuration per mode
    sampling_config = {}
    if args.temperature is not None:
        sampling_config['t'] = float(args.temperature)
    if args.top_p is not None:
        sampling_config['p'] = float(args.top_p)

    # normalize mode
    mode_normalized = MODE
    if MODE == 'inference':
        mode_normalized = 'inference-normal'

    if mode_normalized == 'inference-deterministic':
        # Very low temperature triggers argmax (deterministic) in sampling function
        sampling_config['t'] = 0.001
        sampling_config['p'] = 0.001
        # also ensure tokens that usually had None p get deterministic behavior via per-token overrides
        for token_name in ['barbeat', 'velocity', 'key']:
            sampling_config[f'{token_name}_p'] = 0.001
            sampling_config[f'{token_name}_t'] = 0.001
    elif mode_normalized == 'inference-normal':
        # keep defaults unless user overrides
        pass
    elif mode_normalized == 'inference-primed':
        # require conditions; keep sampling defaults
        if not conditions_indices:
            print('[!] inference-primed selected but no --conditions provided; proceeding without priming')
    elif mode_normalized == 'inference-forced':
        if not force_indices:
            print('[!] inference-forced selected but no --force_tokens provided; proceeding without forcing')

    # config
    n_class = []   # num of classes for each token
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))


    n_token = len(n_class)

    # init model
    net = TransformerModel(n_class, is_training=False)
    net.cuda()
    net.eval()
    
    # load model
    print('[*] load model from:',  path_saved_ckpt)
    
    
    try:
        net.load_state_dict(torch.load(path_saved_ckpt))
    except:
        state_dict = torch.load(path_saved_ckpt)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] 
            new_state_dict[name] = v
            
        net.load_state_dict(new_state_dict)


    # gen
    start_time = time.time()
    song_time_list = []
    words_len_list = []

    cnt_tokens_all = 0 
    sidx = 0
    while sidx < num_songs:
        # try:
        start_time = time.time()
        print('current idx:', sidx)
        res = None

        path_outfile = os.path.join(path_gendir, 'emo_{}_{}'.format( str(emotion_tag), get_random_string(10)))
        # choose key_tag only when available
        use_key_tag = args.key_tag if ('key' in name2idx and args.key_tag is not None) else None
        res, gen_key = net.inference_from_scratch(
            dictionary,
            emotion_tag,
            key_tag=use_key_tag,
            n_token=n_token,
            conditions=conditions_indices if mode_normalized in ['inference-normal','inference-primed','inference-deterministic','inference-forced'] else {},
            sampling_config=sampling_config,
            force_indices=force_indices if mode_normalized in ['inference-forced'] else None,
            generation_timeout=args.generation_timeout
        )
        

        if res is None:
            continue
        np.save(path_outfile + '.npy', res)
        write_midi(res, path_outfile + '.mid', word2event)

        # optional JSON tokens
        tokens_json_path = None
        if args.save_tokens_json == 1:
            try:
                tokens_json_path = path_outfile + '.tokens.json'
                with open(tokens_json_path, 'w', encoding='utf-8') as f:
                    json.dump(res.tolist(), f)
            except Exception:
                tokens_json_path = None

        # save detailed metadata
        meta = {
            'inference_mode': mode_normalized,
            'emotion_tag': int(emotion_tag),
            'key_tag': use_key_tag if use_key_tag is not None else None,
            'generated_key': int(gen_key) if gen_key is not None else None,
            'conditions_raw': json.loads(args.conditions) if args.conditions else None,
            'conditions_indices': {k:int(v) for k,v in conditions_indices.items()} if conditions_indices else {},
            'force_tokens_raw': json.loads(args.force_tokens) if args.force_tokens else None,
            'force_indices': {int(k):int(v) for k,v in force_indices.items()} if force_indices else {},
            'sampling_config': sampling_config,
            'dictionary_path': path_dictionary,
            'vocab_order': list(event2word.keys()),
            'checkpoint_dir': info_load_model[0] if info_load_model else None,
            'checkpoint_loss': info_load_model[1] if info_load_model else None,
            'n_token': int(n_token),
            'paths': {
                'npy': path_outfile + '.npy',
                'midi': path_outfile + '.mid',
                'tokens_json': tokens_json_path
            },
            'created_at': datetime.datetime.utcnow().isoformat() + 'Z'
        }
        with open(path_outfile + '.meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

        song_time = time.time() - start_time
        word_len = len(res)
        print('song time:', song_time)
        print('word_len:', word_len)
        words_len_list.append(word_len)
        song_time_list.append(song_time)

        sidx += 1

    
    print('ave token time:', sum(words_len_list) / sum(song_time_list))
    print('ave song time:', np.mean(song_time_list))

    runtime_result = {
        'song_time':song_time_list,
        'words_len_list': words_len_list,
        'ave token time:': sum(words_len_list) / sum(song_time_list),
        'ave song time': float(np.mean(song_time_list)),
    }

    # with open('runtime_stats.json', 'w') as f:
    #     json.dump(runtime_result, f)





if __name__ == '__main__':
    # -- training -- #
    if MODE == 'train':
        train()
    # -- inference -- #
    elif MODE in ['inference', 'inference-normal', 'inference-deterministic', 'inference-primed', 'inference-forced']:
        generate()
    else:
        pass
