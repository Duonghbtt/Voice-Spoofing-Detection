import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from spafe.features.lfcc import lfcc as spafe_lfcc
from spafe.utils.preprocessing import SlidingWindow

DIR_2019 = r'D:\DL\2019\LA\LA'
DIR_2021_KEYS = r'D:\DL\2021\keys\LA\CM\trial_metadata.txt'
DIR_2021_EVAL = r'D:\DL\2021\ASVspoof2021_LA_eval'

OUTPUT_ROOT = r'data\features'
OUTPUT_2019_DIR = os.path.join(OUTPUT_ROOT, 'output_npy_2019', 'output_lfcc')
OUTPUT_2021_DIR = os.path.join(OUTPUT_ROOT, 'output_npy_2021', 'output_lfcc')

os.makedirs(OUTPUT_2019_DIR, exist_ok=True)
os.makedirs(OUTPUT_2021_DIR, exist_ok=True)

def process_audio(file_path, duration=4, sr=16000):
    try:
        audio, _ = librosa.load(file_path, sr=sr)
        max_len = duration * sr
        if len(audio) > max_len:
            audio = audio[:max_len]
        else:
            audio = np.pad(audio, (0, max_len - len(audio)), 'constant')
        return audio
    except Exception:
        return None

def extract_lfcc(audio, sr=16000, n_filters=40):
    window_config = SlidingWindow(win_len=0.025, win_hop=0.010)
    lfcc_matrix = spafe_lfcc(
        sig=audio, 
        fs=sr, 
        num_ceps=n_filters,
        nfilts=n_filters,
        window=window_config
    )
    lfcc_matrix = lfcc_matrix.T
    lfcc_matrix = lfcc_matrix - np.mean(lfcc_matrix, axis=1, keepdims=True)
    return lfcc_matrix

def run_2019(subset):
    out_dir = os.path.join(OUTPUT_2019_DIR, subset)
    os.makedirs(out_dir, exist_ok=True)
    
    protocols_dir = os.path.join(DIR_2019, 'ASVspoof2019_LA_cm_protocols')
    audio_dir = os.path.join(DIR_2019, f'ASVspoof2019_LA_{subset}', 'flac')
    
    if subset == 'train':
        protocol_file = 'ASVspoof2019.LA.cm.train.trn.txt'
    else:
        protocol_file = f'ASVspoof2019.LA.cm.{subset}.trl.txt'
        
    df = pd.read_csv(os.path.join(protocols_dir, protocol_file), sep=' ', header=None, names=['speaker', 'filename', 'empty', 'attack', 'label'])
    
    labels_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"LFCC 2019 {subset}"):
        file_name = row['filename']
        input_path = os.path.join(audio_dir, file_name + '.flac')
        output_path = os.path.join(out_dir, file_name + '.npy')
        
        if os.path.exists(input_path):
            audio = process_audio(input_path)
            if audio is not None:
                feature = extract_lfcc(audio)
                np.save(output_path, feature)
                labels_list.append([file_name, row['label']])
                
    pd.DataFrame(labels_list, columns=['filename', 'label']).to_csv(
        os.path.join(OUTPUT_2019_DIR, f'labels_{subset}.csv'),
        index=False,
    )
    print(f"Hoan thanh LFCC 2019 {subset}")

def run_2021():
    out_dir = os.path.join(OUTPUT_2021_DIR, 'eval_2021')
    os.makedirs(out_dir, exist_ok=True)
    
    df_meta = pd.read_csv(DIR_2021_KEYS, sep=r'\s+', header=None)
    label_dict = {}
    for _, row in df_meta.iterrows():
        file_name = str(row[1]).split('-')[0]
        for val in row.dropna():
            if str(val).strip() == 'bonafide':
                label_dict[file_name] = 'bonafide'
                break
            if str(val).strip() == 'spoof':
                label_dict[file_name] = 'spoof'
                break
                
    trial_path = os.path.join(DIR_2021_EVAL, 'ASVspoof2021.LA.cm.eval.trl.txt')
    audio_dir = os.path.join(DIR_2021_EVAL, 'flac')
    df_trial = pd.read_csv(trial_path, sep=r'\s+', header=None)
    
    labels_list = []
    for _, row in tqdm(df_trial.iterrows(), total=len(df_trial), desc="LFCC 2021 eval"):
        file_name = None
        for val in row.dropna():
            if str(val).startswith('LA_E_'):
                file_name = str(val).split('-')[0]
                break
                
        if file_name and file_name in label_dict:
            input_path = os.path.join(audio_dir, file_name + '.flac')
            output_path = os.path.join(out_dir, file_name + '.npy')
            
            if os.path.exists(input_path):
                audio = process_audio(input_path)
                if audio is not None:
                    feature = extract_lfcc(audio)
                    np.save(output_path, feature)
                    labels_list.append([file_name, label_dict[file_name]])
                    
    pd.DataFrame(labels_list, columns=['filename', 'label']).to_csv(
        os.path.join(OUTPUT_2021_DIR, 'labels_eval_2021.csv'),
        index=False,
    )
    print("Hoan thanh LFCC 2021 eval")

if __name__ == "__main__":
    run_2019('train')
    run_2019('dev')
    run_2019('eval')
    run_2021()
