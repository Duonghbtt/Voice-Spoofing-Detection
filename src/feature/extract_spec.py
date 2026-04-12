import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
from tqdm import tqdm

DIR_2019_PROTOCOLS = r'D:\DL\2019\LA\ASVspoof2019_LA_cm_protocols'
DIR_2019_AUDIO_BASE = r'D:\DL\2019\LA'
DIR_2021_KEYS = r'D:\DL\2021\keys\LA\CM\trial_metadata.txt'
DIR_2021_EVAL = r'D:\DL\2021\ASVspoof2021_LA_eval'

NPY_BASE_2019 = r'D:\DL\output_npy_2019'
NPY_BASE_2021 = r'D:\DL\output_npy_2021'

OUTPUT_SPEC_2019 = os.path.join(NPY_BASE_2019, 'output_spectrogram')
OUTPUT_SPEC_2021 = os.path.join(NPY_BASE_2021, 'output_spectrogram')

os.makedirs(OUTPUT_SPEC_2019, exist_ok=True)
os.makedirs(OUTPUT_SPEC_2021, exist_ok=True)

def process_audio_for_spec(file_path, duration=4, sr=16000):
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

def extract_spectrogram_live(audio, sr=16000):
    n_fft = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spectrogram = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)
    return spectrogram

def run_extraction_2019(subset):
    out_dir = os.path.join(OUTPUT_SPEC_2019, subset)
    os.makedirs(out_dir, exist_ok=True)
    
    if subset == 'train':
        protocol_file = 'ASVspoof2019.LA.cm.train.trn.txt'
    else:
        protocol_file = f'ASVspoof2019.LA.cm.{subset}.trl.txt'
        
    audio_dir = os.path.join(DIR_2019_AUDIO_BASE, f'ASVspoof2019_LA_{subset}', 'flac')
    df = pd.read_csv(os.path.join(DIR_2019_PROTOCOLS, protocol_file), sep=' ', header=None, names=['speaker', 'filename', 'empty', 'attack', 'label'])
    
    labels_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Spec 2019 {subset}"):
        file_name = row['filename']
        label_num = 1 if row['label'] == 'bonafide' else 0
        input_path = os.path.join(audio_dir, file_name + '.flac')
        output_path = os.path.join(out_dir, file_name + '.npy')
        
        if os.path.exists(input_path):
            audio = process_audio_for_spec(input_path)
            if audio is not None:
                feature = extract_spectrogram_live(audio)
                np.save(output_path, feature)
                labels_list.append([file_name, label_num])
                
    pd.DataFrame(labels_list, columns=['filename', 'label']).to_csv(os.path.join(OUTPUT_SPEC_2019, f'labels_{subset}.csv'), index=False)

def run_extraction_2021():
    out_dir = os.path.join(OUTPUT_SPEC_2021, 'eval')
    os.makedirs(out_dir, exist_ok=True)
    
    df_meta = pd.read_csv(DIR_2021_KEYS, sep=r'\s+', header=None)
    label_dict = {}
    for _, row in df_meta.iterrows():
        file_name = str(row[1]).split('-')[0]
        for val in row.dropna():
            if str(val).strip() == 'bonafide':
                label_dict[file_name] = 1
                break
            if str(val).strip() == 'spoof':
                label_dict[file_name] = 0
                break
                
    trial_path = os.path.join(DIR_2021_EVAL, 'ASVspoof2021.LA.cm.eval.trl.txt')
    audio_dir = os.path.join(DIR_2021_EVAL, 'flac')
    df_trial = pd.read_csv(trial_path, sep=r'\s+', header=None)
    
    labels_list = []
    for _, row in tqdm(df_trial.iterrows(), total=len(df_trial), desc="Spec 2021 eval"):
        file_name = None
        for val in row.dropna():
            if str(val).startswith('LA_E_'):
                file_name = str(val).split('-')[0]
                break
                
        if file_name and file_name in label_dict:
            input_path = os.path.join(audio_dir, file_name + '.flac')
            output_path = os.path.join(out_dir, file_name + '.npy')
            
            if os.path.exists(input_path):
                audio = process_audio_for_spec(input_path)
                if audio is not None:
                    feature = extract_spectrogram_live(audio)
                    np.save(output_path, feature)
                    labels_list.append([file_name, label_dict[file_name]])
                    
    pd.DataFrame(labels_list, columns=['filename', 'label']).to_csv(os.path.join(OUTPUT_SPEC_2021, 'labels_eval_2021.csv'), index=False)

def load_npy(feature_type, filename):
    if feature_type == 'spectrogram':
        path = os.path.join(OUTPUT_SPEC_2019, 'train', filename + '.npy')
    else:
        path = os.path.join(NPY_BASE_2019, f'output_{feature_type}', 'train', filename + '.npy')
        
    if os.path.exists(path):
        return np.load(path)
    return None

def visualize_final_comparison():
    label_path = os.path.join(NPY_BASE_2019, 'output_mfcc', 'labels_train.csv')
    df = pd.read_csv(label_path)
    bonafide_file = df[df['label'] == 1].iloc[0]['filename']
    spoof_file = df[df['label'] == 0].iloc[0]['filename']
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    hop_length = 160

    files = [bonafide_file, spoof_file]
    labels = ['BONAFIDE', 'SPOOF']

    for col in range(2):
        filename = files[col]
        label_text = labels[col]
        
        spec = load_npy('spectrogram', filename)
        if spec is not None:
            img0 = librosa.display.specshow(spec, x_axis='time', y_axis='linear', ax=axes[0, col], hop_length=hop_length)
            axes[0, col].set_title(f'{label_text} - Spectrogram (Log-scale)')
            fig.colorbar(img0, ax=axes[0, col], format='%+2.0f dB')
        
        mfcc = load_npy('mfcc', filename)
        if mfcc is not None:
            img1 = librosa.display.specshow(mfcc, x_axis='time', ax=axes[1, col], hop_length=hop_length)
            axes[1, col].set_title(f'{label_text} - MFCC (Standardized)')
            fig.colorbar(img1, ax=axes[1, col])
            
        lfcc = load_npy('lfcc', filename)
        if lfcc is not None:
            img2 = librosa.display.specshow(lfcc, x_axis='time', ax=axes[2, col], hop_length=hop_length)
            axes[2, col].set_title(f'{label_text} - LFCC (Standardized)')
            fig.colorbar(img2, ax=axes[2, col])

    plt.tight_layout()
    plt.savefig('final_report_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_extraction_2019('train')
    run_extraction_2019('dev')
    run_extraction_2019('eval')
    run_extraction_2021()
    visualize_final_comparison()