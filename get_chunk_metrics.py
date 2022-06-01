"""
Usage:
    get_chunk_metrics.py [--test_path=<test_path>] [--output_path=<output_path>] [--skip_files=<skip_files>]
"""
from forward import get_preds
from chunk_audio import chunk_audio
import pandas as pd
from docopt import docopt
import os
from typing import List
import numpy as np
import pickle

#suppress warnings
import warnings
warnings.filterwarnings('ignore')

SAMPLE_RATE = 8000

LMS_ARGS = {
    "n_fft": 2048,
    "n_mels": 64,
    "hop_length": int(SAMPLE_RATE * 0.02),
    "win_length": int(SAMPLE_RATE * 0.04),
}


def add_chunk_to_onset_offset(df, chunk_size=0.5):
    for i, row in df.iterrows():
        df.at[i, 'onset'] = row['onset'] + chunk_size*int(row['filename'])
        df.at[i, 'offset'] = row['offset'] + chunk_size*int(row['filename'])
    return df


def get_chunk_metrics(audio_path: str, sliding = False, window = 2) -> pd.DataFrame:
    chunks = chunk_audio(audio_path, 0.5)

    if sliding:
        chunks = get_sliding_window_metric(chunks, window) 
    sr = 8000
    df = get_preds(wavlist=chunks)
    df = add_chunk_to_onset_offset(df)
    df['filename'] = audio_path
    return df


def get_sliding_window_metric(chunks: List[np.ndarray], window: int = 2) -> List[np.ndarray]:
    """
    [1,2,3,4,5]
    Slide over the window. And discard the last matrix, if the shapes do not match with the second last
    
    """
    chunk_fin = []
    for i in range(len(chunks) - window):
        npys = np.hstack(chunks[i:i+window])
        chunk_fin.append(npys)
    if chunk_fin[-1].shape != chunk_fin[-2].shape:
        chunk_fin.pop(-1)
    return chunk_fin


def main(df_test: pd.DataFrame, sliding = False, window = 2) -> pd.DataFrame:
    df_ans = pd.DataFrame()
    filenames = df_test['path'].tolist()
    skipped_files = []
    for file in filenames:
        df = pd.DataFrame()
        print(f'Filename: {file}')
        try:
            df = get_chunk_metrics(file, sliding, window)
        except:
            print(f'Skipping FILE: {file}')
            skipped_files.append(file)
        df_ans = df_ans.append(df)

    df_ans = df_ans.reset_index(drop=True)
    print(df_ans.to_markdown(showindex=False))
    return df_ans, skipped_files


if __name__ == "__main__":
    args = docopt(__doc__)
    df_test = pd.read_csv(args['--test_path'])
    main_path = '/home/sanchit/res_vad'
    sliding = True
    df_test['path'] = df_test['path'].apply(lambda x: os.path.join(main_path, x))
    df_ans, skipped_files  = main(df_test, sliding)
    with open(args['--skip_files'], 'wb') as f:
        pickle.dump(skipped_files, f)
    df_ans.to_csv(args['--output_path'], index=False)



"""
1. Get chunks of audio from chunk_audio
2. Get predictions from forward.py
3. Add chunk_size to the preds (sorted by onset)
4. Save the predictions in a csv file
"""