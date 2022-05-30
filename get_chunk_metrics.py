"""
python get_chunk_metrics.py --test_path /home/ubuntu/data/test/audio/ --output_path /home/ubuntu/data/test/chunk_metrics.csv

"""
from forward import get_preds
from chunk_audio import chunk_audio
import pandas as pd
from docopt import docopt
import os

#suppress warnings
import warnings
warnings.filterwarnings('ignore')

SAMPLE_RATE = 22050
LMS_ARGS = {
    "n_fft": 2048,
    "n_mels": 64,
    "hop_length": int(SAMPLE_RATE * 0.02),
    "win_length": int(SAMPLE_RATE * 0.04),
}


def add_chunk_to_onset_offset(df, chunk_size=0.5):
    for i, row in df.iterrows():
        df.at[i, 'onset'] += row['onset'] + chunk_size*int(row['filename'])
        df.at[i, 'offset'] = row['offset'] + chunk_size*int(row['filename'])
    return df


def get_chunk_metrics(audio_path):
    chunks = chunk_audio(audio_path, 0.5, save=True)
    sr = 8000
    df = get_preds(wavlist=chunks)
    df = add_chunk_to_onset_offset(df)
    df['filename'] = audio_path
    return df

def main(df_test):
    df_ans = pd.DataFrame()
    filenames = df_test['path'].tolist()
    for file in filenames:
        df = pd.DataFrame()
        print(f'Filename: {file}')
        try:
            df = get_chunk_metrics(file)
        except:
            print(f'Skipping FILE: {file}')
        df_ans = df_ans.append(df)

    df_ans = df_ans.reset_index(drop=True)
    print(df_ans.to_markdown(showindex=False))
    return df_ans 


if __name__ == "__main__":
    args = docopt(__doc__)
    df_test = pd.read_csv(args['--test_path'])
    main_path = '/home/sanchit/res_vad'
    df_test['path'] = df_test['path'].apply(lambda x: os.path.join(main_path, x))
    df_ans  = main(df_test)
    df_ans.to_csv(args['--output_path'], index=False)



"""
1. Get chunks of audio from chunk_audio
2. Get predictions from forward.py
3. Add chunk_size to the preds (sorted by onset)
4. Save the predictions in a csv file
"""