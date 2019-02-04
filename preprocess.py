import os
import joblib
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split
from tqdm import tqdm


SEQUENCE_LENGTH = 150000
RANDOM_STATE = 42
VAL_SIZE = 0.1
PROCESS_COUNT = 4


def train_reader(fname: str):
    for chunk in pd.read_csv(fname, chunksize=SEQUENCE_LENGTH):
        if len(chunk) == SEQUENCE_LENGTH:
            yield chunk


def test_reader(directory: str):
    files = sorted(os.listdir(directory))
    for file in files:
        df = pd.read_csv(os.path.join(directory, file))
        df['time_to_failure'] = 0.0
        yield df


def spectrum_image(df: pd.DataFrame):
    _, _, spec = spectrogram(df['acoustic_data'].values)
    image = np.log(spec + 1e-6)
    return np.transpose(image, [1, 0])


def _chunk_preprocess(chunk: pd.DataFrame):
    spec = spectrum_image(chunk)
    time = chunk['time_to_failure'].values[-1]
    return spec, time


def signal_preprocessor(reader):
    data = joblib.Parallel(n_jobs=PROCESS_COUNT)(
        joblib.delayed(_chunk_preprocess)(chunk)
        for chunk in tqdm(reader)
    )
    X = np.array([item[0]
                  for item in data])
    y = np.array([item[1]
                  for item in data])
    return X, y


if __name__ == '__main__':
    X_train, y_train = signal_preprocessor(train_reader('../input/train.csv'))
    X_test, _ = signal_preprocessor(test_reader('../input/test'))
    test_ids = [fname.replace('.csv', '') for fname in os.listdir('../input/test')]
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=VAL_SIZE,
                                                      random_state=RANDOM_STATE)

    np.save('../data/X_train.npy', X_train)
    np.save('../data/X_val.npy', X_val)
    np.save('../data/X_test.npy', X_test)
    np.save('../data/y_train.npy', y_train)
    np.save('../data/y_val.npy', y_val)
