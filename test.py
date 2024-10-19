
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.model_selection import StratifiedKFold
import mne
import sys
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage
from mne import Epochs
from mne.decoding import CSP
import os
from joblib import dump, load
import numpy as np
import matplotlib.pyplot as plt
import time
import logging

runs = [[3, 7, 11], [4, 8, 12], [5, 9, 13], [6, 10, 14]]

def determine_path(curr_subject, run):
    return f"./eeg-motor-movementimagery-dataset-1.0.0/files/S{curr_subject:03d}/S{curr_subject:03d}R{run:02d}.edf"

def fetch_data(subject, viz, run=[3, 7, 11]):
    raw_files = []
    for r in run:
        raw_files_imagery = [read_raw_edf(determine_path(subject, r), preload=True, stim_channel='auto')]
        raw_imagery = concatenate_raws(raw_files_imagery)
        raw_files.append(raw_imagery)

    raw = concatenate_raws(raw_files)
    eegbci.standardize(raw)
    if viz:
        montage = make_standard_montage("standard_1020")
        raw.set_montage(montage)
        raw.compute_psd(average=False).plot()
    raw.rename_channels(lambda x: x.strip('.'))
    raw.filter(7, 30, fir_design='firwin')
    raw.notch_filter(freqs=[50], fir_design='firwin')
    events, event_id = mne.events_from_annotations(raw)
    epochs = Epochs(raw, events, event_id=event_id, tmin=-0.5, tmax=4.5, baseline=None, preload=True)
    if viz:
        raw.compute_psd(average=False).plot()
        mne.viz.plot_events(events, raw.info['sfreq'])
        plt.show()
    X = epochs['T1', 'T2'].get_data()
    # X = add_noise(X)
    y = epochs['T1', 'T2'].events[:, -1]
    return X, y, epochs
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
def create_pipeline():
    csp = CSP(n_components=5, reg='ledoit_wolf')
    clf = LogisticRegression(max_iter=1000, solver='saga', penalty='l2', C=0.1)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    pca = PCA(n_components=5)
    pipe = make_pipeline(csp, lda)
    return pipe

def add_noise(X, noise_level=0.01):
    noise = np.random.randn(*X.shape) * noise_level
    return X + noise

def predict(X, y, pipe=None):
    try:
        if pipe == None:
            pipe = load("pipe")
    except:
        print("please train before trying to predict")
        exit(1)
    scores = []
    pipe.fit(X, y)
    for n in range(X.shape[0]):
        pred = pipe.predict(X[n:n + 1, :, :])[0]
        # truth = y[n:n + 1][0]
        # print(f"epoch_{n:2} =      [{pred}]           [{truth}]      {'' if pred == truth else False}")
        scores.append(1 - np.abs(pred - y[n:n + 1][0]))
    return np.mean(scores).round(3)

def main():
    scores = []
    results = []
    viz = False
    if '-v' in sys.argv:
        viz = True
    start = time.time()
    pipe = create_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f'time elapsed after create pipe: {time.time() - start:.2f}')
    start = time.time()
    if 'all' in sys.argv:
        for r in runs:
            for i in range(1, 110):
                X, y, _ = fetch_data(i, viz, r)
                score = cross_val_score(pipe, X, y, cv=cv, n_jobs=-1)
                accuracy = predict(X, y, pipe)
                print(f'subject #{i} accuracy: {accuracy} mean: {np.mean(score):.2f}')
                scores.append(np.mean(score))
            results.append(np.mean(scores).round(3))
        for i, res in enumerate(results):
            print(f'experimen {i}:  accuracy = {res}')
        print(f'Mean accuracy of 6 experiments: {np.mean(results):.2f}')
    elif 'train' in sys.argv or 'predict' in sys.argv:
        X, y, _ = fetch_data(1, viz)
        print(f'time elapsed after fetch: {time.time() - start:.2f}')
        score = cross_val_score(pipe, X, y, cv=cv, n_jobs=-1)
        print(score)
        print(f'time elapsed after cross: {time.time() - start:.2f}')
        if('train' in sys.argv):
            pipe.fit(X, y)
            dump(pipe, "pipe")
            print(f'time elapsed after train: {time.time() - start:.2f}')
        if 'predict' in sys.argv:
            accuracy = predict(X, y)
            print(f'time elapsed after predict: {time.time() - start:.2f}')
            print(f'subject  accuracy: {accuracy} mean: {np.mean(score).round(3)}')
    else:
        print('please choose one or many of the mode -> train, predict or all')
        exit(1)



if __name__ == '__main__':
    mne.set_config('MNE_LOGGING_LEVEL', 'CRITICAL')
    main()