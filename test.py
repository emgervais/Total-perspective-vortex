
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, ShuffleSplit
import mne
import sys
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage
from mne import Epochs
from mne.decoding import CSP
from joblib import dump
import numpy as np
import matplotlib.pyplot as plt
import time

runs = [[3, 7, 11], [4, 8, 12], [5, 9, 13], [6, 10, 14]]

def determine_path(curr_subject, run):
    return f"./eeg-motor-movementimagery-dataset-1.0.0/files/S{curr_subject:03d}/S{curr_subject:03d}R{run:02d}.edf"

def fetch_data(subject, viz, run=[6, 10, 14]):
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
    events, event_id = mne.events_from_annotations(raw)
    # picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = Epochs(raw, events, event_id=event_id, tmin=-1, tmax=4., baseline=None, preload=True)
    if viz:
        raw.compute_psd(average=False).plot()
        mne.viz.plot_events(events, raw.info['sfreq'])
        plt.show()

    X = epochs['T1', 'T2'].get_data()
    y = epochs['T1', 'T2'].events[:, -1] - 1
    return X, y, epochs

def create_pipeline():
    csp = CSP()

    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    mne.set_log_level('WARNING')
    pipe = make_pipeline(csp, lda)
    return pipe

def train(pipe, X, y):
    pipe.fit(X, y)
    scores = []
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
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    print(f'time elapsed after create pipe: {time.time() - start:.2f}')
    start = time.time()
    if '-all' in sys.argv:
        for r in runs:
            for i in range(1, 110):
                X, y, _ = fetch_data(i, viz, r)
                score = cross_val_score(pipe, X, y, cv=cv)
                accuracy = train(pipe, X, y)
                print(f'subject #{i} accuracy: {accuracy} mean: {score}')
                scores.append(accuracy)
                print(f'done with subject #{i}')
            results.append(np.mean(scores).round(3))
    else:
        X, y, _ = fetch_data(34, viz)
        print(f'time elapsed after fetch: {time.time() - start:.2f}')
        score = cross_val_score(pipe, X, y, cv=cv)
        print(f'time elapsed after cross: {time.time() - start:.2f}')
        accuracy = train(pipe, X, y)
        print(f'time elapsed after train: {time.time() - start:.2f}')
        print(f'subject  accuracy: {accuracy} mean: {np.mean(score)}')
    # for i, res in enumerate(results):
    #     print(f'experimen {i}:  accuracy = {res}')
    # print(f'Mean accuracy of 6 experiments: {np.mean(results)}')



if __name__ == '__main__':
    main()