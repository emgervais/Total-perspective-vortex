
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import mne
import sys
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage
from mne import Epochs
from mne.decoding import CSP
from joblib import dump, load
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
runs = [[3, 7, 11], [4, 8, 12], [5, 9, 13], [6, 10, 14]]
def determine_path(curr_subject, run):
    return f"./eeg-motor-movementimagery-dataset-1.0.0/files/S{curr_subject:03d}/S{curr_subject:03d}R{run:02d}.edf"

def compute_channel_importance(epochs):
    """Compute channel importance using mutual information"""
    n_channels = len(epochs.ch_names)
    importance_scores = np.zeros(n_channels)
    
    # Compute power for each channel
    data = epochs['T1', 'T2'].get_data()
    y = epochs['T1', 'T2'].events[:, -1] - 2
    for ch in range(n_channels):
        # Compute band power features
        channel_power = np.mean(data[:, ch, :] ** 2, axis=1)
        importance_scores[ch] = mutual_info_classif(
            channel_power.reshape(-1, 1), y, random_state=42
        )[0]
    
    return importance_scores

def select_best_channels(raw, epochs, n_channels=32):
    """Select best channels based on mutual information scores"""
    importance_scores = compute_channel_importance(epochs)
    
    # Get channel indices sorted by importance
    best_channel_idx = np.argsort(importance_scores)[::-1][:n_channels]
    
    # Get channel names
    best_channels = [epochs.ch_names[idx] for idx in best_channel_idx]
    
    return mne.pick_channels(raw.info['ch_names'], include=best_channels, ordered=True)

def fetch_data(subject, viz, select=32, f=30, run=[3, 7, 11]):
    raw_files = []
    for r in run:
        raw_files_imagery = [read_raw_edf(determine_path(subject, r), preload=True, stim_channel='auto')]
        raw_imagery = concatenate_raws(raw_files_imagery)
        raw_files.append(raw_imagery)
    raw = concatenate_raws(raw_files)
    raw.resample(sfreq=240.0)
    eegbci.standardize(raw)
    if viz:
        montage = make_standard_montage("standard_1020")
        raw.set_montage(montage)
        raw.compute_psd(average=False).plot()
    raw.rename_channels(lambda x: x.strip('.'))
    raw.notch_filter([50, 60], method="fir")
    raw.filter(8, f, fir_design='firwin', skip_by_annotation="edge")
    events, event_id = mne.events_from_annotations(raw)
    temp_epochs = Epochs(raw, events, event_id=event_id, baseline=(None, 0), picks='eeg', preload=True)
    picks = select_best_channels(raw, temp_epochs, select)
    epochs = Epochs(raw, events, event_id=event_id, tmin=-0.2, tmax=0.5, baseline=(None, 0), picks=picks, preload=True)
    if viz:
        raw.compute_psd(average=False).plot()
        mne.viz.plot_events(events, raw.info['sfreq'])
        plt.show()
    X = epochs['T1', 'T2'].get_data()
    y = epochs['T1', 'T2'].events[:, -1] - 2
    return X, y, epochs

param_grid = {
    "csp__n_components": [4,5,6,7,8],
    "csp__reg": [None, 0.06, 0.08,0.1,0.12,0.15],
}

#For freq 20 and features24 and params {'csp__n_components': 8, 'csp__reg': 0.08}: score=0.800
def create_pipeline():
    csp = CSP()
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    pipe = Pipeline(steps=[("csp", csp), ("lda", lda)])
    return pipe

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
def create_pipe_grid(n, r):
    csp = CSP(n_components=n, reg=r)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    return make_pipeline(csp, lda)

def run_grid_search(freq, features, csp_reg, csp_n, runs, cv, viz):
    """Function to execute one combination of parameters and return the results."""
    run_score = []
    for r in runs:
        scores = []
        for i in range(1, 110):
            X, y, _ = fetch_data(i, viz, features, freq, r)
            pipe = create_pipe_grid(csp_n, csp_reg)
            score = cross_val_score(pipe, X, y, cv=cv, n_jobs=-1)
            scores.append(np.mean(score).round(3))
        run_score.append(np.mean(scores).round(3))
    avg_run_score = np.mean(run_score).round(3)
    logger.info(f'freq: {freq}, features: {features}, csp_reg: {csp_reg}, csp_n: {csp_n} = {avg_run_score}')
    return {'params': f'freq: {freq}, features: {features}, csp_reg: {csp_reg}, csp_n: {csp_n}', 'scores': avg_run_score}

def main():
    scores = []
    results = []
    viz = False
    if '-v' in sys.argv:
        viz = True
    start = time.time()
    pipe = create_pipeline()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
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
    elif 'grid' in sys.argv:
        results = []
        param_combinations = [(freq, features, csp_reg, csp_n) 
                              for freq in [13, 20, 30]
                              for features in [8, 16, 24, 32, 40]
                              for csp_reg in [None, 0.06, 0.08, 0.1, 0.12, 0.15]
                              for csp_n in [4, 5, 6, 7, 8]]

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_grid_search, freq, features, csp_reg, csp_n, runs, cv, viz) for (freq, features, csp_reg, csp_n) in param_combinations]

            for future in as_completed(futures):
                results.append(future.result())
        for r in results:
            logger.info(r)
            logger.info(f'param: {r.params} result = {np.mean(r.scores).round(3)}')
    elif 'train' in sys.argv or 'predict' in sys.argv:
        X, y, _ = fetch_data(2, viz)
        # print(f'time elapsed after fetch: {time.time() - start:.2f}')
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
    logger = logging.getLogger('MyLogger')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('output.log')
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)
    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    mne.set_config('MNE_LOGGING_LEVEL', 'CRITICAL')
    main()