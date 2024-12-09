import argparse
import sys
import mne
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from PCA import PCA
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
import joblib
import numpy as np
import time

RUNS = [[3, 7, 11], [4, 8, 12], [5, 9, 13], [6, 10, 14]]
GOOD_CHANNELS = ["FC3", "FCz", "FC4", "C3", "C1", "Cz", "C2", "C4"]

def determine_path(curr_subject, run):
    return f"./physionet.org/files/eegmmidb/1.0.0/S{curr_subject:03d}/S{curr_subject:03d}R{run:02d}.edf"

def fetch_raw(curr_subject, run):
    path = determine_path(curr_subject, run)
    return mne.io.read_raw_edf(path, preload=True)

def preproccess_raw(raws):
    raw = mne.concatenate_raws(raws)
    raw.rename_channels(lambda x: x.strip("."))
    raw.set_eeg_reference(ref_channels="average")
    raw.filter(7, 30)
    bad_channels = [ch for ch in raw.ch_names if ch not in GOOD_CHANNELS]
    raw.info["bads"] = bad_channels
    return raw

def get_epochs(raw):
    events, _ = mne.events_from_annotations(raw)
    picks = mne.pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
    )
    event_id = dict(T1=1, T2=2)
    epochs = mne.Epochs(
        raw=raw,
        events=events,
        event_id=event_id,
        tmin=-1.0,
        tmax=2.0,
        picks=picks,
        baseline=(None),
        preload=True,
        verbose=False,
    )
    epochs.drop_bad()
    return epochs

def get_data(epochs):
    psd_data = epochs.compute_psd(
        fmin=7, fmax=30, method="multitaper"
    )
    psds, freqs = psd_data.get_data(return_freqs=True)
    feature = psds.mean(axis=2)
    label = epochs.events[:, -1]

    return feature, label

def all():
    subject_list = [s for s in range(1, 110)]
    for run in RUNS:
        features = []
        labels = []
        for subject in subject_list:
            raws = []
            for r in run:
                raw = fetch_raw(subject, r)
                raws.append(raw)
            raw = preproccess_raw(raws)
            epochs = get_epochs(raw)
            feature, label = get_data(epochs)

            features.append(feature)
            labels.append(label)
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        X_train, X_test, y_train, y_test = train_test_split( features, labels, test_size=0.2, random_state=42)
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.99)),
                ("clf", LinearDiscriminantAnalysis()),
            ]
        )
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv)
        print("Cross-validation scores:", np.mean(scores))

def train(subject, run):
    raw = fetch_raw(subject, run)
    raw = preproccess_raw([raw])
    epochs = get_epochs(raw)
    feature, label = get_data(epochs)
    X_train, X_test, y_train, y_test = train_test_split( feature, label, test_size=0.2, random_state=42)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.99)),
            ("clf", LinearDiscriminantAnalysis()),
        ]
    )
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    print("cross_val_score:", scores.mean())
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, f"model_subject_{subject}_run_{run}.pkl")

def predict(subject, run):
    raw = fetch_raw(subject, run)
    raw = preproccess_raw([raw])
    epochs = get_epochs(raw)
    feature, label = get_data(epochs)
    try:
        pipeline = joblib.load(f"model_subject_{subject}_run_{run}.pkl")
    except:
        print("Model not found. Please train the model first")
        sys.exit(1)
    score = pipeline.score(feature, label)
    print("epoch nb: [prediction] [truth] equal?")
    for i, x, y in zip(range(len(feature)), feature, label):
        pred = pipeline.predict([x])
        print(f"Epoch {i}:  [{y}], {pred[0]}, {True if y == pred[0] else False}")
        time.sleep(0.5)
    print("Test score:", score)

def main():
    msg = "\
    --mode:\n\
        1. all: Run all the subjects and all the tasks\n\
        2. train: train the model on specific subjects\n\
        3. predict: predict the task on specific subjects and run\n\
    --subject: the subject number\n\
    --run: the run number\n"

    parser = argparse.ArgumentParser(description = msg)
    parser.add_argument("--mode", type=str, default="all", help="Run all the subjects and all the tasks")
    parser.add_argument("--subject", type=int, help="The subject number")
    parser.add_argument("--run", type=int, help="The run number")
    args = parser.parse_args()
    if len(sys.argv) == 1 or args.mode == "all":
        all()
    else:
        if not args.subject or not args.run:
            print("Please provide the subject number and a run number")
            sys.exit(1)
        elif args.mode == "train":
            train(args.subject, args.run)
        elif args.mode == "predict":
            predict(args.subject, args.run)
        else:
            print("Invalid mode")
            sys.exit(1)


if __name__ == "__main__":
    main()