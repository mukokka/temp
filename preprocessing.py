import os
from pathlib import Path
import numpy as np
import scipy
import mne
from matplotlib import pyplot as plt

def extract_current_data(subject_dir, fs):
    os.chdir(subject_dir)
    cnt = scipy.io.loadmat('cnt.mat')
    mrk = scipy.io.loadmat('mrk.mat')

    cnt_data = cnt["cnt"]
    cnt_real_data = [cnt_data[0, i][0, 0]['x']  for i in range(cnt_data.shape[1])]
    cnt_mi = np.concatenate([cnt_real_data[0], cnt_real_data[2], cnt_real_data[4]], axis=0)

    cnt_temp = [cnt_data[0][i][0, 0] for i in range(6)]
    clab = np.array(cnt_temp[0]['clab'])[0]

    mrk_data = mrk["mrk"]
    mrk_real_time = [mrk_data[0][i][0, 0]['time'] for i in range(6)]
    mrk_real_y = [mrk_data[0][i][0, 0]['y'] for i in range(6)]
    mrk_real_time = [mrk_real_time[i].flatten() for i in range(6)]
    mrk_real_y = [np.argmax(mrk_real_y[i], axis=0) for i in range(6)]

    duration0 = len(cnt_data[0, 0][0, 0]['x']) / fs * 1000
    duration2 = len(cnt_data[0, 2][0, 0]['x']) / fs * 1000
    mrk_mi_time = np.concatenate([mrk_real_time[0], mrk_real_time[2]+duration0, mrk_real_time[4]+duration0+duration2])
    mrk_mi_y = np.concatenate([mrk_real_y [0], mrk_real_y [2], mrk_real_y [4]])
    mrk_mi = {'time': mrk_mi_time, 'y': mrk_mi_y}
    return cnt_mi, mrk_mi, clab

def preprocess_eeg(raw_data, ch_names, fs):
    # convert to (C, T) for mne
    data_T = raw_data.T
    eog_keywords = ['EOG']
    ch_types = ['eog' if any(k in ch for k in eog_keywords) else 'eeg' for ch in ch_names]

    # create MNE Raw
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
    raw = mne.io.RawArray(data_T, info, verbose=False)

    # Notch filter (50Hz) -> Bandpass filter (8–30Hz)
    raw.notch_filter(freqs=50, verbose=False)
    raw.filter(l_freq=8, h_freq=30, verbose=False)

    print("ICA start")

    # ICA
    ica = mne.preprocessing.ICA( random_state=42, max_iter='auto', verbose=False)
    ica.fit(raw)

    eog_chs = [ch for ch, t in zip(ch_names, ch_types) if t == 'eog']
    if eog_chs:
        for eog in eog_chs:
            eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog)
            ica.exclude += eog_indices
            # ica.plot_scores(eog_scores)
    else:
        print("no EOG channels")

    raw_cleaned = ica.apply(raw.copy())

    return raw_cleaned

def preprocess_nirs(raw_data, ch_names, fs):
    data_T = raw_data.T
    hbo_keywords = ['lowWL']
    ch_types = ['hbo' if any(k in ch for k in hbo_keywords) else 'hbr' for ch in ch_names]

    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
    raw = mne.io.RawArray(data_T, info, verbose=False)

    raw.filter(l_freq=0.01, h_freq=0.2, verbose=False)
    return raw

def channel_selection(raw, channel_names):
    raw_channel_selected = raw.copy().pick_channels(channel_names)
    return raw_channel_selected

def create_mne_events(marker_times_ms, marker_y, fs):
    sample_points = (marker_times_ms / 1000 * fs).astype(int)
    events = np.column_stack([sample_points, np.zeros(len(sample_points), dtype=int), marker_y])
    unique, counts = np.unique(sample_points, return_counts=True)
    duplicates = unique[counts > 1]
    print(f"# duplicate samples: {len(duplicates)}")
    if len(duplicates) > 0:
        print(f"duplicate samples: {duplicates}")
    return events

NirsMyDataDir = Path(r'C:\Users\USER\python_files\test\EEG+fNIRS\fNIRS')
EegMyDataDir = Path(r'C:\Users\USER\python_files\test\EEG+fNIRS\EEG')
WorkingDir = Path(r'C:\Users\USER\python_files\test\EEG+fNIRS\02 M2NN')

number = 29
subdir_list = ['subject ' + f'{i:02d}' for i in range(1, number + 1)]

fs_nirs = 10
fs_eeg = 200

for subdir in subdir_list:
    print(f"start processing {subdir} ...")
    subject_dir_nirs = NirsMyDataDir / subdir
    subject_dir_eeg = EegMyDataDir / subdir / 'with occular artifact'

    cnt_mi_eeg, mrk_mi_eeg, clab_eeg = extract_current_data(subject_dir_eeg, fs=fs_eeg)
    cnt_mi_nirs, mrk_mi_nirs, clab_nirs = extract_current_data(subject_dir_nirs,fs=fs_nirs)

    print('shape of eeg', cnt_mi_eeg.shape)
    print('shape of nirs', cnt_mi_nirs.shape)

    ch_names_eeg = [ch[0] if isinstance(ch, np.ndarray) else ch for ch in clab_eeg]
    ch_names_nirs = [ch[0] if isinstance(ch, np.ndarray) else ch for ch in clab_nirs]

    print('channels of nirs',ch_names_nirs)
    print('channels of eeg',ch_names_eeg)

    # preprocessing
    eeg_mi_preprocessed = preprocess_eeg(raw_data=cnt_mi_eeg,ch_names=ch_names_eeg, fs=fs_eeg)
    nirs_mi_preprocessed = preprocess_nirs(raw_data=cnt_mi_nirs,ch_names=ch_names_nirs, fs=fs_nirs)

    print('shape of preprocessed eeg',eeg_mi_preprocessed.get_data().shape)
    print('shape of preprocessed nirs',nirs_mi_preprocessed.get_data().shape)

    MotorChannel_eeg = ['FCC3h', 'FCC5h', 'CCP3h', 'CCP5h', 'FCC4h', 'FCC6h', 'CCP4h', 'CCP6h']

    MotorChannel_nirs = ['C5CP5', 'C5FC5', 'C5C3', 'FC3FC5', 'FC3C3', 'FC3FC1', 'CP3CP5', 'CP3C3', 'CP3CP1', 'C1C3', 'C1FC1', 'C1CP1', 'C2FC2', 'C2CP2', 'C2C4', 'FC4FC2', 'FC4C4', 'FC4FC6', 'CP4CP6', 'CP4CP2', 'CP4C4', 'C6CP6', 'C6C4', 'C6FC6']
    MotorChannel_nirs_low = [ch + 'lowWL' for ch in MotorChannel_nirs]
    MotorChannel_nirs_high = [ch + 'highWL' for ch in MotorChannel_nirs]

    eeg_mi_selected = channel_selection(eeg_mi_preprocessed, MotorChannel_eeg)
    hbo_mi_selected = channel_selection(nirs_mi_preprocessed, MotorChannel_nirs_low)
    hbr_mi_selected = channel_selection(nirs_mi_preprocessed, MotorChannel_nirs_low)

    print('shape of selected eeg',eeg_mi_selected.get_data().shape)
    print('shape of selected hbo',hbo_mi_selected.get_data().shape)
    print('shape of selected hbr',hbr_mi_selected.get_data().shape)

    events_eeg = create_mne_events(mrk_mi_eeg['time'], mrk_mi_eeg['y'], fs=fs_eeg)
    events_nirs = create_mne_events(mrk_mi_nirs['time'], mrk_mi_nirs['y'], fs=fs_nirs)

    event_id = {'0': 0, '1': 1}

    epochs_eeg = mne.Epochs(eeg_mi_selected, events_eeg, event_id=event_id, tmin=-2, tmax=10, baseline=(-2, 0), preload=True, verbose=False)
    epochs_hbo = mne.Epochs(hbo_mi_selected, events_nirs, event_id=event_id, tmin=-2, tmax=10, baseline=(-2, 0), preload=True, verbose=False)
    epochs_hbr = mne.Epochs(hbr_mi_selected, events_nirs, event_id=event_id, tmin=-2, tmax=10, baseline=(-2, 0), preload=True, verbose=False)

    print('shape of eeg epochs',epochs_eeg.get_data().shape)               #(60, 8, 2401)
    print('shape of hbo epochs',epochs_hbo.get_data().shape)               #(60, 24, 121)
    print('shape of hbr epochs',epochs_hbr.get_data().shape)

    epochs_eeg_crop = epochs_eeg.crop(tmin=0, tmax=10)
    epochs_hbo_crop = epochs_hbo.crop(tmin=0, tmax=10)
    epochs_hbr_crop = epochs_hbr.crop(tmin=0, tmax=10)

    print('shape of eeg epochs',epochs_eeg_crop.get_data().shape)          #(60, 8, 2001)
    print('shape of hbo epochs',epochs_hbo_crop.get_data().shape)          #(60, 24, 101)
    print('shape of hbr epochs',epochs_hbr_crop.get_data().shape)

    sub_id = subdir.replace("subject ", "")
    save_dir = WorkingDir / "preprocessed_epochs" / f"sub{sub_id}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # save each modality epochs as .fif
    epochs_eeg_crop.save(save_dir / f"epochs_eeg-epo.fif", overwrite=True)
    epochs_hbo_crop.save(save_dir / f"epochs_hbo-epo.fif", overwrite=True)
    epochs_hbr_crop.save(save_dir / f"epochs_hbr-epo.fif", overwrite=True)
    print(f'sub{sub_id} SAVED!')
