import os
from pathlib import Path
import numpy as np
import scipy
import mne
from mne import pick_types
from scipy.io import loadmat
from mne.channels import make_dig_montage, DigMontage
from mne_icalabel import label_components

from scipy.signal import welch
import pandas as pd

def creat_raw(raw, ch_names, fs):
    # convert to (C, T) for mne
    data = raw.T * 1e-6
    # data = raw.T
    eog_keywords = ['EOG']
    ch_types = ['eog' if any(k in ch for k in eog_keywords) else 'eeg' for ch in ch_names]
    # create MNE Raw
    #print(ch_names)
    #print(ch_types)
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    raw.pick_types(eeg=True, eog=False)
    return raw

def SetMontage(raw, mnt):
    '''
    labels = mnt.clab
    pos_3d = mnt.pos_3d.T
    print(labels)
    print(pos_3d.shape)

    mask = [lab not in ('VEOG', 'HEOG') for lab in labels]
    labels_eeg = [lab for lab, keep in zip(labels, mask) if keep]
    pos_3d_eeg = pos_3d[mask, :]
    print(labels_eeg)
    print(pos_3d_eeg.shape)

    rmax = np.linalg.norm(pos_3d_eeg, axis=1).max()
    scale = 0.09 / rmax
    pos_m = pos_3d_eeg * scale

    ch_pos = {
        label: pos.tolist()
        for label, pos in zip(labels_eeg, pos_m)
        if not any(map(lambda v: v != v, pos))
    }

    print("positions：", ch_pos)

    montage = make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage)
    '''
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='warn')
    return raw

def BaselineCorrection(raw):

    data = raw.get_data()
    baseline_mean = data.mean(axis=1, keepdims=True)
    corrected_data = data - baseline_mean

    print("max value of mean：", np.abs(baseline_mean).max())
    print("max value of mean after correction：", np.abs(corrected_data).max())

    return mne.io.RawArray(corrected_data, raw.info)

def Filter(raw):
    raw_notch = raw.notch_filter(freqs=50)
    raw_notch.plot(scalings='auto', title='After Notch Filter', show=True, block=True, duration=5, n_channels=25)

    #raw.filter(l_freq=1, h_freq=30, fir_design='firwin', phase='zero')
    raw_bp = raw_notch.filter(l_freq=8, h_freq=30.0)
    raw_bp.plot(scalings='auto', title='After Bandpass Filter', show=True, block=True, duration=5, n_channels=25)

    '''
    raw.filter(
        l_freq=0.5,
        h_freq=50.0,
        method='iir',
        iir_params=dict(
            order=4,
            ftype='cheby2',
            rs=40 
        ),
        phase='zero'
    )

    raw.filter(
        l_freq=4.0,
        h_freq=30.0,
        method='iir',
        iir_params=dict(
            order=6,
            ftype='butter'
        ),
        phase='zero'
    )
    '''
    return raw_bp

def ICA(raw):
    ica = mne.preprocessing.ICA(n_components=30, random_state=97, max_iter="auto", method="infomax")
    ica.fit(raw)
    ica.plot_components(inst=raw)

    labels = label_components(raw, ica, method='iclabel')
    print(labels)

    target_labels = ['brain', 'other']
    artifact_indices = [
        i for i, (label, prob) in enumerate(zip(labels['labels'], labels['y_pred_proba']))
        if label not in target_labels and prob >= 0.5
    ]
    '''
    target_artifact = ['eye blink', 'eye movement']  # 只去掉眨眼，如果想包含眼动可以写 ['eye_blink', 'eye_movement']

    artifact_indices = [
        i for i, (label, prob) in enumerate(zip(labels['labels'], labels['y_pred_proba']))
        if label in target_artifact and prob >= 0.1  # 概率阈值自己调，比如 0.5
    ]
    '''

    # artifact_indices = [0, 8, 9]
    print(f"exclued components: {artifact_indices}")

    if len(artifact_indices) > 0:
        ica.exclude = artifact_indices
        ica.plot_properties(raw, picks=ica.exclude)
        raw_ica = raw.copy()
        raw_ica.load_data()               # ica.apply는 데이터에 직접 접근이 필요해 메모리에 로드해야함
        ica.apply(raw_ica)                # ica apply
    else:
        raw_ica = raw.copy()
    return raw_ica

def channel_selection(raw, channel_names):
    raw_selected = raw.copy().pick_channels(channel_names)
    raw_selected.plot(scalings='auto', title='After Channel Selection', show=True, block=True)
    return raw_selected

def create_mne_events(marker_times, marker_y):
    sample_points = marker_times.astype(int)
    events = np.column_stack([sample_points, np.zeros(len(sample_points), dtype=int), marker_y])
    return events

NirsMyDataDir = Path(r'xxx')
EegMyDataDir = Path(r'xxx')
WorkingDir = Path(r'xxx')

number = 29
subdir_list = ['subject ' + f'{i:02d}' for i in range(1, number + 1)]

fs = 200
MotorChannel_eeg = ['FCC5h', 'FCC3h', 'CCP5h', 'CCP3h', 'FCC4h', 'FCC6h', 'CCP4h', 'CCP6h']

eeg_pre = 2
eeg_post = 10

for subdir in subdir_list:
    print(f"start processing {subdir} ...")
    subject_dir = EegMyDataDir / subdir / 'with occular artifact'

    os.chdir(subject_dir)
    cnt = loadmat('cnt.mat', struct_as_record=False, squeeze_me=True)
    mrk = loadmat('mrk.mat', struct_as_record=False, squeeze_me=True)
    mnt = loadmat('mnt.mat', struct_as_record=False, squeeze_me=True)

    cnt_data = cnt["cnt"]
    mrk_data = mrk["mrk"]
    mnt = mnt["mnt"]
    clab = np.array(cnt_data[0].clab)
    ch_names = [ch[0] if isinstance(ch, np.ndarray) else ch for ch in clab]

    all_epochs = []
    all_y = []
    epochs_concat = []

    for i in [0, 2, 4]:
        eeg = cnt_data[i].x
        print('eeg.shape = ', eeg.shape)

        mark_time = mrk_data[i].time
        mark_y = mrk_data[i].y
        mark_event = mrk_data[i].event.desc
        mark_time_sample = mark_time / 1000 * fs

        print(f"Session {i} EEG shape: {eeg.shape}")

        # creat raw data
        raw = creat_raw(raw=eeg,ch_names=ch_names, fs=fs)
        raw.plot(scalings='auto', title='Raw Data', show=True, block=True, duration=5, n_channels=25)

        # set montage
        raw_m = SetMontage(raw, mnt)
        raw_m.plot_sensors(kind='topomap', show_names=True, show=True, block=True,)

        # CAR(common average reference)
        raw_car = raw_m.set_eeg_reference('average')
        print(raw_car.info)
        raw.plot(scalings='auto', title='CAR Data', show=True, block=True, duration=5, n_channels=25)

        # baseline correction
        raw_corrected = BaselineCorrection(raw_car)
        raw_corrected.plot(scalings='auto', title='Baseline Correction', show=True, block=True, duration=5, n_channels=25)

        # filter
        raw_filtered = Filter(raw_corrected)

        # ICA
        raw_ica = ICA(raw_filtered)
        raw_ica.plot(scalings='auto', title='ICA', show=True, block=True, duration=5, n_channels=25)

        # channel_selcetion
        raw_selected = channel_selection(raw_ica, MotorChannel_eeg)
        #print('shape of selected eeg',raw_selected.get_data().shape)
        #raw_selected.plot(scalings='auto', title='channel selection', show=True, block=True, duration=5, n_channels=25)
        
        epoch_length = int(round((eeg_pre + eeg_post) * fs))
        pre_sample = int(round(eeg_pre * fs))

        print('epoch_length = ',epoch_length)
        print('pre_sample = ',pre_sample)

        n_trials = 20
        epochs_eeg = np.full((n_trials, 8, epoch_length), np.nan, dtype=np.float64)
        print('eeg_epoch.shape: ', epochs_eeg.shape)

        for t in range(20):

            print('t = ', t)
            event_start = mark_time_sample[t]
            print('event_start = ', event_start)

            a = int(event_start - pre_sample)
            b = int(a + epoch_length)
            print('a = ', a)
            print('b = ', b)

            ep = raw_selected.get_data()[:, a:b]     # (n_ch, epoch_length)
            ep = ep - ep[:, :pre_sample].mean(axis=1, keepdims=True)
            print(ep)
            print(ep.shape)
            epochs_eeg[t] = ep

        print(epochs_eeg.shape)
        print(epochs_eeg[0].shape)

        '''
        events = create_mne_events(mark_time_sample, mark_event)
        print(events)
        event_id = {'left': 16, 'right': 32}
        mne.viz.plot_events(events, sfreq=fs, event_id=event_id)
        print(events)

        epochs_eeg = mne.Epochs(raw = raw_selected, events = events, event_id=event_id, tmin=-2, tmax=10, baseline=(-2, 0), preload=True, verbose=False)
        #epochs_eeg_crop = epochs_eeg.crop(tmin=0, tmax=10)

        print("n_epochs =", len(epochs_eeg))
        print(mne.viz.get_browser_backend())
        epochs_eeg.plot(n_epochs=1, n_channels=8, title="Random Epochs", block=True)
        '''

        all_epochs.append(epochs_eeg)
        all_y.append(mark_y)
        print(f"Session {i} processed. Epoch shape: {epochs_eeg.shape}")          # (20, 8, 2401)

    # 合并所有 session 的 epochs
    epochs_concat = np.concatenate(all_epochs, axis=0)                            # (60, 8, 2400)
    y_concat = np.concatenate(all_y, axis=1)                                      # (2, 60)
    print('epochs_concat.shape = ', epochs_concat.shape)
    print('y_concat.shape = ', y_concat.shape)

    sub_id = subdir.replace("subject ", "")
    save_dir = WorkingDir / "test" / f"sub{sub_id}"
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(save_dir, "1005_eeg_epochs.npy"), epochs_concat)
    np.save(os.path.join(save_dir, "labels.npy"), mark_y)

    #print("saved to:", os.path.join(save_dir, "ICA30_eeg_epochs.npy"))
    #print("saved to:", os.path.join(save_dir, "labels.npy"))

    print(f'sub{sub_id} SAVED!')
