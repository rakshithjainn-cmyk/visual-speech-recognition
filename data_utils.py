import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def load_frame_sequence(folder_path, seq_len=25, img_size=(64,64), grayscale=False):
    """
    Load ordered frames from folder_path into a fixed-length sequence.
    If there are fewer frames than seq_len, last frame is repeated.
    """
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    # If no frames, return zeros
    if len(files) == 0:
        return np.zeros((seq_len, img_size[0], img_size[1], 1 if grayscale else 3), dtype=np.float32)

    frames = []
    for fname in files:
        img = cv2.imread(os.path.join(folder_path, fname))
        if img is None:
            continue
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, img_size)
            img = np.expand_dims(img, axis=-1)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
        frames.append(img.astype(np.float32) / 255.0)

    # If too many frames, sample uniformly
    if len(frames) > seq_len:
        idxs = np.linspace(0, len(frames) - 1, seq_len).astype(int)
        frames = [frames[i] for i in idxs]
    # If fewer, pad by repeating last frame
    while len(frames) < seq_len:
        frames.append(frames[-1])

    return np.stack(frames, axis=0)  # shape: (seq_len, h, w, c)


def scan_dataset(data_dir):
    """
    Scans dataset directory and returns list of (sample_folder, label)
    data_dir expected to contain class subfolders.
    """
    samples = []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        for sample in os.listdir(cls_dir):
            sample_path = os.path.join(cls_dir, sample)
            if os.path.isdir(sample_path):
                samples.append((sample_path, cls))
    return samples, class_names


class FrameSequenceGenerator(Sequence):
    def __init__(self, samples, labels, batch_size=8, seq_len=25, img_size=(64,64),
                 shuffle=True, grayscale=False, augment_fn=None, n_classes=None):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.img_size = img_size
        self.shuffle = shuffle
        self.grayscale = grayscale
        self.augment_fn = augment_fn
        self.indexes = np.arange(len(self.samples))
        self.n_classes = n_classes
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.samples) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_samples = [self.samples[i] for i in batch_indexes]
        X = []
        y = []
        for s in batch_samples:
            seq = load_frame_sequence(s, seq_len=self.seq_len, img_size=self.img_size, grayscale=self.grayscale)
            if self.augment_fn is not None:
                # augment_fn must accept and return a sequence of frames (numpy array)
                seq = self.augment_fn(seq)
            X.append(seq)
            y.append(self.labels[self.samples.index(s)])
        X = np.array(X, dtype=np.float32)  # (batch, seq_len, h, w, c)
        y = np.array(y)
        # one-hot
        if self.n_classes is not None:
            y = np.eye(self.n_classes)[y]
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
