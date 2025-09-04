import os
import cv2
import argparse
from tqdm import tqdm

def extract_from_video(video_path, out_folder, frames_per_clip=25, resize=(64,64)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return 0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize is not None:
            frame = cv2.resize(frame, resize)
        frames.append(frame)
    cap.release()

    # split into overlapping/non-overlapping clips of frames_per_clip
    count = 0
    if len(frames) == 0:
        return 0
    # Non-overlapping
    for i in range(0, len(frames), frames_per_clip):
        clip_frames = frames[i:i+frames_per_clip]
        if len(clip_frames) < frames_per_clip:
            # pad by repeating last frame
            while len(clip_frames) < frames_per_clip:
                clip_frames.append(clip_frames[-1])
        clip_folder = os.path.join(out_folder, f"clip_{count:04d}")
        os.makedirs(clip_folder, exist_ok=True)
        for j, fr in enumerate(clip_frames):
            fname = os.path.join(clip_folder, f"{j:04d}.jpg")
            cv2.imwrite(fname, fr)
        count += 1
    return count

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    video_files = [os.path.join(args.input_videos_dir, f) for f in os.listdir(args.input_videos_dir)
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    total = 0
    for v in tqdm(video_files):
        out_sub = os.path.join(args.output_dir, os.path.splitext(os.path.basename(v))[0])
        os.makedirs(out_sub, exist_ok=True)
        total += extract_from_video(v, out_sub, frames_per_clip=args.frames_per_clip, resize=(args.img_size, args.img_size))
    print(f"Extracted {total} clips.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_videos_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames_per_clip", type=int, default=25)
    parser.add_argument("--img_size", type=int, default=64)
    args = parser.parse_args()
    main(args)
model.py
python
Copy code
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_frame_encoder(input_shape=(64,64,3), dropout=0.3):
    """
    Small CNN to encode single frame into a feature vector.
    """
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    model = models.Model(inputs=inp, outputs=x, name='frame_encoder')
    return model

def build_cnn_lstm(seq_len=25, img_size=64, channels=3, n_classes=10, rnn_units=256, dropout=0.3):
    frame_shape = (img_size, img_size, channels)
    frame_encoder = build_cnn_frame_encoder(input_shape=frame_shape, dropout=dropout)
    # Input: (seq_len, h, w, c)
    seq_input = layers.Input(shape=(seq_len, img_size, img_size, channels), name='video_input')
    # Apply CNN to each frame
    td = layers.TimeDistributed(frame_encoder)(seq_input)  # (batch, seq_len, features)
    # Optionally add a TimeDistributed Dense/BN
    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=False))(td)  # final vector
    # You can also use return_sequences=True and stack LSTM + attention if desired.
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inputs=seq_input, outputs=out, name='cnn_lstm')
    return model
