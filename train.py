import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from data_utils import scan_dataset, FrameSequenceGenerator
from model import build_cnn_lstm
from sklearn.preprocessing import LabelEncoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to dataset root containing train/val/test")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=25)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weights_out", default="best_model.h5")
    return parser.parse_args()

def main():
    args = parse_args()
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    train_samples, class_names = scan_dataset(train_dir)
    val_samples, _ = scan_dataset(val_dir)

    # Create list of sample paths
    train_paths = [s for s, _ in train_samples]
    train_labels = [label for _, label in train_samples]
    val_paths = [s for s, _ in val_samples]
    val_labels = [label for _, label in val_samples]

    # Encode labels
    le = LabelEncoder()
    le.fit(class_names)
    train_labels_enc = le.transform(train_labels)
    val_labels_enc = le.transform(val_labels)

    n_classes = len(class_names)
    print("Classes:", class_names)

    train_gen = FrameSequenceGenerator(train_paths, train_labels_enc, batch_size=args.batch_size,
                                       seq_len=args.seq_len, img_size=(args.img_size,args.img_size),
                                       shuffle=True, grayscale=False, augment_fn=None, n_classes=n_classes)

    val_gen = FrameSequenceGenerator(val_paths, val_labels_enc, batch_size=args.batch_size,
                                     seq_len=args.seq_len, img_size=(args.img_size,args.img_size),
                                     shuffle=False, grayscale=False, augment_fn=None, n_classes=n_classes)

    model = build_cnn_lstm(seq_len=args.seq_len, img_size=args.img_size, channels=3, n_classes=n_classes)
    model.compile(optimizer=Adam(learning_rate=args.lr), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        ModelCheckpoint(args.weights_out, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, verbose=1, restore_best_weights=True)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=callbacks)

    # Save label encoder
    import joblib
    joblib.dump(le, "label_encoder.pkl")
    print("Training finished. Model saved to", args.weights_out)

if __name__ == "__main__":
    main()
