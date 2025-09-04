import os
import argparse
import numpy as np
from data_utils import scan_dataset, FrameSequenceGenerator
from model import build_cnn_lstm
import joblib
from sklearn.metrics import classification_report, confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--seq_len", type=int, default=25)
    parser.add_argument("--img_size", type=int, default=64)
    return parser.parse_args()

def main():
    args = parse_args()
    test_dir = os.path.join(args.data_dir, "test")
    samples, class_names = scan_dataset(test_dir)
    if len(samples) == 0:
        raise ValueError("No test samples found. Check dataset structure.")

    paths = [s for s, _ in samples]
    labels = [label for _, label in samples]

    le = joblib.load("label_encoder.pkl")
    labels_enc = le.transform(labels)
    n_classes = len(class_names)

    gen = FrameSequenceGenerator(paths, labels_enc, batch_size=8, seq_len=args.seq_len,
                                 img_size=(args.img_size, args.img_size), shuffle=False, grayscale=False, augment_fn=None, n_classes=n_classes)

    model = build_cnn_lstm(seq_len=args.seq_len, img_size=args.img_size, channels=3, n_classes=n_classes)
    model.load_weights(args.weights)

    y_true = []
    y_pred = []
    for X, y in gen:
        preds = model.predict(X)
        y_true.extend(np.argmax(y, axis=1).tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    print(classification_report(y_true, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
