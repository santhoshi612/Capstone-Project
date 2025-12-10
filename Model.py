# multimodal_fusion_clean.py
# Multimodal stress/depression detection: Facial + Posture + Gait
# Clean, robust, optimized script

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


base_path = "/content/drive/MyDrive/Capstone"

face_train_dir = os.path.join(base_path, "facial", "images", "train")
face_val_dir   = os.path.join(base_path, "facial", "images", "validation")

posture_train_dir = os.path.join(base_path, "posture", "data")
posture_val_dir   = os.path.join(base_path, "posture", "data")

gait_train_dir = os.path.join(base_path, "gait_analysis", "Train")
gait_val_dir   = os.path.join(base_path, "gait_analysis", "Test")

# -----------------------------
# Gait loader: compute per-file numeric stats and pad to fixed length
# -----------------------------
def load_gait_data_pad(base_dir, pad_to=None):
    """
    Recursively find CSVs under base_dir. For each CSV:
     - select numeric columns,
     - compute [mean, std, min, max] per numeric column,
     - flatten to 1D feature vector.
    To accommodate files with different numbers of numeric columns, pads
    short vectors with zeros up to pad_to (or the max length found).
    Auto-labeling based on filename keywords: 'stress', 'depress'/'depression', 'normal'.
    Returns X (n_samples, feat_len), y (list of labels), and feat_len used.
    """
    csv_files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        return np.empty((0, )), np.empty((0, )), 0

    feats = []
    labels = []
    lengths = []

    for f in csv_files:
        fname = os.path.basename(f).lower()
        # label rules: customize if needed
        if "stress" in fname or "ataxia" in fname:
            label = "stress"
        elif "depress" in fname or "depression" in fname:
            label = "depression"
        elif "normal" in fname or "healthy" in fname:
            label = "normal"
        else:
            # skip unlabeled files
            continue

        try:
            df = pd.read_csv(f)
            df_num = df.select_dtypes(include=[np.number])
            if df_num.shape[1] == 0:
                # skip files with no numeric columns
                continue

            # fill NaNs with column mean
            df_num = df_num.fillna(df_num.mean())

            # compute stats per column: mean, std, min, max
            vec = np.concatenate([
                df_num.mean().values,
                df_num.std().values,
                df_num.min().values,
                df_num.max().values
            ])
            feats.append(vec.astype(np.float32))
            labels.append(label)
            lengths.append(len(vec))
        except Exception as e:
            # skip bad files, but print a short message
            print(f"Skipped file {f}: {e}")

    if len(feats) == 0:
        return np.empty((0, )), np.empty((0, )), 0

    # determine pad length
    max_len = max(lengths)
    if pad_to is None:
        pad_to = max_len

    # pad/truncate vectors to pad_to
    X = np.zeros((len(feats), pad_to), dtype=np.float32)
    for i, v in enumerate(feats):
        L = min(len(v), pad_to)
        X[i, :L] = v[:L]

    y = np.array(labels, dtype=str)
    return X, y, pad_to

# -----------------------------
# Load gait datasets
# -----------------------------
X_gait_train, y_gait_train, gait_feat_len = load_gait_data_pad(gait_train_dir)
X_gait_val,   y_gait_val,   gait_feat_len2 = load_gait_data_pad(gait_val_dir, pad_to=gait_feat_len)

# ensure train & val use same feature length (pad val if needed)
if X_gait_train.size == 0 and X_gait_val.size == 0:
    raise RuntimeError("No gait CSVs found. Check gait directory and filenames.")
if gait_feat_len2 > gait_feat_len:
    # pad training to val len
    pad_new = gait_feat_len2
    X_train_padded = np.zeros((X_gait_train.shape[0], pad_new), dtype=np.float32)
    X_train_padded[:, :X_gait_train.shape[1]] = X_gait_train
    X_gait_train = X_train_padded
    gait_feat_len = pad_new
elif gait_feat_len2 < gait_feat_len:
    # pad val
    pad_new = gait_feat_len
    X_val_padded = np.zeros((X_gait_val.shape[0], pad_new), dtype=np.float32)
    X_val_padded[:, :X_gait_val.shape[1]] = X_gait_val
    X_gait_val = X_val_padded

if X_gait_train.size == 0 or X_gait_val.size == 0:
    raise RuntimeError("Gait loader found insufficient labeled files for train or validation.")

print("Gait feature vector length:", gait_feat_len)
print("Gait train samples:", X_gait_train.shape[0], "val samples:", X_gait_val.shape[0])

# -----------------------------
# Encode labels and scale gait features
# -----------------------------
le = LabelEncoder()
y_all = np.concatenate([y_gait_train, y_gait_val])
le.fit(y_all)  # ensures consistent encoding
y_train_enc = le.transform(y_gait_train)
y_val_enc   = le.transform(y_gait_val)

scaler = StandardScaler()
X_gait_train = scaler.fit_transform(X_gait_train)
X_gait_val   = scaler.transform(X_gait_val)

print("Classes:", le.classes_)

# -----------------------------
# Image generators (binary/multiclass depends on folders)
# -----------------------------
img_h, img_w = 128, 128
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=12,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.08,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)


def _choose_class_mode(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image folder not found: {path}")
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if len(subdirs) == 2:
        return "binary"
    return "categorical"

face_class_mode = _choose_class_mode(face_train_dir)
posture_class_mode = _choose_class_mode(posture_train_dir)

train_face_gen = train_datagen.flow_from_directory(
    face_train_dir, target_size=(img_h, img_w),
    batch_size=batch_size, class_mode=face_class_mode, shuffle=True
)
val_face_gen = val_datagen.flow_from_directory(
    face_val_dir, target_size=(img_h, img_w),
    batch_size=batch_size, class_mode=face_class_mode, shuffle=False
)

train_posture_gen = train_datagen.flow_from_directory(
    posture_train_dir, target_size=(img_h, img_w),
    batch_size=batch_size, class_mode=posture_class_mode, shuffle=True
)
val_posture_gen = val_datagen.flow_from_directory(
    posture_val_dir, target_size=(img_h, img_w),
    batch_size=batch_size, class_mode=posture_class_mode, shuffle=False
)

# -----------------------------
# CNN branch builder (functional)
# -----------------------------
def build_cnn_branch(name, input_shape=(img_h, img_w, 3)):
    inp = keras.Input(shape=input_shape, name=f"{name}_input")
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    return keras.Model(inputs=inp, outputs=x, name=f"{name}_branch")

face_branch = build_cnn_branch("face")
posture_branch = build_cnn_branch("posture")

# -----------------------------
# Gait MLP branch
# -----------------------------
gait_input = keras.Input(shape=(gait_feat_len,), name="gait_input")
xg = layers.Dense(128, activation="relu")(gait_input)
xg = layers.Dropout(0.3)(xg)
xg = layers.Dense(64, activation="relu")(xg)
gait_branch = keras.Model(inputs=gait_input, outputs=xg, name="gait_branch")

# -----------------------------
# Fusion & final classifier
# -----------------------------
combined = layers.concatenate([face_branch.output, posture_branch.output, gait_branch.output])
x = layers.Dense(128, activation="relu")(combined)
x = layers.Dropout(0.4)(x)

binary_problem = (face_class_mode == "binary" and posture_class_mode == "binary" and len(le.classes_) == 2)

if binary_problem:
    final_out = layers.Dense(1, activation="sigmoid", name="output")
    loss = "binary_crossentropy"
else:
    final_out = layers.Dense(len(le.classes_), activation="softmax", name="output")
    loss = "sparse_categorical_crossentropy"

output = final_out(x)
model = keras.Model(inputs=[face_branch.input, posture_branch.input, gait_branch.input], outputs=output, name="fusion_model")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    loss=loss,
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Multimodal generator that yields aligned batches
# - yields ((face, posture, gait), labels)
# - gait labels must be in same encoding space as image labels; we will use gait labels as "source of truth"
# -----------------------------
def multimodal_generator(face_gen, posture_gen, gait_X, gait_y_enc):
    if gait_X.shape[0] == 0:
        raise RuntimeError("No gait samples provided to generator.")
    ptr = 0
    n_gait = gait_X.shape[0]
    while True:
        face_batch, _ = next(face_gen)
        posture_batch, _ = next(posture_gen)
        b = min(len(face_batch), len(posture_batch))
        # sequential gait sampling (deterministic looping)
        idx = np.arange(ptr, ptr + b) % n_gait
        ptr = (ptr + b) % n_gait
        gait_batch = gait_X[idx]
        labels_batch = gait_y_enc[idx]
        # if binary_problem and model expects (n,) labels, ensure dtype
        yield (face_batch[:b], posture_batch[:b], gait_batch.astype(np.float32)), labels_batch.astype(np.int32)

# -----------------------------
# Build tf.data Datasets with signatures
# -----------------------------
spec = (
    (
        tf.TensorSpec(shape=(None, img_h, img_w, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, img_h, img_w, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, gait_feat_len), dtype=tf.float32),
    ),
    tf.TensorSpec(shape=(None,), dtype=tf.int32)
)

train_ds = tf.data.Dataset.from_generator(
    lambda: multimodal_generator(train_face_gen, train_posture_gen, X_gait_train, y_train_enc),
    output_signature=spec
)
val_ds = tf.data.Dataset.from_generator(
    lambda: multimodal_generator(val_face_gen, val_posture_gen, X_gait_val, y_val_enc),
    output_signature=spec
)

# -----------------------------
# Training settings & callbacks
# -----------------------------
steps_per_epoch = min(len(train_face_gen), len(train_posture_gen))
validation_steps = min(len(val_face_gen), len(val_posture_gen))

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_fusion_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1)
]

EPOCHS = 10  # keep small for faster runs; increase as needed

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# Evaluation: iterate deterministic batches from val generators and gait val
# -----------------------------
print("Evaluating saved best model...")
model.load_weights("best_fusion_model.keras")

# Collect predictions over val_steps
y_trues = []
y_preds = []

# deterministic gait pointer
gptr = 0
n_val_g = X_gait_val.shape[0]

for step in range(validation_steps):
    face_batch, _ = next(val_face_gen)
    posture_batch, _ = next(val_posture_gen)
    b = min(len(face_batch), len(posture_batch))
    # gait indices
    idx = np.arange(gptr, gptr + b) % n_val_g
    gptr = (gptr + b) % n_val_g
    gait_batch = X_gait_val[idx]
    true_labels = y_val_enc[idx]

    preds = model.predict((face_batch[:b], posture_batch[:b], gait_batch), verbose=0)
    if binary_problem:
        preds_bin = (preds.flatten() > 0.5).astype(int)
    else:
        preds_bin = np.argmax(preds, axis=1)

    y_trues.extend(true_labels.tolist())
    y_preds.extend(preds_bin.tolist())

y_trues = np.array(y_trues, dtype=int)
y_preds = np.array(y_preds, dtype=int)

print("Accuracy:", accuracy_score(y_trues, y_preds))
print("Confusion matrix:\n", confusion_matrix(y_trues, y_preds))
print("Classification report:\n", classification_report(y_trues, y_preds, target_names=le.classes_))

# -----------------------------
# Plot training curves (with simple smoothing)
# -----------------------------
def smooth(x, window=3):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode='valid')

acc = history.history.get("accuracy", [])
val_acc = history.history.get("val_accuracy", [])
loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(smooth(acc, window=3), label="Train (smoothed)")
plt.plot(smooth(val_acc, window=3), label="Val (smoothed)")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(smooth(loss, window=3), label="Train (smoothed)")
plt.plot(smooth(val_loss, window=3), label="Val (smoothed)")
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.show()
