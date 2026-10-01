# -*- coding: utf-8 -*-

"""
Written by Kambana on February 2nd 2026
"""

# IMPORTATION
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
)

# LOADING DATA
file_path = 'Ton_IoT_Backdoor_dataset_cleaned_feature.csv'
df = pd.read_csv(file_path)

y = df['label'].iloc[1:].values.astype(np.int32)
X = df.drop(columns=['label']).iloc[1:].values.astype(np.float32)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

input_dim = X_train.shape[1]

# BUILD A VALID SUPPORT SET FOR RELATION MODEL DEPLOYEMENT (16 samples)

SUPPORT_SIZE = 16
MAX_POS_RATIO = 0.25
MAX_POS = int(SUPPORT_SIZE * MAX_POS_RATIO) 
MIN_POS = 1

# Indices by class
idx_pos = np.where(y_test == 1)[0]
idx_neg = np.where(y_test == 0)[0]

# Checkings

if len(idx_pos) < MIN_POS:
    raise ValueError(
        f"Not enough positive samples in test set "
        f"(found {len(idx_pos)}, required at least {MIN_POS})"
    )

if len(idx_neg) == 0:
    raise ValueError(
        "No negative samples available : support set would be invalid"
    )

# Select positives samples

n_pos = min(len(idx_pos), MAX_POS)
n_pos = max(n_pos, MIN_POS)

selected_pos_idx = np.random.choice(
    idx_pos,
    size=n_pos,
    replace=False
)

# Select neagatives samples

n_neg = SUPPORT_SIZE - n_pos

if len(idx_neg) < n_neg:
    raise ValueError(
        f"Not enough negative samples to fill support set "
        f"(needed {n_neg}, found {len(idx_neg)})"
    )

selected_neg_idx = np.random.choice(
    idx_neg,
    size=n_neg,
    replace=False
)

# Build support set for Relation Network Deployement 

selected_idx = np.concatenate([selected_pos_idx, selected_neg_idx])
np.random.shuffle(selected_idx)

# samples form test_set
support = X_test[selected_idx]
support_labels = y_test[selected_idx]

# logging

print("Support set shape:", support.shape)                 
print("Support labels shape:", support_labels.shape)     
print("Unique support labels:", np.unique(support_labels))
print(
    "Positive count:",
    np.sum(support_labels == 1),
    "| Negative count:",
    np.sum(support_labels == 0)
)

# Save models
np.save("support.npy", support)
np.save("support_labels.npy", support_labels)

print("Support set saved to support.npy and support_labels.npy")


# SAMPLING EPISODE FOR EACH EPOCH
def sample_class(X, y, label, n):
    Xc = X[y == label]
    n_available = len(Xc)

    if n <= n_available:
        idx = np.random.choice(n_available, n, replace=False)
    else:
        idx_full = np.random.permutation(n_available)
        remaining = n - n_available
        idx_extra = np.random.choice(n_available, remaining, replace=True)
        idx = np.concatenate([idx_full, idx_extra])

    return Xc[idx]

def make_episode(X, y, K=16, Q=128):
    # Support set
    supp_pos = sample_class(X, y, 1, K)
    supp_neg = sample_class(X, y, 0, K)

    support = np.vstack([supp_pos, supp_neg])
    support_labels = np.array([1]*K + [0]*K)

    perm = np.random.permutation(len(support))
    support, support_labels = support[perm], support_labels[perm]

    # Query set
    q_pos = Q // 2
    q_neg = Q - q_pos

    query_pos = sample_class(X, y, 1, q_pos)
    query_neg = sample_class(X, y, 0, q_neg)

    query = np.vstack([query_pos, query_neg])
    query_labels = np.array([1]*q_pos + [0]*q_neg)

    perm = np.random.permutation(len(query))
    query, query_labels = query[perm], query_labels[perm]

    return support, support_labels, query, query_labels
# BUILDING MODELS
def build_embedder(input_dim, emb_dim=64):
    reg = tf.keras.regularizers.l2(1e-4)

    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.LayerNormalization()(inputs)

    attn = tf.keras.layers.Dense(input_dim, activation="softmax")(x)
    x = tf.keras.layers.Multiply()([x, attn])

    x = tf.keras.layers.Dense(128, activation="gelu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Residual block
    residual = x
    x = tf.keras.layers.Dense(128, activation="gelu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=reg)(x)
    x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.Activation("gelu")(x)

    x = tf.keras.layers.Dense(64, activation="gelu")(x)

    # Dropout improves generalization and stabilizes embeddings
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(emb_dim)(x)

    # Normalize embeddings for distance-based comparison
    outputs = tf.keras.layers.UnitNormalization(axis=1)(x)

    return tf.keras.Model(inputs, outputs)

def build_relation_module(emb_dim):
    inputs = tf.keras.Input(shape=(None, 2 * emb_dim))

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.LayerNormalization()
    )(inputs)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(128, activation="gelu")
    )(x)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dropout(0.1)
    )(x)

    # Residual block
    residual = x
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(128, activation="gelu")
    )(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(128)
    )(x)
    x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.Activation("gelu")(x)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(32, activation="gelu")
    )(x)

    # Activation with sigmoid for prediction
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1, activation="sigmoid")
    )(x)

    return tf.keras.Model(inputs, outputs)
# TRAIN THE MODEL
embedder = build_embedder(input_dim)
relation = build_relation_module(64)

n_epochs = 10
episodes_per_epoch = 7108

# Cosine decay learning rate for stability
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=42648,#We note that htis params stabilize the training
    alpha=1e-4
)

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=1e-4,
    clipnorm=1.0
)

# Train step
@tf.function
def train_step(support, slabels, query, qlabels):

    with tf.GradientTape() as tape:

        supp_emb = embedder(support, training=True)
        query_emb = embedder(query, training=True)

        supp_exp = supp_emb[None, :, :]
        query_exp = query_emb[:, None, :]

        diff = tf.abs(supp_exp - query_exp)
        prod = supp_exp * query_exp

        pair = tf.concat([diff, prod], axis=-1)

        relations = relation(pair, training=True)
        relations = tf.squeeze(relations, axis=-1)

        # Aggregate relation scores per query
        class_scores = tf.reduce_mean(relations, axis=1)

        # Binary classification loss at query level
        loss = tf.keras.losses.binary_crossentropy(
            qlabels, class_scores
        )

        loss = tf.reduce_mean(loss)

    variables = embedder.trainable_variables + relation.trainable_variables
    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return loss

# Train Loop
all_epoch_losses = []
all_epoch_lrs = []

for epoch in range(n_epochs):

    losses = []

    for _ in range(episodes_per_epoch):
        support, slabels, query, qlabels = make_episode(X_train, y_train)

        loss = train_step(
            tf.convert_to_tensor(support, tf.float32),
            tf.convert_to_tensor(slabels, tf.float32),
            tf.convert_to_tensor(query, tf.float32),
            tf.convert_to_tensor(qlabels, tf.float32)
        )

        losses.append(loss.numpy())

    current_lr = optimizer.learning_rate.numpy()
    mean_loss = np.mean(losses)
    all_epoch_losses.append(mean_loss)

    all_epoch_lrs.append(current_lr)

    print(f"Epoch {epoch+1}/{n_epochs} | Loss: {mean_loss:.4f} | LR: {current_lr:.7f}")

# PLOT LOSS
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(range(1, n_epochs + 1), all_epoch_losses, marker="o", label="Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(range(1, n_epochs + 1), all_epoch_lrs, linestyle="--", label="LR")
ax2.set_ylabel("Learning Rate")

fig.legend(loc="upper right")
plt.title("Learning Curve with LR Cosine Decay")
plt.savefig("learning_curve_with_lr.png", dpi=400, bbox_inches="tight")
plt.close()

# SAVE MODELS
embedder.save("embedder.h5")
relation.save("relation.h5")

print("Models saved.")
# FEW- SHOT LEARNING EVALUATION : RELATION NETWORKS

# COMPUTE CI95

def compute_ci95(data, n_boot=1000):
    data = np.array(data)
    means = []

    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))

    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)

    mean = np.mean(data)
    ci95 = (upper - lower) / 2  # take +/- CI95

    return mean, ci95, lower, upper

# EPISODE EVALUATION

def evaluate_episode(support, slabels, query, qlabels):

    supp_emb = embedder(support, training=False)
    query_emb = embedder(query, training=False)

    supp_exp = supp_emb[None, :, :]
    query_exp = query_emb[:, None, :]

    diff = tf.abs(supp_exp - query_exp)
    prod = supp_exp * query_exp

    pair = tf.concat([diff, prod], axis=-1)

    relations = relation(pair, training=False)
    relations = tf.squeeze(relations, axis=-1)

    scores = tf.reduce_mean(relations, axis=1).numpy()

    return scores, qlabels

# GLOBAL META-EVALUATION

n_test_episodes = 1000
threshold = 0.50

all_scores = []
all_labels = []

acc_list = []
prec_list = []
rec_list = []
f1_list = []

for _ in range(n_test_episodes):

    support, slabels, query, qlabels = make_episode(X_test, y_test)

    scores, labels = evaluate_episode(
        tf.convert_to_tensor(support, tf.float32),
        tf.convert_to_tensor(slabels, tf.float32),
        tf.convert_to_tensor(query, tf.float32),
        tf.convert_to_tensor(qlabels, tf.float32)
    )

    preds = (scores >= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='macro', zero_division=0)
    rec = recall_score(labels, preds, average='macro', zero_division=0)
    f1_val = f1_score(labels, preds, average='macro', zero_division=0)

    acc_list.append(acc)
    prec_list.append(prec)
    rec_list.append(rec)
    f1_list.append(f1_val)

    all_scores.extend(scores)
    all_labels.extend(labels)

acc_mean, acc_ci95, acc_low, acc_high = compute_ci95(acc_list)
prec_mean, prec_ci95, prec_low, prec_high = compute_ci95(prec_list)
rec_mean, rec_ci95, rec_low, rec_high = compute_ci95(rec_list)
f1_mean, f1_ci95, f1_low, f1_high = compute_ci95(f1_list)

# Confusion matrix

all_scores = np.array(all_scores)
all_labels = np.array(all_labels)
preds_global = (all_scores >= threshold).astype(int)

cm = confusion_matrix(all_labels, preds_global)

print("\n  CONFUSION MATRIX ")
print(cm)

# Roc Curve and AUC

fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
roc_auc = auc(fpr, tpr)

print("\n   ROC / AUC  ")
print(f"AUC : {roc_auc:.6f}")

plt.figure(figsize=(6, 5))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Meta Learning")
plt.legend(loc="lower right")

plt.grid(True)

plt.savefig("roc_curve.png", dpi=400, bbox_inches="tight")
plt.close()

print("ROC curve saved as roc_curve.png")

# Plot Confusion matrix

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Normal (0)", "Attack (1)"],
    yticklabels=["Normal (0)", "Attack (1)"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Meta Learning")

plt.savefig("confusion_matrix.png", dpi=400, bbox_inches="tight")
plt.close()

print("Confusion matrix saved as confusion_matrix.png")

# Print result

print("\n     RELATION NETWORKS PERFORMANCE (MACRO)    ")
print(f"Fixed Threshold : {threshold:.2f}")

print(f"Mean Accuracy  : {acc_mean*100:.6f}% +/- CI95 {acc_ci95*100:.6f}")
print(f"Mean Precision : {prec_mean*100:.6f}% +/- CI95 {prec_ci95*100:.6f}")
print(f"Mean Recall    : {rec_mean*100:.6f}% +/- CI95 {rec_ci95*100:.6f}")
print(f"Mean F1 Score  : {f1_mean*100:.6f}% +/- CI95 {f1_ci95*100:.6f}")

# Plot metrics (CI95)

metrics_names = ["Accuracy", "Precision", "Recall", "F1"]
metrics_values = [acc_mean, prec_mean, rec_mean, f1_mean]
metrics_ci95 = [acc_ci95, prec_ci95, rec_ci95, f1_ci95]

plt.figure(figsize=(8, 5))
plt.bar(metrics_names, metrics_values, yerr=metrics_ci95, capsize=5)

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Relation Networks Performance Metrics (Mean +/- CI95)")

for i, v in enumerate(metrics_values):
    plt.text(i, v + 0.02, f"{v*100:.1f}%", ha='center')

plt.grid(True, axis='y')

plt.savefig("relation_networks_metrics.png", dpi=400, bbox_inches="tight")
plt.close()

print("\nGraph saved as relation_networks_metrics.png")
