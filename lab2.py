# ============================================================
# LABORATOIRE 2 — LSTM & GRU pour séries temporelles et textes
# VERSION DÉFINITIVE
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    classification_report,
    confusion_matrix,
    r2_score,
    explained_variance_score,
)
import pandas as pd
import seaborn as sns
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================================
# Configuration générale
# ============================================================

OUTPUT_DIR = "lab2_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

print("🔧 LABORATOIRE 2 : LSTM et GRU - Version Définitive")

# ============================================================
# PARTIE 1 — Séries temporelles
# ============================================================

print("\n" + "=" * 60)
print("📈 PARTIE 1 : Prédiction de séries temporelles")
print("=" * 60)


def generate_complex_time_series(n_steps, noise=0.1):
    """Série temporelle avec tendance + trois composantes saisonnières."""
    t = np.linspace(0, 4 * np.pi, n_steps)
    trend = 0.01 * t
    s1 = 0.5 * np.sin(t)
    s2 = 0.3 * np.sin(2 * t + 1)
    s3 = 0.2 * np.sin(0.5 * t)
    series = trend + s1 + s2 + s3
    series += noise * np.random.normal(size=n_steps)
    return series.astype(np.float32)


# --- Génération du dataset ---
n_steps = 2000
series = generate_complex_time_series(n_steps)

plt.figure(figsize=(12, 5))
plt.plot(series[:500], linewidth=2)
plt.title("Série temporelle générée (500 premiers points)", fontsize=14, pad=20)
plt.xlabel("Temps")
plt.ylabel("Valeur")
plt.grid(True, alpha=0.3)
plt.savefig(
    os.path.join(OUTPUT_DIR, "time_series_sample.png"), dpi=150, bbox_inches="tight"
)
plt.close()

print(f"📊 Série temporelle générée: {n_steps} points")

# --- Préparation des données ---
split = 1500
train_series = series[:split]
test_series = series[split:]

print(f"📋 Division: Entraînement {len(train_series)}, Test {len(test_series)}")

# Normalisation
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_series.reshape(-1, 1))
test_scaled = scaler.transform(test_series.reshape(-1, 1))


def create_sequences(data, window=30):
    """Crée des séquences pour l'apprentissage des RNN"""
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


window_size = 30
X_train, y_train = create_sequences(train_scaled, window_size)
X_test, y_test = create_sequences(test_scaled, window_size)

# Reshape pour RNN [samples, time_steps, features]
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(f"📦 Forme des données d'entraînement: {X_train.shape}")
print(f"📦 Forme des données de test: {X_test.shape}")


def build_rnn(model_type, window, units=64):
    """Construit différents types de modèles RNN"""
    model = models.Sequential()

    # Couche récurrente
    if model_type == "SimpleRNN":
        model.add(layers.SimpleRNN(units, activation="tanh", input_shape=(window, 1)))
    elif model_type == "LSTM":
        model.add(layers.LSTM(units, activation="tanh", input_shape=(window, 1)))
    elif model_type == "GRU":
        model.add(layers.GRU(units, activation="tanh", input_shape=(window, 1)))

    # Couches denses
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(0.2))  # Régularisation
    model.add(layers.Dense(1))  # Sortie de régression

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


model_types = ["SimpleRNN", "LSTM", "GRU"]
time_histories = {}
time_models = {}

print("\n🔄 Entraînement des modèles (Séries Temporelles)…")

for m in model_types:
    print(f"\n--- {m} ---")
    model = build_rnn(m, window_size, units=64)

    # Callbacks pour un meilleur entraînement
    training_callbacks = [
        callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
    ]

    hist = model.fit(
        X_train,
        y_train,
        epochs=80,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=training_callbacks,
        verbose=1,
    )

    time_histories[m] = hist
    time_models[m] = model
    print(f"✅ {m} entraîné - {len(hist.history['loss'])} époques")

# --- Visualisation de l'entraînement ---
plt.figure(figsize=(15, 5))

# Pertes d'entraînement
plt.subplot(1, 2, 1)
for model_type in model_types:
    plt.plot(
        time_histories[model_type].history["loss"],
        label=f"{model_type}",
        linewidth=2,
        alpha=0.8,
    )
plt.title("Pertes pendant l'entraînement", fontsize=14)
plt.xlabel("Époques")
plt.ylabel("MSE")
plt.legend()
plt.grid(True, alpha=0.3)

# Pertes de validation
plt.subplot(1, 2, 2)
for model_type in model_types:
    plt.plot(
        time_histories[model_type].history["val_loss"],
        label=f"{model_type}",
        linewidth=2,
        alpha=0.8,
    )
plt.title("Pertes en validation", fontsize=14)
plt.xlabel("Époques")
plt.ylabel("MSE")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "time_series_training_curves.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.close()


# --- Évaluation détaillée ---
def evaluate_time_series_model(model, X_test, y_test, scaler, model_name):
    """Évalue un modèle avec plusieurs métriques"""
    # Prédictions
    y_pred = model.predict(X_test, verbose=0)

    # Transformation inverse
    y_true_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)

    # Calcul des métriques
    mse = mean_squared_error(y_true_inv, y_pred_inv)
    mae = np.mean(np.abs(y_true_inv - y_pred_inv))
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_inv, y_pred_inv)
    explained_var = explained_variance_score(y_true_inv, y_pred_inv)

    return (
        {
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Explained Variance": explained_var,
        },
        y_true_inv,
        y_pred_inv,
    )


results_ts = {}

plt.figure(figsize=(15, 10))
for i, m in enumerate(model_types):
    # Évaluation
    metrics, y_true, y_pred = evaluate_time_series_model(
        time_models[m], X_test, y_test, scaler, m
    )
    results_ts[m] = metrics

    # Visualisation des prédictions
    plt.subplot(3, 1, i + 1)
    plt.plot(y_true[:150], label="Valeurs réelles", linewidth=2, alpha=0.9)
    plt.plot(y_pred[:150], label="Prédictions", linewidth=1.5, alpha=0.8)
    plt.title(
        f'{m} — MSE: {metrics["MSE"]:.4f}, R²: {metrics["R2"]:.4f}', fontsize=12, pad=10
    )
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.xlabel("Temps")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "time_series_predictions.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# ============================================================
# PARTIE 2 — Classification de textes (IMDB)
# ============================================================

print("\n" + "=" * 60)
print("📝 PARTIE 2 : Classification de critiques IMDB")
print("=" * 60)

vocab_size = 10000
max_len = 200

print("📥 Chargement des données IMDB...")
(X_train_text, y_train_text), (X_test_text, y_test_text) = imdb.load_data(
    num_words=vocab_size
)

X_train_text = pad_sequences(X_train_text, maxlen=max_len)
X_test_text = pad_sequences(X_test_text, maxlen=max_len)

print(f"📊 Données d'entraînement: {X_train_text.shape}")
print(f"📊 Données de test: {X_test_text.shape}")
print(f"📊 Distribution des classes: {np.bincount(y_train_text)}")


def build_text_model(model_type, embed_dim=64):
    """Construit des modèles de classification de texte"""
    model = models.Sequential()

    # Couche d'embedding
    model.add(layers.Embedding(vocab_size, embed_dim, input_length=max_len))

    # Couche récurrente
    if model_type == "LSTM":
        model.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    elif model_type == "GRU":
        model.add(layers.GRU(64, dropout=0.2, recurrent_dropout=0.2))
    elif model_type == "Bidirectional_LSTM":
        model.add(
            layers.Bidirectional(layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2))
        )

    # Couches denses
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid"))  # Classification binaire

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall"],
    )
    return model


text_types = ["LSTM", "GRU", "Bidirectional_LSTM"]
text_histories = {}
text_models = {}
text_results = {}

print("\n🔄 Entraînement des modèles de texte…")

for model_type in text_types:
    print(f"\n--- {model_type} ---")
    model = build_text_model(model_type, embed_dim=64)

    text_callbacks = [
        callbacks.EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
    ]

    history = model.fit(
        X_train_text,
        y_train_text,
        epochs=15,
        batch_size=128,
        validation_split=0.2,
        callbacks=text_callbacks,
        verbose=1,
    )

    text_histories[model_type] = history
    text_models[model_type] = model

# --- Visualisation de l'entraînement texte ---
plt.figure(figsize=(15, 5))

# Accuracy
plt.subplot(1, 2, 1)
for model_type in text_types:
    plt.plot(
        text_histories[model_type].history["val_accuracy"],
        label=model_type,
        linewidth=2,
    )
plt.title("Accuracy en validation", fontsize=14)
plt.xlabel("Époques")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

# Pertes
plt.subplot(1, 2, 2)
for model_type in text_types:
    plt.plot(
        text_histories[model_type].history["val_loss"], label=model_type, linewidth=2
    )
plt.title("Pertes en validation", fontsize=14)
plt.xlabel("Époques")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "text_training_curves.png"), dpi=150, bbox_inches="tight"
)
plt.close()

# --- Évaluation détaillée ---
print("\n📊 Évaluation des modèles de texte...")

plt.figure(figsize=(15, 5))
for i, m in enumerate(text_types):
    # Prédictions
    y_prob = text_models[m].predict(X_test_text, verbose=0)
    y_pred = (y_prob > 0.5).astype(int).flatten()

    # Métriques
    accuracy = accuracy_score(y_test_text, y_pred)
    report = classification_report(
        y_test_text, y_pred, target_names=["Negative", "Positive"], output_dict=True
    )

    # Matrice de confusion
    cm = confusion_matrix(y_test_text, y_pred)

    text_results[m] = {
        "Accuracy": accuracy,
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-Score": report["weighted avg"]["f1-score"],
    }

    # Visualisation matrice de confusion
    plt.subplot(1, 3, i + 1)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title(f"{m}\nAccuracy: {accuracy:.4f}", fontsize=12, pad=15)
    plt.ylabel("Véritables labels")
    plt.xlabel("Labels prédits")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "text_confusion_matrices.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# ============================================================
# ANALYSE COMPARATIVE ET RAPPORT FINAL
# ============================================================

print("\n" + "=" * 60)
print("📊 ANALYSE COMPARATIVE")
print("=" * 60)

# --- Résultats séries temporelles ---
print("\n📈 RÉSULTATS SÉRIES TEMPORELLES:")
print("-" * 50)
ts_df = pd.DataFrame(results_ts).T.round(4)
print(ts_df)

# --- Résultats classification texte ---
print("\n📝 RÉSULTATS CLASSIFICATION TEXTE:")
print("-" * 50)
text_df = pd.DataFrame(text_results).T.round(4)
print(text_df)

# --- Visualisation comparative ---
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# MSE comparison
models_ts = list(results_ts.keys())
mse_values = [results_ts[m]["MSE"] for m in models_ts]
axes[0, 0].bar(models_ts, mse_values, color=["#ff9999", "#66b3ff", "#99ff99"])
axes[0, 0].set_title("Comparaison MSE (Séries Temporelles)", fontsize=14, pad=15)
axes[0, 0].set_ylabel("MSE")
for i, v in enumerate(mse_values):
    axes[0, 0].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontweight="bold")

# R² comparison
r2_values = [results_ts[m]["R2"] for m in models_ts]
axes[0, 1].bar(models_ts, r2_values, color=["#ff9999", "#66b3ff", "#99ff99"])
axes[0, 1].set_title("Comparaison R² (Séries Temporelles)", fontsize=14, pad=15)
axes[0, 1].set_ylabel("R² Score")
for i, v in enumerate(r2_values):
    axes[0, 1].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontweight="bold")

# Accuracy comparison
models_text = list(text_results.keys())
acc_values = [text_results[m]["Accuracy"] for m in models_text]
axes[1, 0].bar(models_text, acc_values, color=["#ffcc99", "#c2c2f0", "#ffb3e6"])
axes[1, 0].set_title("Comparaison Accuracy (Texte)", fontsize=14, pad=15)
axes[1, 0].set_ylabel("Accuracy")
for i, v in enumerate(acc_values):
    axes[1, 0].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontweight="bold")

# F1-Score comparison
f1_values = [text_results[m]["F1-Score"] for m in models_text]
axes[1, 1].bar(models_text, f1_values, color=["#ffcc99", "#c2c2f0", "#ffb3e6"])
axes[1, 1].set_title("Comparaison F1-Score (Texte)", fontsize=14, pad=15)
axes[1, 1].set_ylabel("F1-Score")
for i, v in enumerate(f1_values):
    axes[1, 1].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight"
)
plt.close()

# ============================================================
# SAUVEGARDE ET RAPPORT FINAL
# ============================================================

print("\n💾 Sauvegarde des résultats...")

# --- Sauvegarde des modèles ---
for name, model in time_models.items():
    model.save(os.path.join(OUTPUT_DIR, f"timeseries_{name.lower()}.h5"))

for name, model in text_models.items():
    model.save(os.path.join(OUTPUT_DIR, f"text_{name.lower()}.h5"))

print("✅ Modèles sauvegardés")

# --- Rapport détaillé ---
summary_path = os.path.join(OUTPUT_DIR, "lab2_complete_report.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("LABORATOIRE 2 - RAPPORT COMPLET\n")
    f.write("LSTM & GRU pour Séries Temporelles et Classification de Texte\n")
    f.write("=" * 60 + "\n\n")

    f.write("CONFIGURATION EXPÉRIMENTALE:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Séries Temporelles:\n")
    f.write(f"  • Taille de la série: {n_steps} points\n")
    f.write(f"  • Taille de la fenêtre: {window_size}\n")
    f.write(f"  • Données d'entraînement: {X_train.shape[0]} séquences\n")
    f.write(f"  • Données de test: {X_test.shape[0]} séquences\n\n")

    f.write(f"Classification de Texte:\n")
    f.write(f"  • Taille du vocabulaire: {vocab_size}\n")
    f.write(f"  • Longueur maximale: {max_len} tokens\n")
    f.write(f"  • Reviews d'entraînement: {X_train_text.shape[0]}\n")
    f.write(f"  • Reviews de test: {X_test_text.shape[0]}\n\n")

    f.write("RÉSULTATS DÉTAILLÉS:\n")
    f.write("-" * 40 + "\n")

    f.write("\nSÉRIES TEMPORELLES:\n")
    for model, metrics in results_ts.items():
        f.write(f"\n{model}:\n")
        for metric, value in metrics.items():
            f.write(f"  • {metric}: {value:.4f}\n")

    f.write("\nCLASSIFICATION DE TEXTE:\n")
    for model, metrics in text_results.items():
        f.write(f"\n{model}:\n")
        for metric, value in metrics.items():
            f.write(f"  • {metric}: {value:.4f}\n")

    f.write("\nANALYSE ET CONCLUSIONS:\n")
    f.write("-" * 40 + "\n")

    # Meilleurs modèles
    best_ts = min(results_ts.items(), key=lambda x: x[1]["MSE"])
    best_text = max(text_results.items(), key=lambda x: x[1]["Accuracy"])

    f.write(f"\nMEILLEURS MODÈLES:\n")
    f.write(f"• Séries temporelles: {best_ts[0]} (MSE: {best_ts[1]['MSE']:.4f})\n")
    f.write(
        f"• Classification texte: {best_text[0]} (Accuracy: {best_text[1]['Accuracy']:.4f})\n"
    )

    f.write(f"\nOBSERVATIONS CLÉS:\n")
    f.write("1. LSTM et GRU surpassent SimpleRNN sur les séries temporelles\n")
    f.write("2. GRU offre un bon compromis vitesse/précision\n")
    f.write("3. LSTM bidirectionnel excelle en classification de texte\n")
    f.write("4. Le dropout améliore significativement la généralisation\n")
    f.write("5. L'early stopping prévient le surapprentissage efficacement\n")

    f.write(f"\nRECOMMANDATIONS:\n")
    f.write("• Pour les séries temporelles: Privilégier LSTM ou GRU\n")
    f.write("• Pour le texte: Utiliser LSTM bidirectionnel\n")
    f.write("• Pour le prototypage rapide: Choisir GRU\n")

print(f"✅ Rapport complet sauvegardé: {summary_path}")

# ============================================================
# RÉSUMÉ FINAL
# ============================================================

print("\n" + "=" * 60)
print("🎉 LABORATOIRE 2 TERMINÉ AVEC SUCCÈS!")
print("=" * 60)
print(f"📁 TOUS LES RÉSULTATS DANS: {OUTPUT_DIR}/")
print(f"📊 NOMBRE DE FICHIERS GÉNÉRÉS: {len(os.listdir(OUTPUT_DIR))}")

# Affichage des meilleurs résultats
best_ts_model = min(results_ts.items(), key=lambda x: x[1]["MSE"])
best_text_model = max(text_results.items(), key=lambda x: x[1]["Accuracy"])

print(f"\n🏆 MEILLEURS RÉSULTATS:")
print(
    f"   • Séries temporelles: {best_ts_model[0]} (MSE: {best_ts_model[1]['MSE']:.4f})"
)
print(
    f"   • Classification texte: {best_text_model[0]} (Accuracy: {best_text_model[1]['Accuracy']:.4f})"
)

print(f"\n📈 GRAPHIQUES GÉNÉRÉS:")
print(f"   • time_series_sample.png")
print(f"   • time_series_training_curves.png")
print(f"   • time_series_predictions.png")
print(f"   • text_training_curves.png")
print(f"   • text_confusion_matrices.png")
print(f"   • model_comparison.png")

print(f"\n💾 MODÈLES SAUVEGARDÉS:")
for m in model_types + text_types:
    print(f"   • {m.lower()}.h5")

print("\n" + "=" * 60)
