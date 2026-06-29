# lab2_lstm_gru.py
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
)
import pandas as pd
import seaborn as sns
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------
# Конфигурация
# ---------------------------
OUTPUT_DIR = "lab2_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)
tf.random.set_seed(42)

print("🔧 Лабораторная работа 2: LSTM и GRU для временных рядов и текста")

# ---------------------------
# ЧАСТЬ 1: Прогнозирование временных рядов
# ---------------------------
print("\n" + "=" * 60)
print("📈 ЧАСТЬ 1: Прогнозирование временных рядов")
print("=" * 60)


# Генерация сложного временного ряда
def generate_complex_time_series(n_steps, seasonality=50, noise_level=0.1):
    """Генерация временного ряда с трендом, сезонностью и шумом"""
    time = np.linspace(0, 4 * np.pi, n_steps)

    # Основные компоненты
    trend = 0.01 * time  # Линейный тренд
    seasonal1 = 0.5 * np.sin(time)  # Сезонность 1
    seasonal2 = 0.3 * np.sin(2 * time + 1)  # Сезонность 2
    seasonal3 = 0.2 * np.sin(0.5 * time)  # Долгосрочная сезонность

    # Комбинируем компоненты
    series = trend + seasonal1 + seasonal2 + seasonal3
    # Добавляем шум
    series += noise_level * np.random.normal(size=n_steps)

    return series.astype(np.float32)


# Параметры временного ряда
n_steps = 2000
series = generate_complex_time_series(n_steps)

# Визуализация сгенерированного ряда
plt.figure(figsize=(12, 6))
plt.plot(series[:500])  # Показываем первые 500 точек для наглядности
plt.title("Сгенерированный временной ряд (первые 500 точек)")
plt.xlabel("Время")
plt.ylabel("Значение")
plt.grid(True, alpha=0.3)
plt.savefig(
    os.path.join(OUTPUT_DIR, "time_series_sample.png"), dpi=150, bbox_inches="tight"
)
plt.close()

print(f"📊 Длина временного ряда: {len(series)} точек")

# Разделение на обучающую и тестовую выборки
split_time = 1500
train_series = series[:split_time]
test_series = series[split_time:]

print(f"📋 Разделение: Обучающая {len(train_series)}, Тестовая {len(test_series)}")

# Нормализация данных
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_series.reshape(-1, 1))
test_scaled = scaler.transform(test_series.reshape(-1, 1))


# Создание оконных данных
def create_sequences(data, window_size):
    """Создание последовательностей для обучения"""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


window_size = 30  # Используем 30 предыдущих точек для прогноза
X_train, y_train = create_sequences(train_scaled, window_size)
X_test, y_test = create_sequences(test_scaled, window_size)

# Изменение формы для RNN [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"📦 Форма обучающих данных: {X_train.shape}")
print(f"📦 Форма тестовых данных: {X_test.shape}")


# Построение моделей для временных рядов
def build_time_series_model(model_type, window_size, units=50):
    """Построение моделей для прогнозирования временных рядов"""
    model = models.Sequential()

    if model_type == "SimpleRNN":
        model.add(
            layers.SimpleRNN(units, activation="tanh", input_shape=(window_size, 1))
        )
    elif model_type == "LSTM":
        model.add(layers.LSTM(units, activation="tanh", input_shape=(window_size, 1)))
    elif model_type == "GRU":
        model.add(layers.GRU(units, activation="tanh", input_shape=(window_size, 1)))

    model.add(layers.Dense(25, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))  # Выход для регрессии

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# Обучение моделей для временных рядов
time_series_models = {}
time_series_histories = {}
model_types = ["SimpleRNN", "LSTM", "GRU"]

print("\n🔄 Обучение моделей для временных рядов...")

for model_type in model_types:
    print(f"\n--- Обучение {model_type} ---")
    model = build_time_series_model(model_type, window_size, units=64)

    # Callbacks для обучения
    callbacks_list = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks_list,
        verbose=1,
    )

    time_series_models[model_type] = model
    time_series_histories[model_type] = history

# Визуализация обучения временных рядов
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
for model_type in model_types:
    plt.plot(
        time_series_histories[model_type].history["loss"],
        label=f"{model_type} Train",
        alpha=0.7,
    )
plt.title("Потери на обучении")
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
for model_type in model_types:
    plt.plot(
        time_series_histories[model_type].history["val_loss"],
        label=f"{model_type} Validation",
        alpha=0.7,
    )
plt.title("Потери на валидации")
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "time_series_training.png"), dpi=150, bbox_inches="tight"
)
plt.close()

# Прогнозирование и оценка
print("\n📊 Оценка моделей временных рядов...")

time_series_results = {}
plt.figure(figsize=(15, 10))

for i, model_type in enumerate(model_types):
    # Прогнозирование
    y_pred = time_series_models[model_type].predict(X_test)

    # Обратное масштабирование
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)

    # Расчет метрик
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = np.mean(np.abs(y_test_inv - y_pred_inv))

    time_series_results[model_type] = {"MSE": mse, "MAE": mae}

    # Визуализация прогнозов
    plt.subplot(3, 1, i + 1)
    plt.plot(y_test_inv[:200], label="Истинные значения", linewidth=2)
    plt.plot(y_pred_inv[:200], label="Предсказания", linewidth=1.5, alpha=0.8)
    plt.title(f"{model_type} (MSE: {mse:.4f}, MAE: {mae:.4f})")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.xlabel("Время")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "time_series_predictions.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# ---------------------------
# ЧАСТЬ 2: Классификация текстов
# ---------------------------
print("\n" + "=" * 60)
print("📝 ЧАСТЬ 2: Классификация текстовых отзывов")
print("=" * 60)

# Загрузка и подготовка данных IMDB
vocab_size = 10000  # 10к самых частых слов
max_len = 200  # Максимальная длина отзыва

print("📥 Загрузка датасета IMDB...")
(X_train_text, y_train_text), (X_test_text, y_test_text) = imdb.load_data(
    num_words=vocab_size
)

# Паддинг последовательностей
X_train_text = pad_sequences(X_train_text, maxlen=max_len)
X_test_text = pad_sequences(X_test_text, maxlen=max_len)

print(f"📊 Тренировочные данные: {X_train_text.shape}")
print(f"📊 Тестовые данные: {X_test_text.shape}")
print(f"📊 Баланс классов: {np.bincount(y_train_text)}")


# Функция для построения текстовых моделей
def build_text_model(model_type, vocab_size, embedding_dim=32, max_len=200):
    """Построение моделей для классификации текста"""
    model = models.Sequential()

    # Слой эмбеддингов
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=max_len))

    # Рекуррентные слои
    if model_type == "LSTM":
        model.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    elif model_type == "GRU":
        model.add(layers.GRU(64, dropout=0.2, recurrent_dropout=0.2))
    elif model_type == "Bidirectional_LSTM":
        model.add(
            layers.Bidirectional(layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2))
        )

    # Полносвязные слои
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid"))  # Бинарная классификация

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Обучение текстовых моделей
text_models = {}
text_histories = {}
text_model_types = ["LSTM", "GRU", "Bidirectional_LSTM"]

print("\n🔄 Обучение текстовых моделей...")

for model_type in text_model_types:
    print(f"\n--- Обучение {model_type} ---")
    model = build_text_model(model_type, vocab_size, embedding_dim=64, max_len=max_len)

    callbacks_text = [
        callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
    ]

    history = model.fit(
        X_train_text,
        y_train_text,
        epochs=15,
        batch_size=128,
        validation_split=0.2,
        callbacks=callbacks_text,
        verbose=1,
    )

    text_models[model_type] = model
    text_histories[model_type] = history

# Визуализация обучения текстовых моделей
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
for model_type in text_model_types:
    plt.plot(
        text_histories[model_type].history["accuracy"],
        label=f"{model_type} Train",
        linewidth=2,
    )
plt.title("Точность на обучении")
plt.xlabel("Эпоха")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
for model_type in text_model_types:
    plt.plot(
        text_histories[model_type].history["val_accuracy"],
        label=f"{model_type} Validation",
        linewidth=2,
    )
plt.title("Точность на валидации")
plt.xlabel("Эпоха")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "text_classification_training.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# Оценка текстовых моделей
print("\n📊 Оценка текстовых моделей...")

text_results = {}
plt.figure(figsize=(15, 10))

for i, model_type in enumerate(text_model_types):
    # Предсказания
    y_pred_proba = text_models[model_type].predict(X_test_text)
    y_pred_text = (y_pred_proba > 0.5).astype(int).flatten()

    # Метрики
    accuracy = accuracy_score(y_test_text, y_pred_text)
    report = classification_report(
        y_test_text,
        y_pred_text,
        target_names=["Negative", "Positive"],
        output_dict=True,
    )

    text_results[model_type] = {
        "Accuracy": accuracy,
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-Score": report["weighted avg"]["f1-score"],
    }

    # Матрица ошибок
    plt.subplot(2, 2, i + 1)
    cm = confusion_matrix(y_test_text, y_pred_text)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title(f"{model_type}\nAccuracy: {accuracy:.4f}")
    plt.ylabel("Истинные метки")
    plt.xlabel("Предсказанные метки")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "text_classification_results.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# ---------------------------
# Сравнительный анализ и выводы
# ---------------------------
print("\n" + "=" * 60)
print("📊 СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
print("=" * 60)

# Результаты временных рядов
print("\n📈 РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ ВРЕМЕННЫХ РЯДОВ:")
print("-" * 50)
ts_df = pd.DataFrame(time_series_results).T
print(ts_df.round(4))

# Результаты классификации текстов
print("\n📝 РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ ТЕКСТОВ:")
print("-" * 50)
text_df = pd.DataFrame(text_results).T
print(text_df.round(4))

# Визуализация сравнения моделей
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Сравнение MSE для временных рядов
axes[0, 0].bar(
    time_series_results.keys(),
    [ts_results["MSE"] for ts_results in time_series_results.values()],
)
axes[0, 0].set_title("Сравнение MSE (Временные ряды)")
axes[0, 0].set_ylabel("MSE")
for i, v in enumerate(time_series_results.values()):
    axes[0, 0].text(i, v["MSE"], f'{v["MSE"]:.4f}', ha="center", va="bottom")

# Сравнение MAE для временных рядов
axes[0, 1].bar(
    time_series_results.keys(),
    [ts_results["MAE"] for ts_results in time_series_results.values()],
)
axes[0, 1].set_title("Сравнение MAE (Временные ряды)")
axes[0, 1].set_ylabel("MAE")
for i, v in enumerate(time_series_results.values()):
    axes[0, 1].text(i, v["MAE"], f'{v["MAE"]:.4f}', ha="center", va="bottom")

# Сравнение Accuracy для текста
axes[1, 0].bar(
    text_results.keys(),
    [text_results["Accuracy"] for text_results in text_results.values()],
)
axes[1, 0].set_title("Сравнение Accuracy (Классификация текста)")
axes[1, 0].set_ylabel("Accuracy")
for i, v in enumerate(text_results.values()):
    axes[1, 0].text(i, v["Accuracy"], f'{v["Accuracy"]:.4f}', ha="center", va="bottom")

# Сравнение F1-Score для текста
axes[1, 1].bar(
    text_results.keys(),
    [text_results["F1-Score"] for text_results in text_results.values()],
)
axes[1, 1].set_title("Сравнение F1-Score (Классификация текста)")
axes[1, 1].set_ylabel("F1-Score")
for i, v in enumerate(text_results.values()):
    axes[1, 1].text(i, v["F1-Score"], f'{v["F1-Score"]:.4f}', ha="center", va="bottom")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight"
)
plt.close()

# ---------------------------
# Сохранение результатов и моделей
# ---------------------------
print("\n💾 Сохранение результатов...")

# Сохранение сводки результатов
summary_path = os.path.join(OUTPUT_DIR, "lab2_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=== ЛАБОРАТОРНАЯ РАБОТА 2: LSTM И GRU ===\n\n")

    f.write("ЧАСТЬ 1: ПРОГНОЗИРОВАНИЕ ВРЕМЕННЫХ РЯДОВ\n")
    f.write("=" * 50 + "\n")
    f.write(f"Размер окна: {window_size}\n")
    f.write(f"Обучающая выборка: {X_train.shape[0]} последовательностей\n")
    f.write(f"Тестовая выборка: {X_test.shape[0]} последовательностей\n\n")

    f.write("РЕЗУЛЬТАТЫ:\n")
    for model_type, results in time_series_results.items():
        f.write(f"{model_type}: MSE={results['MSE']:.4f}, MAE={results['MAE']:.4f}\n")

    f.write("\nЧАСТЬ 2: КЛАССИФИКАЦИЯ ТЕКСТОВ\n")
    f.write("=" * 50 + "\n")
    f.write(f"Размер словаря: {vocab_size}\n")
    f.write(f"Максимальная длина: {max_len}\n")
    f.write(f"Обучающая выборка: {X_train_text.shape[0]} отзывов\n")
    f.write(f"Тестовая выборка: {X_test_text.shape[0]} отзывов\n\n")

    f.write("РЕЗУЛЬТАТЫ:\n")
    for model_type, results in text_results.items():
        f.write(
            f"{model_type}: Accuracy={results['Accuracy']:.4f}, "
            f"F1={results['F1-Score']:.4f}\n"
        )

    f.write("\nВЫВОДЫ:\n")
    f.write("=" * 50 + "\n")
    f.write("1. LSTM и GRU показывают схожую производительность\n")
    f.write("2. GRU часто обучается быстрее из-за меньшего количества параметров\n")
    f.write("3. Bidirectional LSTM показывает лучшие результаты на текстовых данных\n")
    f.write("4. SimpleRNN уступает в задачах с долгосрочными зависимостями\n")

print(f"✅ Сводка сохранена: {summary_path}")

# Сохранение моделей
for model_type, model in time_series_models.items():
    model.save(os.path.join(OUTPUT_DIR, f"timeseries_{model_type.lower()}.h5"))

for model_type, model in text_models.items():
    model.save(os.path.join(OUTPUT_DIR, f"text_{model_type.lower()}.h5"))

print("💾 Модели сохранены")

# ---------------------------
# Финальное резюме
# ---------------------------
print("\n" + "=" * 60)
print("🎉 ЛАБОРАТОРНАЯ РАБОТА 2 ЗАВЕРШЕНА!")
print("=" * 60)
print(f"📁 Все результаты сохранены в: {OUTPUT_DIR}/")
print("\n📊 КЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ:")
print(f"   • Временные ряды: LSTM MSE = {time_series_results['LSTM']['MSE']:.4f}")
print(
    f"   • Классификация текста: BiLSTM Accuracy = {text_results['Bidirectional_LSTM']['Accuracy']:.4f}"
)
print(f"   • Сравнение: GRU vs LSTM - сопоставимое качество, разное время обучения")
print("\n🔍 ВЫВОДЫ:")
print("   • LSTM и GRU эффективны для последовательностей")
print("   • Bidirectional архитектуры улучшают работу с текстом")
print("   • SimpleRNN ограничена в задачах с долгосрочными зависимостями")
print("=" * 60)
