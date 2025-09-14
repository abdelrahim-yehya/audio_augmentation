# audio_augmentation

**LEARNING AUGMENTATION STRATEGIES FROM DATA**

This repository provides tools and code to experiment with audio data augmentation and evaluate its impact on audio classification models. The main workflow is implemented in a Jupyter notebook that demonstrates preprocessing, augmentation, and model training using a custom audio dataset.

---

## Features

- **Audio Preprocessing:** Turns audio files (.wav) into spectrograms suitable for neural network input.
- **Data Augmentation:** Uses image-based augmentation (rotation, contrast adjustment, inversion) on spectrograms to increase training data variety.
- **Multi-Class Audio Classification:** Train a convolutional neural network (CNN) to classify audio events (e.g., guitar, applause, knock, dog, violin).
- **Model Evaluation:** Visualizes training metrics (loss, precision, recall) and tests model performance on new data.

---

## Getting Started

### 1. Requirements

Install dependencies using pip:

```bash
pip install numpy pandas matplotlib seaborn scipy librosa tensorflow pillow scikit-learn
```

### 2. Dataset Structure

Organize your audio data in the following structure:

```
Data/
└── AudioEventDataset/
    ├── train/
    │   ├── acoustic_guitar/
    │   ├── applause/
    │   ├── knock/
    │   ├── dog_barking/
    │   └── violin/
    └── test/
        └── tracks/
            ├── acoustic_guitar_0.wav
            ├── applause_1.wav
            └── ...
```

---

## Usage

Open and run the notebook [`Modèle avec les données augmentées.ipynb`](Mod%C3%A8le%20avec%20les%20donn%C3%A9es%20augment%C3%A9es.ipynb) for a full step-by-step workflow.

### Key Code Snippets

#### Audio Preprocessing

```python
def preprocess(file_path): 
    wav , sr = librosa.load(file_path, sr=16000)
    wav = wav[:50000]
    zero_padding = tf.zeros([50000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram
```

#### Data Augmentation (Spectrogram as Image)

```python
def autoAugment(x, n=1):
    img = Image.fromarray(np.uint8(tf.transpose(x)[0] * 255), 'L')
    for i in range(n):
        angle = np.random.randint(0, 360)
        factor = np.random.randint(-2, 2)
        enhancer = ImageEnhance.Contrast(img)
        img_aug = img.rotate(angle)
        img_aug = enhancer.enhance(factor)
        pix = np.array(img_aug, dtype='float32') / 255
        pix = tf.transpose(pix)
        pix = tf.expand_dims(pix, axis=2)
    return pix
```

#### Model Definition & Training

```python
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(1553, 257, 1)),
    MaxPooling2D((2,2)),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()],
    optimizer='adam'
)

model.fit(X_train, y_train, batch_size=4, epochs=10, validation_data=(X_test, y_test))
```

#### Evaluation

```python
test_accuracy = model.evaluate(x_test_data, y_test_data)
print("Test metrics:", test_accuracy)
```

For the full workflow, see the notebook: [`Modèle avec les données augmentées.ipynb`](Mod%C3%A8le%20avec%20les%20donn%C3%A9es%20augment%C3%A9es.ipynb)
```

---

## Visualizations

The notebook includes code to plot loss and precision curves to help analyze the value of augmentation:

```python
plt.plot(hist.history['loss'], 'r')
plt.plot(hist.history['val_loss'], 'b')
plt.title('Loss')
plt.show()
```

---

## Citation & References

If you use this code, please cite the repository and/or reference the project in your work.

---

## Contact

For questions or collaboration, open an issue or contact [abdelrahim-yehya](https://github.com/abdelrahim-yehya).

---

## Quick Start

To test the efficiency of audio data augmentation, simply run the attached notebook and follow the steps to preprocess your data, create augmented samples, train the CNN, and evaluate results.

---

**Happy experimenting with audio data augmentation!**

---

Let me know if you want to add a project logo, more details on data, or example results!
