# Music Genre Classifer Model

---

## Overview

This CNN model classifies songs into one of ten genres based on the GZTAN dataset. The audio data is converted to a spectrogram, which is a visual representation of sound waves to fit with the required CNN data.

![spectogram example](imgs/spectrogram-example.png)

---

## Architecture

The model consists of three convolutional layers with max pooling and dropout applied after each layer. Each image is a 3 second clip of the original song spectrograms. The shape of the images are (130, 128) or (time steps per clip, mel frequency bands). 

Spectrograms are single channel becuase audio data is recorded only as amplitude of frequency, therefore final input shape is (130, 128, 1)

### Mockup 
![example model](imgs/nn.svg)

---

## Accessing Model 

Model versions are found in the models directory with their version number. To access use Keras.load_model

```python
import keras

model = keras.models.load_model("*/model/models/V*")
```

## Interfacing with model

Use predict method on models to get array of length 10 with softmax probabilities of genres. This works with any batch size. The predict method will give shape of (batch_size, num_of_genres) if batch_size greater than 1. The mel spectrograms must be shape (130, 128) with all the values normalized using min-man (0 to 1).

```python
prediction = model.predict(mel_spectrogram_batch)  # prediction.shape = (batch_size, num_of_genres = 10)
```

