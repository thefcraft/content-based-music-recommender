# Content-Based Music Recommender

here i implemented music recommender using BYOL SSL technique for my local music library

## How to run

please put all your audio file in the music_dir see src/config.py and .env.example

### Train

`uv run train.py`

### Inference

add filename or path to audio in the code and run (innitially it takes time to build vector db on the music_dir)

`uv run inference.py`

## NOTE (for me)

i can improve this more but due to some personal problems stopping work on this for now and pre commiting the repo to github.

## Understanding the Mel Spectrogram

[Understanding the Mel Spectrogram](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)

### The Fourier Transform

The Fourier transform is a mathematical formula that allows us to decompose a signal into it’s individual frequencies and the frequency’s amplitud.

In other words, it converts the signal from the time domain into the frequency domain. The result is called a spectrum.

### The Spectrogram

The fast Fourier transform is a powerful tool that allows us to analyze the frequency content of a signal, but what if our signal’s frequency content varies over time? Such is the case with most audio signals such as music and speech.

These signals are known as non periodic signals. We need a way to represent the spectrum of these signals as they vary over time. You may be thinking, “hey, can’t we compute several spectrums by performing FFT on several windowed segments of the signal?” Yes! This is exactly what is done, and it is called the short-time Fourier transform.

The FFT is computed on overlapping windowed segments of the signal, and we get what is called the spectrogram.

### The Mel Scale and Mel Spectrogram

Studies have shown that humans do not perceive frequencies on a linear scale. We are better at detecting differences in lower frequencies than higher frequencies. For example, we can easily tell the difference between 500 and 1000 Hz, but we will hardly be able to tell a difference between 10,000 and 10,500 Hz, even though the distance between the two pairs are the same.

In 1937, Stevens, Volkmann, and Newmann proposed a unit of pitch such that equal distances in pitch sounded equally distant to the listener. This is called the mel scale. We perform a mathematical operation on frequencies to convert them to the mel scale.

A mel spectrogram is a spectrogram where the frequencies are converted to the mel scale.
