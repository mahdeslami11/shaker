<p align="center">
    <a target="_blank" href="https://colab.research.google.com/github/bshall/soft-vc/blob/main/soft-vc-demo.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
</p>

# Acoustic-Model

Training and inference scripts for the acoustic models in [A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion](https://ieeexplore.ieee.org/abstract/document/9746484). For more details see [soft-vc](https://github.com/bshall/soft-vc). Audio samples can be found [here](https://bshall.github.io/soft-vc/). Colab demo can be found [here](https://colab.research.google.com/github/bshall/soft-vc/blob/main/soft-vc-demo.ipynb).

<div align="center">
    <img width="100%" alt="Soft-VC"
      src="https://raw.githubusercontent.com/bshall/acoustic-model/main/acoustic-model.png">
</div>
<div>
  <sup>
    <strong>Fig 1:</strong> Architecture of the voice conversion system. a) The <strong>discrete</strong> content encoder clusters audio features to produce a sequence of discrete speech units. b) The <strong>soft</strong> content encoder is trained to predict the discrete units. The acoustic model transforms the discrete/soft speech units into a target spectrogram. The vocoder converts the spectrogram into an audio waveform.
  </sup>
</div>

## Example Usage

### Programmatic Usage

```python
import torch
import numpy as np

# Load checkpoint (either hubert_soft or hubert_discrete)
acoustic = torch.hub.load("bshall/acoustic-model:main", "hubert_soft").cuda()

# Load speech units
units = torch.from_numpy(np.load("path/to/units"))

# Generate mel-spectrogram
mel = acoustic.generate(units)
```

### Script-Based Usage

```
usage: generate.py [-h] {soft,discrete} in-dir out-dir

Generate spectrograms from input speech units (discrete or soft).

positional arguments:
  {soft,discrete}  available models (HuBERT-Soft or HuBERT-Discrete)
  in-dir           path to the dataset directory.
  out-dir          path to the output directory.

optional arguments:
  -h, --help       show this help message and exit
```

## Training

### Step 1: Dataset Preparation

Download and extract the [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset. The training script expects the following tree structure for the dataset directory:

```
└───wavs
    ├───dev
    │   ├───LJ001-0001.wav
    │   ├───...
    │   └───LJ050-0278.wav
    └───train
        ├───LJ002-0332.wav
        ├───...
        └───LJ047-0007.wav
```

The `train` and `dev` directories should contain the training and validation splits respectively. The splits used for the paper can be found [here](https://github.com/bshall/acoustic-model/releases/tag/v0.1).

### Step 2: Extract Spectrograms

Extract mel-spectrograms using the `mel.py` script:

```
usage: mels.py [-h] in-dir out-dir

Extract mel-spectrograms for an audio dataset.

positional arguments:
  in-dir      path to the dataset directory.
  out-dir     path to the output directory.

optional arguments:
  -h, --help  show this help message and exit
```

for example:

```
python mel.py path/to/LJSpeech-1.1/wavs path/to/LJSpeech-1.1/mels
```

At this point the directory tree should look like:

```
├───mels
│   ├───...
└───wavs
    ├───...
```

### Step 3: Extract Discrete or Soft Speech Units

Use the HuBERT-Soft or HuBERT-Discrete content encoders to extract speech units. First clone the [content encoder repo](https://github.com/bshall/hubert) and then run `encode.py` (see the repo for details):

```
usage: encode.py [-h] [--extension EXTENSION] {soft,discrete} in-dir out-dir

Encode an audio dataset.

positional arguments:
  {soft,discrete}       available models (HuBERT-Soft or HuBERT-Discrete)
  in-dir                path to the dataset directory.
  out-dir               path to the output directory.

optional arguments:
  -h, --help            show this help message and exit
  --extension EXTENSION
                        extension of the audio files (defaults to .flac).
```

for example:

```
python encode.py soft path/to/LJSpeech-1.1/wavs path/to/LJSpeech-1.1/soft --extension .wav
```

At this point the directory tree should look like:

```
├───mels
│   ├───...
├───soft/discrete
│   ├───...
└───wavs
    ├───...
```

### Step 4: Train the Acoustic-Model

```
usage: train.py [-h] [--resume RESUME] [--discrete] dataset-dir checkpoint-dir

Train the acoustic model.

positional arguments:
  dataset-dir      path to the data directory.
  checkpoint-dir   path to the checkpoint directory.

optional arguments:
  -h, --help       show this help message and exit
  --resume RESUME  path to the checkpoint to resume from.
  --discrete       Use discrete units.
```

## Links

- [Soft-VC repo](https://github.com/bshall/soft-vc)
- [Soft-VC paper](https://ieeexplore.ieee.org/abstract/document/9746484)
- [HuBERT content encoders](https://github.com/bshall/hubert)
- [HiFiGAN vocoder](https://github.com/bshall/hifigan)

## Citation

If you found this work helpful please consider citing our paper:

```
@inproceedings{
    soft-vc-2022,
    author={van Niekerk, Benjamin and Carbonneau, Marc-André and Zaïdi, Julian and Baas, Matthew and Seuté, Hugo and Kamper, Herman},
    booktitle={ICASSP}, 
    title={A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion}, 
    year={2022}
}
```



1- target

The purpose of this project is to eliminate noise on the spectrum

2-Describe innovation

Table 2: Comparison of loss scores for model variations
Model Variation L1 (Train) L1 (Validation) Guided Attention (Validation)
M1 Standard attention, CE loss 0.0288 0.0611 27.5 × 10−4
M2 M1 + guided attention 0.0249 0.0484 3.99 × 10−4
M3 M2 + local char encodings 0.0245 0.0485 4.19 × 10−4
M4 M1 + positional encodings 0.0230 0.0490 8.42 × 10−4
M5 M4 without CE loss 0.0235 0.0490 17 × 10−4
cially in a language like English where the same characters are spoken many different ways this is
not trivial. We observe a simple visualization of the learned character embeddings L in Fig. 3 with
some interesting behavior from the model. We could potentially analyze further by examining the
output encodings produced by TextEnc, and seeing how these change for words like ”car” and ”cat”.
Figure 3: Visualization of implicitly-learned character embeddings. Characters tend to cluster based
on their semantics, as well as those with similar sound. We notice an interesting separation of
characters with non-vocalized sounds (percussive sounds like ”sh” ”th” etc. that don’t engage our
voice)
5 Conclusion and Future Work
In conclusion, an interesting learning from this project has been an appreciation of the sheer power
and flexibility of deep learning models to learn alignments, character-phoneme relationships, char-
acter embeddings, and an audio language model simultaneously. While we used a larger amount of
cloud compute credits due to the various experiments we ran - given our training times it is quite
possible to build a model with decent prosody, voice and pronunciation in under $75 using our code.
We plan to try to extend this model to learning different languages to get a better estimate of the
above. It is possible to improve quality by constraining the attention at inference as in [9]. Also, we
would like to experiment with two semi-supervised methods for a better initialization of the character
embeddings L and weights of AudioEnc and AudioDec. This could be done via Word2Vec-like
embeddings for L, and separately training the audio language model (see Section 4.2.1) on audio
without labeled transcripts. This could be useful when labeled transcripts might not be as abundant
for other languages. Finally, we are also interested in extending this general idea of ’symbols-to-
sound’ to a more general problem of generating audio stylistically from a sequence of token inputs
(this could be MIDI inputs for music or similar).




