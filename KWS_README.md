# THE 2020 PERSONALIZED VOICE TRIGGER CHALLENGE KEYWORD SPOTTING BASELINE SYSTEM

Implement Chinese keyword spotting using LSTM+Average Pooling. This model is supposed to run on Raspberry devices, with low cpu and memory requirement.

training data: 213k speech wav

dev data: task1: 24k speech wav ; task2: 50k speech wav 

keyword for experiment: 小乐小乐

basic structure: preprocessing -> LSTM -> decode

## environment setup

```
pip install -r ./src/requirements.txt
```

Replace KALDI_ROOT in ./src/path.sh with your Kaldi path. Download and compile Kaldi[https://github.com/kaldi-asr/kaldi]()


## preprocessing

signal -> stft(linear spectrogram) -> mel spectrogram

In the experiment, we use **fft_size=25ms** and **hop_size=10ms** for stft, **n_mel=80** for mel filter bank with LSTM **hidden_size=128**  is enough.

Maybe larger hidden size and deeper network can perform better. But in our case, there is no need to use that large model.

## label

We determine target word labels by force-alignment with an LVCSR system trained with the AISHELL-2 dataset. Here, for keyword "xiao le xiao# le#", we find out the ending time of "xiao", "le", and "xiao#", and include its previous 20 frames and next 20 frames to construct a window of 40 frames. Log fbank is adopted as our input acoustic features. 

For example,

* 0 for filler
* 1 for xiaole
* 2 for lexiao#
* 3 for xiao#le#

And therefore we have a output space of 4.

Feature **xiaole_xiaole_PVTC0001-xiaole-1-0010.wav.npy** will be labeled as **xiaole**

Feature **lexiao#_xiaole_PVTC0001-xiaole-1-0142.wav.npy** will be labeled as **lexiao#**

Feature **xiao#le#_xiaole_PVTC0001-xiaole-1-0210.wav.npy** will be labeled as **xiao#le#**

Feature **other_PVTC0121-xiaole-6-0443.wav.npy** will be labeled as **filler**

## model

LSTM -> Average Pooling -> Fully-connection layer -> Cross-entropy loss(forward decode)

***Training model***

The model is constructed with a two-layer stacked LSTM structure with **hidden_size=128**, followed by an average pooling layer and a final linear layer.


The baseline system is trained with cross-entropy loss. Stochastic gradient descent with Nesterov momentum is selected as the optimizer. The learning rate is first initialized as 0.01 and decreases by a factor of 0.1 when the training loss plateau. We train the model for 100 epochs with a batch size of 128 and employ early stopping when the training loss is not decreasing. 

In the evaluation period, we use a sliding window of 100 frames to compute the confidence score.


***Deployment Model***


After the training process, the sequence of acoustic features is projected to a posterior probabilities sequence of keywords by the model. In the module of confidence computation, we adopt the method proposed in [https://arxiv.org/abs/2005.03633]() to make the decisions. In this approach, we define a sliding window with the length of 150 frames which is used to compute scores. We smooth the output probabilities at a length of 50 frames by taking average. The system triggers whenever the confidence score exceed the pre-defined threshold.

We add a simple VAD to detect voice activity.

A major drawback of mel filter bank feature is that it will toally mess up when carrying the state of a long speech. **So we clear the rnn state after each trigger and each non-speech segment detected by the VAD.** The VAD must be carefully tuned, otherwise it will cut off unfinished speech and clean the rnn state.

***Result***


Results are shown in wake_task1.jpg and wake_task2.jpg. We choose the false rejection rate under one false alarm per hour as model's performance criterion. Table presents the KWS performance of the model regarding false rejection rate when the false alarm rate per hour is 1.

| Model | Task1 | Task2 |
| :----:| :----: | :----: |
| KWS baseline | 2.00% | 5.09% |


