# THE 2020 PERSONALIZED VOICE TRIGGER CHALLENGE BASELINE SYSTEM

This is the baseline system for PVTC2020 which is a satellite event of ISCSLP2021(https://www.iscslp2021.org), for more information about the challenge and dataset you can visit the website https://www.pvtc2020.org. 

Our baseline method consists of a wake-up system and a speaker verification system. As shown in the figure below, we designed a two-pass approach to respond whenever the target speaker says the wake-up word. When the wake-up word detection system triggers, the speaker verification system starts to decide whether the audio segment that triggered the wake-up word detector is indeed spoken by the enrolled target speaker.

![image](https://github.com/jiay7/THE-2020-PERSONALIZED-VOICE-TRIGGER-CHALLENGE-BASELINE-SYSTEM/blob/master/wake_sv.png)

Pleaes refer to KWS_README.md and SV_README.md for more details
.
In this challenge, we provide a leaderboard, ranked by the metric ![1](http://latex.codecogs.com/svg.latex?score_{wake-up}). As for the ![2](http://latex.codecogs.com/svg.latex?score_{wake-up}) metric, the lower the better. The ![3](http://latex.codecogs.com/svg.latex?score_{wake-up}) is provided as our challenge metric and it is calculated from the miss rate and the false alarm (FA) rate in the following form:

![4](http://latex.codecogs.com/svg.latex?\begin{equation}score_{wake-up}=Miss+alpha*FA\end{equation})

Results are shown in S_kws_task1.jpg and S_kws_task2.jpg.  We choose the final score under alpha is equal to nineteen as model's performance criterion. (![4](http://latex.codecogs.com/svg.latex?p_{target})=0.05, ![4](http://latex.codecogs.com/svg.latex?\begin{equation}score_{wake-up}=(p_{target}*Miss+(1-p_{target})*FA)*20\end{equation}))

By updating the method of determining the threshold of the speaker verification system (using the mean threshold of EER and minDCF instead of the threshold of EER), we proposed Baseline_v2, which has been greatly improved in the development set.

| Model | Task1 | Task2 |
| :----:| :----: | :----: |
| Baseline_v1 | 0.1981 | 0.3334 |
| Baseline_v2 | 0.1009 | 0.1415 |

All the result are based on the development set.

The run.sh is the current recommended recipe.



Contact us: PVTC2020@lenovo.com

