# THE 2020 PERSONALIZED VOICE TRIGGER CHALLENGE BASELINE SYSTEM

Our baseline method consists of a wake-up system and a speaker verification system. As shown in the figure below, we designed a two-pass approach to respond whenever the target speaker says the wake-up word. When the wake-up word detection system triggers, the speaker verification system starts to decide whether the audio segment that triggered the wake-up word detector is indeed spoken by the enrolled target speaker.

![image](https://github.com/jiay7/THE-2020-PERSONALIZED-VOICE-TRIGGER-CHALLENGE-BASELINE-SYSTEM/blob/master/wake_sv.png)

Pleaes refer to KWS_README.md and SV_README.md for more details
.
In this challenge, we provide a leaderboard, ranked by the metric ![1](http://latex.codecogs.com/svg.latex?score_{wake-up}). As for the ![2](http://latex.codecogs.com/svg.latex?score_{wake-up}) metric, the lower the better. The ![3](http://latex.codecogs.com/svg.latex?score_{wake-up}) is provided as our challenge metric and it is calculated from the miss rate and the false alarm (FA) rate in the following form:

![3](http://latex.codecogs.com/svg.latex?\begin{equation}score_{wake-up}=Miss+alpha*FA\end{equation})

Results are shown in S_kws_task1.jpg and S_kws_task2.jpg.  We choose the final score under alpha is equal to nineteen as model's performance criterion. 

![5](http://latex.codecogs.com/svg.latex?P_{target})=0.05,![6](http://latex.codecogs.com/svg.latex?Score=(P_{target}*Miss_{rate}+(1-P_{target})


| Model | Task1 | Task2 |
| :----:| :----: | :----: |
| Baseline | 0.2121 | 0.3138 |

The run.sh is the current recommended recipe.





