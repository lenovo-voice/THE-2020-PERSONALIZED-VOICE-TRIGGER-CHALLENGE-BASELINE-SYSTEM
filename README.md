# THE 2020 PERSONALIZED VOICE TRIGGER CHALLENGE BASELINE SYSTEM

Our baseline system consists of a wake-up system and a speaker verification system. As shown in figure below, we designed a two-pass system to respond whenever anyone in the vicinity says the trigger phrase. When the KWS system triggers, the speaker verification system starts to decide whether the sound that triggered the  detector is likely to be keywords spoken by the enrolled user. 

![image](https://github.com/jiay7/THE-2020-PERSONALIZED-VOICE-TRIGGER-CHALLENGE-BASELINE-SYSTEM/blob/master/wake_sv.png)

See KWS_README.md and SV_README.md for more details.

In this challenge, we provide a leaderboard, ranked by the metric ![1](http://latex.codecogs.com/svg.latex?score_{wake-up}). As for the ![2](http://latex.codecogs.com/svg.latex?score_{wake-up}) metric, the lower the better. The ![3](http://latex.codecogs.com/svg.latex?score_{wake-up}) is provided as our challenge metric and it is calculated from the miss rate and the false alarm (FA) rate in the following form:

![3](http://latex.codecogs.com/svg.latex?\begin{equation}score_{wake-up}=Miss+alpha*FA\end{equation})

Results are shown in S_kws_task1.jpg and S_kws_task2.jpg.  We choose the final score under alpha is twenty as model's performance criterion.

| Model | Task1 | Task2 |
| :----:| :----: | :----: |
| Baseline | 20.7% | 34.8% |

The run.sh is the current recommended recipe.





