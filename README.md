# StarCraft2-Happy-Kitting
StarCraft2 A.I. with DQN

```
Group Members
Qing Lin,           
Chu-Hung Cheng,         
Yean Li,           
Siyuan Yao,             
Oi Lam Sou,           
Nghiem Trong Van,       
```

#### main.py

```python
Map
Player Units: 2 Reapers
Enemy Unit: 1 Zealor
Victory Condition: Eliminate the enemy
Time Limit: 1 min 30 sec
Modified from PySC2 mini games (DefeatRoaches)

```
Run:
`python3 main.py` in `~\DQN`
without any commandline parameters

The overall performance is satisfying. The main problem we have so far is SELETECING units. There are a lot of misclickings happened in the games. If we take a close look at the demo, at 1:16, the agent is actrually trying to select the unit that was closest to the Zealot. However, it misclicked the Zealot, and thus it failed to select the Reaper for future instructions.

`click to watch the demo`

[![DQN Demo](https://img.youtube.com/vi/EH-jz9o_wDg/0.jpg)](https://www.youtube.com/watch?v=EH-jz9o_wDg "DQN Demo")
