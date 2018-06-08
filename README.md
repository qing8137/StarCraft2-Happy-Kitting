# StarCraft2-Happy-Kitting
StarCraft2 A.I. with DDPG


#### main.py
Instruction:

Only two parts to edit:
```python
# modify agent name here: "agent", "YourAgentFileName.YourAgentClassName", "Description"
flags.DEFINE_string("agent", "test_agent.TestAgent",
                    "Which agent to run")

# edit map used here
flags.DEFINE_string("map", "HappyKiting3V2", "Name of a map to use.")
```
Run:
`python agent.py`
without any commandline parameters

<a href="https://www.youtube.com/watch?v=Gx3hAEql9gA" target="_blank"><img src="http://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg" 
alt="DQN demo" width="240" height="180" border="10" /></a>