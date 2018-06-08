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

[![DQN Demo](https://img.youtube.com/vi/Gx3hAEql9gA/0.jpg)](https://www.youtube.com/watch?v=Gx3hAEql9gA "DQN Demo")
