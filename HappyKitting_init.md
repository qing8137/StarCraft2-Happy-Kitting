                    Init smart_agent and HappyKitting mini map

1. Download (HK3V2)HappyKitting3V2.SC2Map 
2. Move HK3V2 to StarCraft2/Maps/mini_games
3. Remove the original BuildMarines.SC2Map
4. Rename HK3V2 as BuildMarines.SC2Map
    ->  So we don't have to change the configurations for the pysc2 to run our 
        map
5. Download smart_agent.py from our github repository
6. Move smart_agent.py to pysc2/agents

7. run command 

python -m pysc2.bin.agent \
--map BuildMarines \
--agent smart_agent.SmartAgent \
--agent_race T \
--max_agent_steps 0 \
--norender