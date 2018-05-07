                    Init smart_agent and HappyKitting mini map

1. Download (HK3V2)HappyKitting3V2.SC2Map 
2. Move HK3V2 to StarCraft2/Maps/mini_games
3. To run the Map, there is two routes we can do.
 	*Remove the original BuildMarines.SC2Map*
 	*Rename HK3V2 as BuildMarines.SC2Map*

 	#OR#

 	*Locate your pysc2 folder in your computer, for mac, if you downloaded python3.6 through homebrew. the path should be 
 	```/usr/local/lib/python3.6/site-packages/pysc2```, then add the Map_name under minigame map*
4. Download smart_agent.py from our github repository

5. make sure you cd into the directory where smart_agent is located.
6. Run command python3 -m pysc2.bin.agent -- map HappyKitting3V2 --agent smart_agent.SmartAgent, ignore step 7 if you can run your map.


7. run command 

python -m pysc2.bin.agent \
--map BuildMarines \
--agent smart_agent.SmartAgent \
--agent_race T \
--max_agent_steps 0 \
--norender

### P.S.
    The smart_agent.py is not designed for our game, but a full game agent.
    Thus it does not learn much by running on  our map. This is just a demo of 
    how to run a "SmartAgent" on our map. All we will have to do is to modify
    the smart_agent.py to make it a better fit to our game.