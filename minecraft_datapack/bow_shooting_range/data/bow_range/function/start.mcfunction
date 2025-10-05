# Start or resume the game
scoreboard players set @e[type=marker,tag=gameController] SpawnTimer 0
tellraw @a {"text":"[Bow Range] Game started!","color":"green"}