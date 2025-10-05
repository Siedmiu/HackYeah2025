# Stop spawning new targets
scoreboard players set @e[type=marker,tag=gameController] SpawnTimer -999999
tellraw @a {"text":"[Bow Range] Game paused!","color":"yellow"}