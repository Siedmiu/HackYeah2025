# Clear all targets
kill @e[type=armor_stand,tag=shootTarget]

# Clear all arrows
kill @e[type=arrow]

# Reset all scores
scoreboard players reset * BowScore

# Reset spawn timer
scoreboard players set @e[type=marker,tag=gameController] SpawnTimer 0

tellraw @a {"text":"[Bow Range] Game reset!","color":"gold"}