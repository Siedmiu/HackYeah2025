# Run game functions only when SpawnTimer is >= 0 (game is active)
execute if entity @e[type=marker,tag=gameController,scores={SpawnTimer=0..}] run function bow_range:spawn_target
execute if entity @e[type=marker,tag=gameController,scores={SpawnTimer=0..}] run function bow_range:position_target
execute if entity @e[type=marker,tag=gameController,scores={SpawnTimer=0..}] run function bow_range:detect_hits
execute if entity @e[type=marker,tag=gameController,scores={SpawnTimer=0..}] run function bow_range:handle_hits
execute if entity @e[type=marker,tag=gameController,scores={SpawnTimer=0..}] run function bow_range:cleanup
