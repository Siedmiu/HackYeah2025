# Create scoreboards
scoreboard objectives add BowScore dummy "Bow Range Score"
scoreboard objectives setdisplay sidebar BowScore
scoreboard objectives add TargetTimer dummy
scoreboard objectives add SpawnTimer dummy
scoreboard objectives add RandomX dummy
scoreboard objectives add RandomZ dummy
scoreboard objectives add TargetCount dummy

# Create game controller marker
kill @e[type=marker,tag=gameController]
summon minecraft:marker 0 64 0 {Tags:["gameController"]}
scoreboard players set @e[type=marker,tag=gameController] SpawnTimer 0

# Set max targets constant
scoreboard players set #maxTargets TargetCount 5

# Confirmation message
tellraw @a {"text":"[Bow Range] Datapack loaded! Use /function bow_range:start to begin","color":"green"}
