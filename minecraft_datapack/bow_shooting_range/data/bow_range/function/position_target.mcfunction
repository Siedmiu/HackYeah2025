# Position targets at controller location
execute as @e[type=armor_stand,tag=needsPosition] at @e[type=marker,tag=gameController,limit=1] run tp @s ~ ~ ~

# Randomize Y position between 65 and 75
execute as @e[type=armor_stand,tag=needsPosition] store result score @s TargetTimer run random value 65..75
execute as @e[type=armor_stand,tag=needsPosition] store result entity @s Pos[1] double 1 run scoreboard players get @s TargetTimer

# Randomize Z position (spread targets across z-axis)
execute as @e[type=armor_stand,tag=needsPosition] at @s run spreadplayers ~ ~ 0 10 false @s

# Initialize target timer
execute as @e[type=armor_stand,tag=needsPosition] run scoreboard players set @s TargetTimer 0

# Remove positioning tag
tag @e[type=armor_stand,tag=needsPosition] remove needsPosition
