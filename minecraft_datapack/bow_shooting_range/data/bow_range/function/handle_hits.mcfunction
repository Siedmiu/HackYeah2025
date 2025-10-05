# Particle effects
execute as @e[type=armor_stand,tag=hit] at @s run particle minecraft:explosion ~ ~1 ~ 0.3 0.3 0.3 0.1 20 force
execute as @e[type=armor_stand,tag=hit] at @s run particle minecraft:firework ~ ~1 ~ 0.5 0.5 0.5 0.1 30 force

# Sound effects
execute as @e[type=armor_stand,tag=hit] at @s run playsound minecraft:entity.generic.explode master @a ~ ~ ~ 1 1.2
execute as @e[type=armor_stand,tag=hit] at @s run playsound minecraft:entity.player.levelup master @a ~ ~ ~ 0.5 2

# Award points to nearest player
execute as @e[type=armor_stand,tag=hit] at @s run scoreboard players add @p[distance=..100] BowScore 10

# Remove hit targets (passengers will be removed automatically)
kill @e[type=armor_stand,tag=hit]
