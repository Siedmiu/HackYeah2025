# Increment target lifetime timer
scoreboard players add @e[type=armor_stand,tag=shootTarget] TargetTimer 1

# Detect when arrow hits armor stand
execute as @e[type=armor_stand,tag=shootTarget,nbt={HurtTime:10s}] run tag @s add hit

# Remove expired targets (not hit within 200 ticks = 10 seconds)
execute as @e[type=armor_stand,tag=shootTarget,scores={TargetTimer=200..}] at @s run particle minecraft:smoke ~ ~1 ~ 0.2 0.2 0.2 0.05 15 force
execute as @e[type=armor_stand,tag=shootTarget,scores={TargetTimer=200..}] at @s run playsound minecraft:block.fire.extinguish master @a ~ ~ ~ 0.5 0.8
kill @e[type=armor_stand,tag=shootTarget,scores={TargetTimer=200..}]