# Count current targets
execute store result score @e[type=marker,tag=gameController,limit=1] TargetCount run execute if entity @e[type=armor_stand,tag=shootTarget]

# Spawn new targets until we have 5
execute as @e[type=marker,tag=gameController] if score @s TargetCount < #maxTargets TargetCount at @s run summon minecraft:armor_stand ~ ~ ~ {Invisible:1b,Invulnerable:0b,NoGravity:1b,Tags:["shootTarget","needsPosition"],DisabledSlots:4144959,Passengers:[{id:"minecraft:block_display",block_state:{Name:"minecraft:target"},transformation:{left_rotation:[0f,0f,0f,1f],right_rotation:[0f,0f,0f,1f],translation:[-0.5f,0.0f,-0.5f],scale:[1f,1f,1f]},Tags:["targetDisplay"]}]}