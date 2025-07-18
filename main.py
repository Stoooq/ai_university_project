import random
import pygame
import sys
import joblib
import pandas as pd
from forklift import Forklift
from chest import Chest
from functions.bfs import shortestPath
from map import Map
from functions.forklift_collision import check_collision
from copy import deepcopy
 
SCREEN_WIDTH = 960
SCREEN_HEIGHT = 640
 
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
running = True
 
game_map = Map()
game_map.generate_all_layers()
game_map.generate_visible_layers()
game_bfs_map = game_map.create_bfs_map()
shelf_list_iron,shelf_list_gold = game_map.create_shelf_list()
 
free_iron_spots = shelf_list_iron.copy()
free_gold_spots = shelf_list_gold.copy()
 
forklift = pygame.sprite.GroupSingle()
forklift.add(Forklift(pos=((SCREEN_WIDTH //2) +1024, (SCREEN_HEIGHT //2) +1024)))
 
chest_group = pygame.sprite.Group()
chest_spawn_timer = pygame.time.get_ticks()
chest_locations=[]
 
DEPOSIT_TILE_X = 10
DEPOSIT_TILE_Y = 10
DEPOSIT_LOC_TUPLE = (DEPOSIT_TILE_X, DEPOSIT_TILE_Y)
model = joblib.load('warehouse_priority_tree_v6.joblib')
t=0
chest_used=[]
nprchest=[[],[],[],[],[],[]]
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
 
    current_time = pygame.time.get_ticks()
    if current_time - chest_spawn_timer > 1500:
        if len(game_map.take_table_group) > 0:
            random_chest_spawn_sprite = random.choice(game_map.take_table_group.sprites())
            chest_types = ['gold', 'iron']
            chest_size = ['small', 'big']
            random_type = random.choice(chest_types)
            random_size=random.choice(chest_size)
 
            spawn_tile_x = random_chest_spawn_sprite.rect.x // 32
            spawn_tile_y = random_chest_spawn_sprite.rect.y // 32
 
            if game_bfs_map[spawn_tile_y][spawn_tile_x] != 1:
                chest_group.add(Chest(pos=(random_chest_spawn_sprite.rect.x, random_chest_spawn_sprite.rect.y), chest_type=random_type,chest_size=random_size, time=current_time, priority=0, weight=random.uniform(5, 20)))
                #chest_group.add(Chest(pos=(random_chest_spawn_sprite.rect.x, random_chest_spawn_sprite.rect.y), time=current_time, priority=0, weight=random.uniform(5, 20)))
                game_bfs_map[spawn_tile_y][spawn_tile_x] = 1
                chest_locations.append((spawn_tile_x, spawn_tile_y))
        chest_spawn_timer = current_time
       
    screen.fill("black")
    for sprite_group in game_map.visible_sprite_groups:
        sprite_group.draw(screen)
    chest_group.draw(screen)
   
    forklift_sprite = forklift.sprite
 
    if not forklift_sprite.carrying_chest:
        if not forklift_sprite.bfs_active and forklift_sprite.target_chest is None and chest_locations:
           
            if len(chest_locations) >= 1:
                nprchest=[[],[],[],[],[],[]]
                for chest in chest_group:
                    if chest.on_conter:
                    #if  chest not in chest_used:
                        current_time = pygame.time.get_ticks()
                        print(nprchest)
                        if chest.chest_type=='iron':
                           
                            X_new = pd.DataFrame([{
                                'size': str(chest.chest_size),
                                'type': str(chest.chest_type),
                                'weight': float(chest.weight),
                                'dist_f_ch': len(shortestPath(game_bfs_map, forklift_sprite.rect.y // 32, forklift_sprite.rect.x // 32, forklift_sprite.direction,[[chest.pos[1]//32, chest.pos[0]//32]])[0]),
                                'dist_ch_s': len(shortestPath(game_bfs_map, chest.pos[1]//32, chest.pos[0]//32, 2,shelf_list_iron)[0]),
                                'time_sec': current_time - chest.time,
                                'bench_free': random.randint(0,10)
                            }])
                            priority_pred = model.predict(X_new)
                            chest.priority = priority_pred[0]  
                           
                        else:
       
                            X_new = pd.DataFrame([{
                                'size': str(chest.chest_size),
                                'type': str(chest.chest_type),
                                'weight': float(chest.weight),
                                'dist_f_ch': len(shortestPath(game_bfs_map, forklift_sprite.rect.y // 32, forklift_sprite.rect.x // 32, forklift_sprite.direction,[[chest.pos[1]//32, chest.pos[0]//32]])[0]),
                                'dist_ch_s': len(shortestPath(game_bfs_map, chest.pos[1]//32, chest.pos[0]//32, 2,shelf_list_gold)[0]),
                                'time_sec': current_time - chest.time,
                                'bench_free': random.randint(0,10)
                            }])
                            priority_pred = model.predict(X_new)
                            chest.priority = priority_pred[0]
   
                           
                        #print(t)
                        t=t+1
                        #chest_used.append(chest)
                        nprchest[chest.priority].append([chest.pos[0]//32, chest.pos[1]//32])
 
                temp_chest_loc=[]
                for i in range(1,6):
                    #print(i,nprchest[-i])
                    if len(nprchest[-i])>0:
                        temp_chest_loc= deepcopy(nprchest[-i])
                        break
                #
                if temp_chest_loc:
                   
                    path_data = shortestPath(
                        game_bfs_map,
                        forklift_sprite.rect.y // 32,
                        forklift_sprite.rect.x // 32,
                        forklift_sprite.direction,
                        temp_chest_loc
                    )
 
                   
 
                    if path_data and path_data[0]:
                        instr, target = path_data
                        tx_px, ty_px = target[0] * 32, target[1] * 32
                        for c in chest_group.sprites():
                            if c.rect.x == tx_px and c.rect.y == ty_px:
                                forklift_sprite.target_chest = c
                                forklift_sprite.set_bfs_instructions(instr)
                                c.on_conter=False
                                break
 
                        for chest in chest_group:
                            if target == [chest.pos[1]//32, chest.pos[0]//32]:
                                #chest_used.append(chest)
                                chest.on_conter=False
                                break
                            #print(chest_used)
        elif not forklift_sprite.bfs_active and forklift_sprite.target_chest:
            fx, fy = forklift_sprite.rect.x // 32, forklift_sprite.rect.y // 32
            tc = forklift_sprite.target_chest
            tcx, tcy = tc.rect.x // 32, tc.rect.y // 32
            forklift_sprite.carrying_chest = True
            game_bfs_map[tcy][tcx] = 2
            if (tcx, tcy) in chest_locations:
                chest_locations.remove((tcx, tcy))
            spots = free_gold_spots if tc.chest_type == 'gold' else free_iron_spots
            path_dep = shortestPath(
                game_bfs_map,
                fy, fx,
                forklift_sprite.direction,
                spots
            )
            if path_dep and path_dep[0]:
                instr_dep, target_dep = path_dep
                forklift_sprite.deposit_target = target_dep
                forklift_sprite.set_bfs_instructions(instr_dep)
            else:
                forklift_sprite.carrying_chest = False
                game_bfs_map[tcy][tcx] = 1
                chest_locations.append((tcx, tcy))
                tc.rect.topleft = (tcx * 32, tcy * 32)
                forklift_sprite.target_chest = None
 
    elif forklift_sprite.carrying_chest:
        if not forklift_sprite.bfs_active and forklift_sprite.deposit_target:
            fx, fy = forklift_sprite.rect.x // 32, forklift_sprite.rect.y // 32
            dx, dy = forklift_sprite.deposit_target
            if (abs(fx - dx) <= 1 and fy == dy) or (abs(fy - dy) <= 1 and fx == dx):
                tc = forklift_sprite.target_chest
                tc.rect.topleft = (dx * 32, dy * 32)
                chest_group.add(tc)
                game_bfs_map[dy][dx] = 1
                if tc.chest_type == 'gold' and (dx, dy) in free_gold_spots:
                    free_gold_spots.remove((dx, dy))
 
                elif tc.chest_type == 'iron' and (dx, dy) in free_iron_spots:
                    free_iron_spots.remove((dx, dy))
                forklift_sprite.target_chest = None
                forklift_sprite.deposit_target = None
                forklift_sprite.carrying_chest = False
            else:
                forklift_sprite.set_bfs_instructions([])
 
    forklift.draw(screen)
    forklift.update()
   
    check_collision(forklift, SCREEN_WIDTH, SCREEN_HEIGHT)
 
    pygame.display.flip()
    clock.tick(60)
   
pygame.quit()
sys.exit()