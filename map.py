import pygame
from pytmx.util_pygame import load_pygame

class Tile(pygame.sprite.Sprite):
    def __init__(self, pos, surf, groups):
        super().__init__(groups)
        self.image = surf
        
        # self.overlay = pygame.Surface((30, 30), pygame.SRCALPHA)
        # self.overlay.fill((0, 0, 0, 16))
        sx = (self.image.get_width() - 30) // 2
        sy = (self.image.get_height() - 30) // 2
        # self.image.blit(self.overlay, (sx, sy))
        
        self.rect = self.image.get_rect(topleft=pos)

    def draw(self, screen):
        screen.blit(self.image, self.rect.topleft)

class Map():
    def __init__(self):
        super().__init__()
        self.tmx_data = load_pygame('./assets/map/mapa/nowa_mapa.tmx')
        self.layers = [layer for layer in self.tmx_data.layers]
        self.sprite_groups = []
        self.visible_sprite_groups = []
    
    def generate_all_layers(self):
        self.sprite_groups = []

        for layer in self.layers:
            print(layer)
            if hasattr(layer, 'tiles'):
                group = pygame.sprite.Group()
                setattr(self, f"{layer.name}_group", group)
                print(group)
                self.sprite_groups.append(group)

                for x, y, surf in layer.tiles():
                    scaled_surf = pygame.transform.scale(
                        surf, (surf.get_width() * 2, surf.get_height() * 2)
                    )
                    pos = (x * 32, y * 32)
                    Tile(pos=pos, surf=scaled_surf, groups=group)

    def generate_visible_layers(self):
        self.visible_sprite_groups = []

        for layer in self.layers:
            if not getattr(layer, 'visible', True):
                continue

            if hasattr(layer, 'tiles'):
                group = pygame.sprite.Group()
                setattr(self, f"{layer.name}_visible_group", group)
                self.visible_sprite_groups.append(group)

                for x, y, surf in layer.tiles():
                    if surf is None:
                        continue
                    scaled_surf = pygame.transform.scale(
                        surf, (surf.get_width() * 2, surf.get_height() * 2)
                    )
                    pos = (x * 32, y * 32)
                    Tile(pos=pos, surf=scaled_surf, groups=group)
    
    def create_bfs_map(self):
        gold_group = getattr(self, "gold_chest_tables_group", None)
        silver_group = getattr(self, "silver_chest_tables_group", None)
        collision_group = getattr(self, "collisions_group", None)
        carpets_group = getattr(self, "carpets_group", None)
        deliver_silver_table_group = getattr(self, "deliver_silver_table_group", None)
        deliver_gold_table_group = getattr(self, "deliver_gold_table_group", None)

        def tile_exists_in_group(group, grid_x, grid_y):
            if not group:
                return False
            for spr in group:
                if spr.rect.x == grid_x * 32 and spr.rect.y == grid_y * 32:
                    return True
            return False

        self.bfs_map = []
        map_height = self.tmx_data.height
        map_width = self.tmx_data.width

        for y in range(map_height):
            row = []
            for x in range(map_width):
                if tile_exists_in_group(gold_group, x, y):
                    row.append(3)
                elif tile_exists_in_group(silver_group, x, y):
                    row.append(4)
                elif tile_exists_in_group(deliver_silver_table_group, x, y):
                    row.append(6)
                elif tile_exists_in_group(deliver_gold_table_group, x, y):
                    row.append(7)
                elif tile_exists_in_group(collision_group, x, y):
                    row.append(2)
                elif tile_exists_in_group(carpets_group, x, y):
                    row.append(5)
                else:
                    row.append(0)
            self.bfs_map.append(row)

        return self.bfs_map
    
    def create_shelf_list(self):
        gold_group = getattr(self, "gold_chest_tables_group", None)
        silver_group = getattr(self, "silver_chest_tables_group", None)
        def tile_exists_in_group(group, grid_x, grid_y):
            if not group:
                return False
            for spr in group:
                if spr.rect.x == grid_x * 32 and spr.rect.y == grid_y * 32:
                    return True
            return False

        self.shelf_list_gold = []
        self.shelf_list_iron = []
        map_height = self.tmx_data.height
        map_width = self.tmx_data.width

        for y in range(map_height):
            for x in range(map_width):
                if tile_exists_in_group(gold_group, x, y):
                    self.shelf_list_gold.append((x,y))
                elif tile_exists_in_group(silver_group, x, y):
                    self.shelf_list_iron.append((x,y)) 
        return self.shelf_list_iron,self.shelf_list_gold
    