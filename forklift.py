import pygame

from forklift_states import IDLE_DOWN, IDLE_SIDE, IDLE_UP, RUN_DOWN, RUN_SIDE, RUN_UP, SPRITES

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

class Forklift(pygame.sprite.Sprite):
    def __init__(self, pos, game_map=None, bfs_map=None):
        super().__init__()
        self.width = 16 * 2
        self.height = 16 * 2

        self.original_image = SPRITES[IDLE_DOWN]['imageSrc']
        self.subsurface_image = self.original_image.subsurface(
            pygame.Rect(0, 0, 16, 16)
        )
        self.scaled_image = pygame.transform.scale(
            self.subsurface_image,
            (self.subsurface_image.get_width() * 2, self.subsurface_image.get_height() * 2)
        )
        self.image = self.scaled_image
        self.rect = self.image.get_rect(
            topleft=(pos[0] - self.width // 2, pos[1] - self.height // 2)
        )

        self.velocity = 10
        self.maxVelocity = 10
        self.capacity = 0
        self.maxCapacity = 2
        self.game_map = game_map
        self.direction = SOUTH

        self.state = IDLE_DOWN
        self.current_frame = 0
        self.animation_speed = 10
        self.last_update = pygame.time.get_ticks()
        self.flipped = False
        self.grid_size = 16 * 2
        self.moving = False
        self.target_pos = self.rect.topleft
        self.prev_keys = pygame.key.get_pressed()

        self.target_chest = None
        self.carrying_chest = False

        self.bfs_map = bfs_map[:] if bfs_map else []
        self.bfs_index = 0

        self.bfs_active = False
        if self.bfs_map:
            self.bfs_active = True

    def set_bfs_instructions(self, instructions):
        self.bfs_map = instructions[:]
        self.bfs_index = 0
        self.bfs_active = bool(instructions)

    def follow_bfs_instructions(self):

        if self.moving or not self.bfs_map:
            if not self.bfs_map:
                self.bfs_active = False
            return

        instruction = self.bfs_map[self.bfs_index]

        if instruction == 'go':
            target_x, target_y = self.rect.topleft
            if self.direction == NORTH:
                target_y -= self.grid_size
            elif self.direction == SOUTH:
                target_y += self.grid_size
            elif self.direction == EAST:
                target_x += self.grid_size
            elif self.direction == WEST:
                target_x -= self.grid_size

            if self.is_valid_position(target_x, target_y):
                self.target_pos = (target_x, target_y)
                self.moving = True
                self.velocity = self.maxVelocity

        elif instruction == 'left':
            self.direction = (self.direction - 1) % 4

        elif instruction == 'right':
            self.direction = (self.direction + 1) % 4

        self.bfs_index += 1
        if self.bfs_index >= len(self.bfs_map):
            self.bfs_map = []
            self.bfs_active = False

    def update(self):
        self.follow_bfs_instructions() 
        self.forklift_control()
        self.animate()

        if self.carrying_chest and self.target_chest is not None:
            self.target_chest.rect.centerx = self.rect.centerx
            self.target_chest.rect.centery = self.rect.centery

        if self.moving:
            target_x, target_y = self.target_pos
            current_x, current_y = self.rect.topleft

            dx = 0
            dy = 0

            if current_x < target_x:
                dx = min(self.velocity, target_x - current_x)
            elif current_x > target_x:
                dx = max(-self.velocity, target_x - current_x)

            if current_y < target_y:
                dy = min(self.velocity, target_y - current_y)
            elif current_y > target_y:
                dy = max(-self.velocity, target_y - current_y)

            self.rect.x += dx
            self.rect.y += dy

            if (abs(self.rect.x - target_x) < self.velocity / 2 and abs(self.rect.y - target_y) < self.velocity / 2):
                self.rect.topleft = self.target_pos
                self.moving = False

    def is_valid_position(self, x, y):
        if x < 0 or y < 0 or x >= 960 or y >= 640:
            return False

        grid_x = x // 32
        grid_y = y // 32

        if self.game_map is not None:
            if 0 <= grid_y < len(self.game_map) and 0 <= grid_x < len(self.game_map[0]):
                if self.game_map[grid_y][grid_x] == 2:
                    return False
        return True

    def forklift_control(self):
        keys = pygame.key.get_pressed()

        if not self.moving and not self.bfs_map:
            if keys[pygame.K_LEFT] and not self.prev_keys[pygame.K_LEFT]:
                self.direction = (self.direction - 1) % 4
            elif keys[pygame.K_RIGHT] and not self.prev_keys[pygame.K_RIGHT]:
                self.direction = (self.direction + 1) % 4

            if keys[pygame.K_UP] and not self.prev_keys[pygame.K_UP]:
                target_x, target_y = self.rect.topleft
                if self.direction == NORTH:
                    target_y -= self.grid_size
                elif self.direction == SOUTH:
                    target_y += self.grid_size
                elif self.direction == EAST:
                    target_x += self.grid_size
                elif self.direction == WEST:
                    target_x -= self.grid_size

                if self.is_valid_position(target_x, target_y):
                    self.target_pos = (target_x, target_y)
                    self.moving = True
                    self.velocity = self.maxVelocity

        if self.moving:
            if self.direction == NORTH:
                self.set_state(RUN_UP)
            elif self.direction == SOUTH:
                self.set_state(RUN_DOWN)
            elif self.direction == EAST or self.direction == WEST:
                self.set_state(RUN_SIDE)
                self.flipped = (self.direction == WEST)
        else:
            if self.direction == NORTH:
                self.set_state(IDLE_UP)
            elif self.direction == SOUTH:
                self.set_state(IDLE_DOWN)
            elif self.direction == EAST or self.direction == WEST:
                self.set_state(IDLE_SIDE)
                self.flipped = (self.direction == WEST)

        self.prev_keys = keys

    def animate(self):
        now = pygame.time.get_ticks()
        if now - self.last_update > 100:
            self.last_update = now
            self.current_frame = (self.current_frame + 1) % SPRITES[self.state]['columns']
            frame_width = 16
            frame_height = 16
            frame_x = self.current_frame * frame_width
            frame = self.original_image.subsurface(
                pygame.Rect(frame_x, 0, frame_width, frame_height)
            )
            scaled_frame = pygame.transform.scale(
                frame, (frame_width * 2, frame_height * 2)
            )
            if self.flipped:
                scaled_frame = pygame.transform.flip(scaled_frame, True, False)
            self.image = scaled_frame

    def set_state(self, state):
        if self.state != state:
            self.state = state
            self.original_image = SPRITES[state]['imageSrc']
            self.current_frame = 0