import pygame

IDLE_DOWN = 'IDLE_DOWN'
IDLE_SIDE = 'IDLE_SIDE'
IDLE_UP = 'IDLE_UP'
RUN_DOWN = 'RUN_DOWN'
RUN_SIDE = 'RUN_SIDE'
RUN_UP = 'RUN_UP'

idle_down_image = pygame.image.load('./assets/character/idle_down.png')
idle_side_image = pygame.image.load('./assets/character/idle_side.png')
idle_up_image = pygame.image.load('./assets/character/idle_up.png')
run_down_image = pygame.image.load('./assets/character/run_down.png')
run_side_image = pygame.image.load('./assets/character/run_side.png')
run_up_image = pygame.image.load('./assets/character/run_up.png')

SPRITES = {
    IDLE_DOWN: {
        'state': 'IDLE_DOWN',
        'imageSrc': idle_down_image,
        'columns': 4,
    },
    IDLE_SIDE: {
        'state': 'IDLE_SIDE',
        'imageSrc': idle_side_image,
        'columns': 4,
    },
    IDLE_UP: {
        'state': 'IDLE_UP',
        'imageSrc': idle_up_image,
        'columns': 4,
    },
    RUN_DOWN: {
        'state': 'RUN_DOWN',
        'imageSrc': run_down_image,
        'columns': 8,
    },
    RUN_SIDE: {
        'state': 'RUN_SIDE',
        'imageSrc': run_side_image,
        'columns': 8,
    },
    RUN_UP: {
        'state': 'RUN_UP',
        'imageSrc': run_up_image,
        'columns': 8,
    },
}