def check_collision(forklift, game_width, game_height):
    if (forklift.sprite.rect.top < 0):
        forklift.sprite.rect.top = 0
    if (forklift.sprite.rect.bottom > game_height):
        forklift.sprite.rect.bottom = game_height
    if (forklift.sprite.rect.left < 0):
        forklift.sprite.rect.left = 0
    if (forklift.sprite.rect.right > game_width):
        forklift.sprite.rect.right = game_width