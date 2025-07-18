import pygame

class Shelf(pygame.sprite.Sprite):
    def __init__(self, pos, length, start_image, middle_image, end_image):
        super().__init__()
        self.image = pygame.Surface((length * 32, 32), pygame.SRCALPHA)
        x_offset = 0
        # Dodaj startowy kafelek
        self.image.blit(start_image, (x_offset, 0))
        x_offset += 32
        # Dodaj środkowe kafelki
        for _ in range(length - 2):  
            self.image.blit(middle_image, (x_offset, 0))
            x_offset += 32
        # Dodaj końcowy kafelek
        self.image.blit(end_image, (x_offset, 0))
        self.rect = self.image.get_rect(topleft=pos)
        self.capacity = length 