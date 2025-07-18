import pygame
import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn


class Chest(pygame.sprite.Sprite):
    CHEST_TYPES = {
        'big_gold': {
            'image': {
                './assets/chests/gold_chest_big.png',
                './assets/chests/Gold Chest.png',
                './assets/chests/Patry Chest.png',
                './assets/chests/big_gold2.png',
            }
        },
        'small_gold': {
            'image': {
                './assets/chests/gold_chest_small.png',
                './assets/chests/WoodenChest_Gold_animation.png',
                './assets/chests/gold_small2.png',
                './assets/chests/small_gold3.png',
            }
        },
        'small_iron': {
            'image': {
                './assets/chests/iron_chest_small.png',
                './assets/chests/silver_chest_1.png',
                './assets/chests/silver_chest_small2.png',
                './assets/chests/small_silver4.png',
            }
        },
        'big_iron': {
            'image': {
                './assets/chests/iron_chest_big.png',
                './assets/chests/Demon Chest.png',
                './assets/chests/Holy Chest.png',
                './assets/chests/Silver Chest.png',
            }
        }
    }
 
    def __init__(self, pos, chest_type='iron',chest_size='small', time=0, priority=0, weight=0.0,on_conter=True):
        def add_noise(img):
            arr = pygame.surfarray.array3d(img)
            arr = np.transpose(arr, (1, 0, 2))

            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                arr = cv2.rotate(arr, {
                    90: cv2.ROTATE_90_CLOCKWISE,
                    180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE
                }[angle])

            noise_type = random.choice(["gauss", "s&p"])
            if noise_type == "gauss":
                row, col, ch = arr.shape
                mean = 0
                sigma = 25
                gauss = np.random.normal(mean, sigma, (row, col, ch)).astype('uint8')
                noisy = cv2.add(arr, gauss)
            elif noise_type == "s&p":
                s_vs_p = 0.5
                amount = 0.04
                noisy = np.copy(arr)
                num_salt = np.ceil(amount * arr.size * s_vs_p)
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in arr.shape]
                noisy[tuple(coords)] = 255
 
                num_pepper = np.ceil(amount * arr.size * (1. - s_vs_p))
                coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in arr.shape]
                noisy[tuple(coords)] = 0
 
            noisy = np.transpose(noisy, (1, 0, 2))
            return pygame.surfarray.make_surface(noisy)
        super().__init__()
        self.width = 32
        self.height = 32
        self.time = time
        self.priority = priority
        self.weight = weight
        self.chest_size = chest_size
        self.chest_type = chest_type
        self.pos = pos
        chest_sprit= chest_size+'_'+chest_type
        self.on_conter=on_conter




        chest_data = self.CHEST_TYPES[chest_sprit]
 
 
        image_path = random.choice(list(chest_data['image']))
        self.original_image = pygame.image.load(image_path)
        self.original_image=add_noise(self.original_image)

        class MyModel(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 64)
                self.relu = nn.Tanh()
                self.fc2 = nn.Linear(64, 4)
                self.logsoftmax = nn.LogSoftmax(dim=1)

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.logsoftmax(self.fc2(x))
                return x
        
        model = MyModel(input_dim=768)
        model.load_state_dict(torch.load('model_scratch_new_abs2.pth', map_location=torch.device('cpu')))
        model.eval()

        def pygame_surface_to_tensor(surface):
            array = pygame.surfarray.array3d(pygame.transform.scale(surface, (16, 16))).astype(np.float32)
            array = np.transpose(array, (2, 0, 1))  
            array /= 255.0  
            array = array.reshape(-1)  
            return torch.tensor(array).unsqueeze(0)  

        input_tensor = pygame_surface_to_tensor(self.original_image)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        print(f"Predicted class for chest: {predicted_class}")
        print(chest_size,chest_type) 

        if predicted_class==0:
            self.chest_size = 'big'
            self.chest_type = 'gold' 
        elif predicted_class==1:
            self.chest_size = 'big'
            self.chest_type = 'iron'   
        elif predicted_class==2:
            self.chest_size = 'small'
            self.chest_type = 'gold'     
        elif predicted_class==3:
            self.chest_size = 'small'
            self.chest_type = 'iron'                     
        

        self.scaled_image = pygame.transform.scale(
            self.original_image,
            (self.original_image.get_width() * 2, self.original_image.get_height() * 2)
        )
        self.image = self.scaled_image
        self.rect = self.image.get_rect(topleft=(pos[0], pos[1]))