# coding=UTF-8
"""
This code has been previously reverted to an old version due to bug fixes
Current Priority:
# TODO: Add waves of entities based on player level:
Level 1 -> 100 score
Level 2 -> 250 score
Level 3 -> 500 score
Level 4 -> 1000 score
Level 5 -> 2000 score
Level 6 -> 3000 score
Level 7 -> 4000 score
Level 8 -> 5000 score
Level 9 -> 10000 score
Level 10 -> 20000 score
Level 11 -> 30000 score
Level 12 -> 40000 score
Level 13 -> 50000 score
Level 14 -> 100000 score
Level 15 -> Victory!!!
High Priority:
# TODO: Fix the wall bug (Re-texturing is required)
# TODO: Retexture and use proper fonts
# TODO: Make the game 'hard'
Medium Priority
# TODO: ANSI fix should belong before libraries
# TODO: Replace the raytracing lighting engine by the pygame-light2d
# TODO: Replace the menu with pygame-menu with minimal importations
Low Priority:
# TODO: Convert from the wav to ogg format for performance
# TODO: Ctypes optimizations for all operating systems
# TODO: Cut unnecessary sounds such as end-music part for more channel availability
Future Priority:
# TODO: Migrate to the CFFI library for performance
# TODO: Cythonize entire 'colorama' library for performance.
# TODO: Multiprocessing for preloading images with cached memory
# TODO: Boost math using numba and CUDA
# TODO: Disable GIL (single-thread) python if using python 3.13
# TODO: Targets tag extension
# TODO: Use the icecream module for debugging (print())
"""

from collections import defaultdict
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading
import ctypes
import sys
import os

# ANSI escape codes for colors
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    OKMAGENTA = '\033[95m'
    OKCYAN = '\033[96m'
    OKYELLOW = '\033[93m'
    OKORANGE = '\033[33m'

    @classmethod
    def get(cls, color_name):
        return getattr(cls, color_name, '')


# List of required libraries
required_libraries = ['pygame', 'numpy', 'asyncio', 'numba', 'colorama']
try:
    from numba import jit, float32, int32, njit, int8
    from numpy.random import randint
    from random import choice
    from pygame.mixer import Sound, SoundType, Channel
    from pygame import Surface, Surface, Vector2, Rect, Mask
    from pygame.rect import RectType
    from pygame.locals import *
    from pygame.font import Font
    from pygame.time import Clock
    from typing import *
    import polars as pl
    import pygame
    import numpy
    import sqlite3
    import asyncio
except Exception as e:
    print(f"{Colors.FAIL}Please install the following libraries: numba, pygame_ce, numpy, asyncio, polars, sqlite3: {Colors.ENDC + str(e)}")
    sys.exit(1)

def memoize(func):
    """
    :param func:
    :return:
    """
    cache: dict[Any, Any] = {}

    @wraps(func)
    def wrapper(*args: object, **kwargs: object):
        key: tuple[tuple[object, ...], frozenset[tuple[str, object]]] = (args, frozenset(kwargs.items()))
        if key not in cache:
            result: object = func(*args, **kwargs)
            cache[key] = result
        return cache[key]

    return wrapper


# Collect resource_path on PyInstaller --add_data for usage inside a executable file
@memoize
@lru_cache(maxsize=None)
def resource_path(relative_path: str) -> str:
    """ Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


# Initialize Pygame and Mixer
pygame.init()

try:
    max_channels = pygame.mixer.get_num_channels()  # Maximum channels depending on the system
    channels = [pygame.mixer.Channel(i) for i in range(max_channels)]
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=max_channels, buffer=512, devicename=None,
                          allowedchanges=AUDIO_ALLOW_FREQUENCY_CHANGE | AUDIO_ALLOW_CHANNELS_CHANGE)
    pygame.mixer.init(frequency=44100, size=-16, channels=max_channels, buffer=512, devicename=None,
                      allowedchanges=AUDIO_ALLOW_FREQUENCY_CHANGE | AUDIO_ALLOW_CHANNELS_CHANGE | AUDIO_ALLOW_FORMAT_CHANGE)
    executor = ThreadPoolExecutor(max_workers=max_channels) # TODO:
    explosion_sound: Sound = pygame.mixer.Sound(resource_path('data/media/explosion.wav'))
    bullet_sound: Sound = pygame.mixer.Sound(resource_path('data/media/wind_bullet.mp3'))
except pygame.error as e:
    print(f"Warning: Mixer initialization failed: {e}")
    pygame.mixer = None
    executor = ThreadPoolExecutor()

# Constants
WIDTH, HEIGHT = 900, 600
HALF_WIDTH = WIDTH // 2
HALF_HEIGHT = HEIGHT // 2
PLAYER_SPEED = 4.0 # 7.5
BULLET_SPEED = 0.5 # 5.0
TARGET_SPEED = 1.0 # 3.0
MAX_TARGETS = 30 # 20 # Set the maximum number of targets
MAX_FROZEN_DURATION = 5000  # in milliseconds
MAX_HASTE_DURATION = 1000  # in milliseconds
MAX_HASTE_MULTIPLIER = 2
SPAWN_INTERVAL = numpy.sin(numpy.radians(60))  # Convert degrees to radians
SHOOT_COOLDOWN = 10  # in seconds
SPECIAL_TARGET_PROBABILITY = 10  # Percentage
DECELERATION = 0.25 # 0.25
FROZEN_TIMER = 0
FROZEN_DURATION = 1000  # in milliseconds
HASTE_TIMER = 0
HASTE_DURATION = 300  # in milliseconds
FPS = 60  # Frames Per Second

# Colors
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (50, 255, 0)
GREY = (250, 250, 250)
YELLOW = (255, 255, 0)
LIGHT_BLUE = (173, 216, 230)
LIGHT_COLOR = (255, 255, 200)
DARK_COLOR = (50, 50, 50)
PLAYER_COLOR = RED  # Same as RED

# List of colors
COLORS = [
    RED,
    (255, 165, 0),  # Orange
    YELLOW,
    (0, 128, 0),  # Green
    BLUE,
    (0, 255, 255),  # Cyan
    (128, 0, 128),  # Purple
    (255, 192, 203),  # Pink
    (238, 130, 238)  # Violet
]

# Initialize player and light
try:
    from lighting import calculate_lighting  # type: ignore
except ImportError as e:
    @memoize
    @jit(int32(float32), nopython=True, fastmath=True, cache=True)
    def calculate_lighting(distance: float32) -> int32:
        max_light: int32 = 255
        min_light: int32 = 225
        attenuation: float32 = 0.01  # Adjust this value for different lighting effects

        # Ensure that distance is positive
        distance = (distance + abs(distance)) / 2.0

        # Calculate the lighting intensity with fast math
        intensity = max_light / (1.0 + attenuation * distance)
        return max(min_light, int(intensity))
    raise ImportWarning("\nPlease run setup.py for the cythonized lighting library.\n Using unoptimized lighting library. The game might be unplayable. \n\n", e)


def mixer_play(sound: Sound):
    """
    Play a sound using Pygame's mixer module with separate threads
    
    :param sound: Pygame Sound object
    """
    if sound:
        def play_sound():
            # Try to find a free channel
            free_channel = next((ch for ch in channels if not ch.get_busy()), None)
            if free_channel is None:
                # If no free channel is found, use the first channel (or any existing one)
                free_channel = channels[0]

            if free_channel:
                free_channel.play(sound)
            else:
                raise RuntimeError("No available mixer channel could be created.")

        # Run play_sound in a separate thread to avoid blocking
        executor.submit(play_sound)

# Create the game window
offscreen_surface: Surface = pygame.Surface((WIDTH, HEIGHT))
screen: Surface = pygame.display.set_mode((WIDTH, HEIGHT), HWSURFACE | DOUBLEBUF, depth=1)
pygame.display.set_caption("Advanced Shooting Game")
pygame.display.set_icon(pygame.image.load(resource_path('data/image/frozen_special_egg2.png')).convert_alpha())

# Windows Fixes
hwnd = pygame.display.get_wm_info()['window']
ctypes.windll.user32.SetForegroundWindow(hwnd)

# Main images
background_image: Surface = pygame.image.load(resource_path('data/image/nebula2.png')).convert()
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))
coin_image: Surface = pygame.image.load(resource_path('data/image/coin.png')).convert_alpha()
coin_image = pygame.transform.scale(coin_image, (40, 40))  # Adjust the size as needed

# Sprite images
# offscreen_surface.blit(background_image, (0, 0))

# Main fonts
menu_font: Font = pygame.font.Font(resource_path('data/fonts/OpenSans-Semibold.ttf'), 36)
version_font: Font = pygame.font.Font(resource_path('data/fonts/OpenSans-Regular.ttf'), 12)
credits_font: Font = pygame.font.Font(resource_path('data/fonts/OpenSans-Bold.ttf'), 12)
title_font: Font = pygame.font.Font(resource_path('data/fonts/OpenSans-ExtraBold.ttf'), 60)
big_message_font: Font = pygame.font.Font(resource_path('data/fonts/OpenSans-Bold.ttf'), 42)
high_score_font: Font = pygame.font.Font(resource_path('data/fonts/Pixel.otf'), 12)


# ___________________________________________ DATA BASE ________________________________________________________________

# Load player data using Polars DataFrames
def load_player_data():
    """ Load player data using Polars (highly efficient). """
    data_path = resource_path("data/game_data.parquet")
    if os.path.exists(data_path):
        return pl.read_parquet(data_path)
    else:
        # Initialize with default data if file does not exist
        data = pl.DataFrame({
            "id": [1],
            "coins": [0],
            "high_score": [0]
        })
        save_player_data(data)
        return data


# Save player data
def save_player_data(df: pl.DataFrame):
    """ Save player data efficiently using Polars. """
    df.write_parquet(resource_path("data/game_data.parquet"))


# Function to get player's coins
def get_player_coins() -> int:
    df = load_player_data()
    return df.filter(pl.col("id") == 1).select("coins").item()


# Function to update player's coins
def update_player_coins(coins: int):
    df = load_player_data()
    df = df.with_columns(pl.lit(coins).alias("coins"))
    save_player_data(df)


# Function to get player's high score
def get_high_score() -> int:
    df = load_player_data()
    return df.filter(pl.col("id") == 1).select("high_score").item()


# Function to update player's high score
def update_high_score(new_high_score: int):
    df = load_player_data()
    df = df.with_columns(pl.lit(new_high_score).alias("high_score"))
    save_player_data(df)


# Function to update both coins and high score (polars optimization)
def update_player_stats(coins: int, score: int):
    df = load_player_data()
    df = df.with_columns([
        pl.lit(coins).alias("coins"),
        pl.lit(score).alias("high_score")
    ])
    save_player_data(df)

# Unused overcomplicated function
"""   
# Fast computation of coins (using Numba for optimization)
@jit(nopython=True, cache=True, fastmath=True)
def compute_total_coins(coins_array: numpy.ndarray) -> int:
    return sum(coins_array) """

# Initialize player database (polars)
def initialize_player():
    coins = get_player_coins()
    if coins is None:
        update_player_coins(0)
    highest_score = get_high_score()
    if highest_score is None:
        update_high_score(0)
    return coins, highest_score


# ______________________________________________ DATA BASE: END ________________________________________________________
PLAYER_SIZE: int = 30

walls: list[Rect] = [
    pygame.Rect(100, 100, 20, 200),
    pygame.Rect(300, 50, 20, 150),
    pygame.Rect(500, 200, 20, 100),
    pygame.Rect(700, 100, 20, 200),
]


async def perform_explosion(explosion_frame_index, explosion_frame_counter, special_egg_destroyed, explosion_frames,
                            explosion_rect, explosion_frame_delay):
    if not special_egg_destroyed or not (0 <= explosion_frame_index < len(explosion_frames)):
        return explosion_frame_index, explosion_frame_counter, special_egg_destroyed

    screen.blit(explosion_frames[explosion_frame_index], explosion_rect.topleft, special_flags=BLEND_ALPHA_SDL2)
    explosion_frame_counter += 1

    # Simulate an asynchronous operation for playing sound
    # loop = asyncio.get_event_loop()
    # await loop.run_in_executor(executor, mixer_play, explosion_sound)
    if pygame.mixer:
        thread = threading.Thread(target=mixer_play, args=(explosion_sound,))
        thread.start()

    if explosion_frame_counter >= explosion_frame_delay:
        explosion_frame_index += 1
        explosion_frame_counter = 0

    # Return the updated values
    return explosion_frame_index, explosion_frame_counter, special_egg_destroyed


async def handle_explosions(explosion_frame_index, explosion_frame_counter, special_egg_destroyed, explosion_frames,
                            explosion_rect, explosion_frame_delay):
    if special_egg_destroyed and 0 <= explosion_frame_index < len(explosion_frames):
        # Perform explosion if conditions are met
        explosion_frame_index, explosion_frame_counter, special_egg_destroyed = await perform_explosion(
            explosion_frame_index, explosion_frame_counter, special_egg_destroyed,
            explosion_frames, explosion_rect, explosion_frame_delay
        )
    elif explosion_frame_index >= len(explosion_frames):
        # Reset values if the explosion frame index is out of bounds
        explosion_frame_index = -1
        special_egg_destroyed = False

    return explosion_frame_index, explosion_frame_counter, special_egg_destroyed


@njit(fastmath=True)
def move_bullets(bullets: numpy.ndarray, bullet_speed: int):
    return bullets - bullet_speed


@njit(fastmath=True)
def check_collisions(bullets: numpy.ndarray, targets: numpy.ndarray):
    hit_indices = []
    for i, target in enumerate(targets):
        for j, bullet in enumerate(bullets):
            if (target[0] <= bullet[0] <= target[0] + target[2]) and \
                    (target[1] <= bullet[1] <= target[1] + target[3]):
                hit_indices.append((i, j))
    return hit_indices


def render_text(screen, font, text, color, position, cache=None):
    if cache is None:
        cache = {}
    if text not in cache:
        cache[text] = font.render(text, True, color)
    screen.blit(cache[text], position)


def play_game():
    """ :return: """
    # Initialize game variables here
    bullets: list[tuple[Any, Any]] = []  # Store bullets as (x, y) tuples
    targets: list[dict[str, Rect | Surface | Surface | int | bool | Mask | tuple[int, int, int]] | dict[
        str, Rect | Surface | Surface | int | bool | tuple[
            int, int, int]]] = []  # Use 'targets' to keep track of the active targets
    score: int = 0
    shoot_cooldown: int8 = 0  # Initialize the shoot cooldown timer
    player_speed: int8 = 0  # Initialize player speed
    max_speed: int8 = 5  # Maximum speed when controls are held
    deceleration: float = DECELERATION  # Deceleration factor for slippery movement
    special_egg_destroyed: bool = False
    iterable = range(10)

    player_img: Surface = pygame.image.load(resource_path('data/image/chicken2.png')).convert_alpha()
    player_img = pygame.transform.rotozoom(player_img, 0, 2.0)
    player_img = pygame.transform.scale(player_img, (50, 75))
    player_rect = pygame.Rect(HALF_WIDTH - PLAYER_SIZE // 2, HALF_HEIGHT - PLAYER_SIZE // 2, PLAYER_SIZE, PLAYER_SIZE)
    player_rect: Rect | RectType = player_img.get_rect() or player_rect
    player_rect.center = (HALF_WIDTH, HEIGHT - 50)
    player_mask = pygame.mask.from_surface(player_img)
    # player_mask_img = player_mask.to_surface()
    normal_target_image: Surface = pygame.image.load(resource_path('data/image/egg.png')).convert_alpha()
    normal_target_image = pygame.transform.scale(normal_target_image, (40, 40))
    special_target_frozen_image: Surface = pygame.image.load(
        resource_path('data/image/frozen_special_egg2.png')).convert_alpha()
    special_target_frozen_image = pygame.transform.scale(special_target_frozen_image, (90, 90))
    special_target_image: Surface = pygame.image.load(resource_path('data/image/special_egg.png')).convert_alpha()
    special_target_image = pygame.transform.scale(special_target_image, (80, 80))
    explosion_frames: list[Surface | Surface] = [
        pygame.transform.scale(pygame.image.load(resource_path('data/image/explosion2.gif')).convert_alpha(),
                               (160, 160)),
        pygame.transform.scale(pygame.image.load(resource_path('data/image/explosion1.gif')).convert_alpha(),
                               (160, 160))]

    explosion_frame_index: int = 0
    explosion_frame_delay: int = 10
    explosion_frame_counter: int = 0
    explosion_rect: Rect = explosion_frames[explosion_frame_index].get_rect()
    clock: Clock = pygame.time.Clock()
    spawn_timer: int = 0  # Initialize a timer for target spawning
    coins, last_highest_score = initialize_player()  # Initialize player's coins

    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_code(coins=coins, score=last_highest_score)

        keys = pygame.key.get_pressed()

        # Increase speed when holding controls
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            if player_speed >= -max_speed:
                player_speed -= 0.5
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            if player_speed <= max_speed:
                player_speed += 0.5
        elif player_speed > 0:
            player_speed -= deceleration
        elif player_speed < 0:
            player_speed += deceleration

        # Apply the background.
        screen.blit(background_image, (0, 0))

        # Apply speed to the player's position
        new_x = player_rect.x + player_speed
        if 0 <= new_x <= WIDTH - player_rect.width:
            player_rect.x = new_x

        # Handle bullet shooting cooldown
        if shoot_cooldown > 0:
            shoot_cooldown -= 1

        # Shoot bullets with cooldown
        if keys[pygame.K_SPACE] and shoot_cooldown == 0:
            bullet = (player_rect.centerx, player_rect.top)  # Store bullets as (x, y) tuples
            bullets.append(bullet)
            if pygame.mixer:
                thread = threading.Thread(target=mixer_play, args=(bullet_sound,))
                thread.start()
            shoot_cooldown = SHOOT_COOLDOWN  # Set the cooldown timer

        # Respond to the display when display is not active
        while not pygame.display.get_active():
            list(map(lambda _: pygame.event.pump() or pygame.time.wait(1), iterable))

        # Move and remove bullets and assuming bullets is a list of (x, y) tuples
        bullets = list(
            map(lambda bullet: (bullet[0], bullet[1] - BULLET_SPEED), filter(lambda bullet: bullet[1] > 0.1, bullets)))

        # list(map(lambda wall: pygame.draw.rect(screen, WHITE, wall), walls))

        # player_center = player_rect.center

        # targets per second (TPS) = (1000 / FPS) / spawn_timer
        # Check if it's time to spawn a new target
        spawn_timer += 1
        if spawn_timer >= 25: # 0.67 TPS.
            if len(targets) < MAX_TARGETS:
                if randint(1, 100) < SPECIAL_TARGET_PROBABILITY:
                    # Create a special target
                    special_target = pygame.Rect(randint(0, WIDTH - 30), 0, special_target_image.get_width(), special_target_image.get_height())
                    special_mask = pygame.mask.from_surface(special_target_image)  # Create mask for the target
                    special_color = BLUE
                    special_health = 3
                    special_frozen = True
                    # Use the loaded special_target_image for the special target
                    targets.append({'rect': special_target, 'mask': special_mask, 'image': special_target_image, 'color': special_color,
                                    'health': special_health, 'score': 3, 'is_special': True,
                                    'coins': randint(3, 10), 'frozen': special_frozen})
                else:
                    # Create a normal target
                    normal_target = pygame.Rect(randint(0, WIDTH - 30), 0, 30, 30)
                    normal_mask = pygame.mask.from_surface(normal_target_image)  # Create mask for the target
                    normal_color = choice(COLORS)
                    normal_health = 1
                    # TODO: Add frozen consequences
                    normal_frozen = False
                    targets.append(
                        {'rect': normal_target, 'color': normal_color, 'mask': normal_mask, 'health': normal_health, 'score': 1,
                         'coins': randint(1, 2), 'image': normal_target_image, 'frozen': normal_frozen, 'is_special': False})
            spawn_timer = 0  # Reset the spawn timer

        # Move and remove targets
        new_targets = []
        for target in targets:
            target['rect'].y += TARGET_SPEED
            if target.get('is_special', False):
                # Apply a spinning animation to the special target
                target['rotation_angle'] = (target.get('rotation_angle', 0) + 2) % 360
                rotated_surface = pygame.transform.rotate(target['image'], target['rotation_angle'])
                target['rect'] = rotated_surface.get_rect(center=target['rect'].center)
                screen.blit(rotated_surface, target['rect'].topleft)
            else:
                screen.blit(target['image'], target['rect'])
                # pygame.draw.rect(screen, target['color'], target['rect'])

            # Generation of new targets and termination of targets which touch the bottom of the screen
            new_targets = list(map(lambda target: target if target['rect'].bottom <= HEIGHT else None, targets))
            new_targets = list(filter(lambda target: target is not None, new_targets))

        targets = new_targets

        # Check for bullet-target collisions and update target health
        bullets_to_remove = []  # New list to store bullets to remove
        for i, bullet in enumerate(bullets):
            for target in targets:
                if target['rect'].colliderect(
                        pygame.Rect(bullet[0] - 2, bullet[1], 4, 10)):  # Create a temporary rect for the bullet
                    bullets_to_remove.append(bullet)
                    if target.get('is_special', False):
                        special_egg_position = target['rect'].center
                        special_egg_destroyed = True
                        explosion_rect.center = special_egg_position
                        explosion_frame_index = 0
                    target['health'] -= 1
                    if target['health'] == 0:
                        targets.remove(target)
                        score += target['score']
                        coins += target.get('coins', 0)

        # Using map and filter to achieve the same effect and remove bullets after the iteration
        bullets = list(map(lambda bullet: bullet, filter(lambda bullet: bullet not in bullets_to_remove, [(x, y) for x, y in bullets if (x, y) not in bullets_to_remove])))

        # Draw everything
        # Pixel-perfect collision detection between player and targets
        for target in targets:
            offset = (target['rect'].x - player_rect.x, target['rect'].y - player_rect.y)
            if player_mask.overlap(target['mask'], offset):
                # Handle pixel-perfect collision (e.g., player gets hit)
                print("Pixel-perfect collision detected!")

        # Draw bullets and targets first
        bullet_rects = list(map(lambda bullet: pygame.Rect(bullet[0] - 2, bullet[1], 4, 10), bullets))
        list(map(lambda rect: pygame.draw.rect(screen, BLUE, rect), bullet_rects))
        explosion_frame_index, explosion_frame_counter, special_egg_destroyed = asyncio.run(handle_explosions(
            explosion_frame_index, explosion_frame_counter, special_egg_destroyed, explosion_frames, explosion_rect,
            explosion_frame_delay
        ))
        # screen.blit(player_img, player_rect)

        # Draw the targets
        for target in targets:
            if target.get('is_special', False):
                if target.get('frozen', True):
                    # Apply a frozen spinning animation to the special target
                    target['image'] = special_target_frozen_image
                    target['rotation_angle'] = (target.get('rotation_angle', 0) + 1) % 360
                    rotated_surface = pygame.transform.rotate(target['image'], target['rotation_angle'])
                    target['rect'] = rotated_surface.get_rect(center=target['rect'].center)
                    screen.blit(rotated_surface, target['rect'].topleft)
                else:
                    # Apply a spinning animation to the special target
                    target['rotation_angle'] = (target.get('rotation_angle', 0) + 2) % 360
                    rotated_surface = pygame.transform.rotate(target['image'], target['rotation_angle'])
                    target['rect'] = rotated_surface.get_rect(center=target['rect'].center)
                    screen.blit(rotated_surface, target['rect'].topleft)
            else:
                if target.get('frozen', True):
                    screen.blit(target['image'], target['rect'])
                    # pygame.draw.rect(screen, target['color'], target['rect'])
                else:
                    screen.blit(target['image'], target['rect'])
                    # pygame.draw.rect(screen, target['color'], target['rect'])

        player_center: Vector2 = pygame.Vector2(player_rect.center)
        lighting_intensity: int = calculate_lighting(player_center.distance_to(pygame.Vector2(HALF_WIDTH, HALF_HEIGHT)))
        player_img_with_lighting: Surface = pygame.Surface(player_img.get_size(), pygame.SRCALPHA)
        player_img_with_lighting.fill((255, 255, 255, lighting_intensity))
        player_img_with_lighting.blit(player_img, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(player_img_with_lighting, player_rect)

        # Display the score and coins
        font: Font = pygame.font.Font(None, 36)
        score_text: Surface | Surface = font.render(f"Score: {str(score)}", True, RED)
        coins_text: Surface | Surface = font.render(str(coins), True, YELLOW)
        screen.blit(coin_image, (10, 50))
        screen.blit(score_text, (10, 10))
        screen.blit(coins_text, (55, 50))

        pygame.display.flip()
        clock.tick_busy_loop(FPS)

    # noinspection PyUnreachableCode
    # Update coins and high score in the database
    last_highest_score = max(last_highest_score, score)
    update_high_score(last_highest_score)
    update_player_stats(coins, last_highest_score)
    update_high_score(last_highest_score)

    # Return the targets and bullets for the next game
    return targets, bullets


def main_settings():
    last_flash_time: int = pygame.time.get_ticks()
    flash_interval: int = 250  # Flash every 250ms
    text_flash: bool = True
    selected_option = -1  # Initialize with no option selected
    options: list[str] = ["3D Settings", "Sound Settings", "More Settings", "Back"]
    slide_bar_value = [0.5, 0.7, 0.3, 0.8]

    while True:
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_code(coins=get_player_coins(), score=get_high_score())
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_option = (selected_option - 1) % len(options)
                if event.key == pygame.K_DOWN:
                    selected_option = (selected_option + 1) % len(options)
                if event.key == pygame.K_RETURN:
                    if selected_option == len(options) - 1:
                        return main_menu()  # Go back to the main menu
                    elif selected_option == len(options) - 2:
                        print("More settings")
                    elif selected_option == len(options) - 3:
                        print("Sound settings")
                    elif selected_option == len(options) - 4:
                        print("3D settings")
                    # Handle other settings options here (e.g., open submenus or toggle settings)
            # Handle mouse click events
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    x, y = event.pos
                    for i, option in enumerate(options):
                        text = menu_font.render(option, True, RED)
                        x_text = HALF_WIDTH - text.get_width() // 2
                        y_text = 300 + i * 60
                        option_rect = pygame.Rect(x_text - 10, y_text, text.get_width() + 20, text.get_height() + 3)
                        if option_rect.collidepoint(x, y):
                            if i == 0:
                                return print("3D Settings")
                            elif i == 1:
                                return print("Sound Settings")
                            elif i == 2:
                                return print("More Settings")
                            elif i == 3:
                                return main_menu()
        screen.fill(WHITE)
        screen.blit(background_image, (0, 0))

        title_text = title_font.render("Settings", True, BLUE)
        title_rect = title_text.get_rect()
        title_rect.centerx = HALF_WIDTH
        title_rect.y = 50
        screen.blit(title_text, title_rect)

        # Create option rectangles for mouse interaction
        option_rects = []
        for i, option in enumerate(zip(options, slide_bar_value)):
            text_color = GREEN if i == selected_option and text_flash else RED
            text = menu_font.render(option, True, text_color)
            x = HALF_WIDTH - text.get_width() // 2
            y = 300 + i * 60

            option_rect = pygame.Rect(x - 10, y, text.get_width() + 20, text.get_height() + 3)
            option_rects.append(option_rect)

            # Check if the mouse cursor is over the option and highlight it
            list(map(lambda r: pygame.draw.rect(screen, LIGHT_BLUE, option_rect,
                                                border_radius=10) if option_rect.collidepoint(
                pygame.mouse.get_pos()) else None, [i]))

            list(map(lambda r: pygame.draw.rect(screen, BLUE, option_rect, border_radius=10,
                                                width=2) if i == selected_option else None, [i]))

            # Draw a vertical slider next to each option
            slider_width = 10
            slider_height = 100
            slider_x = WIDTH // 4 * 3
            slider_y = y + text.get_height() // 2 - slider_height // 2
            pygame.draw.rect(screen, DARK_COLOR, (slider_x, slider_y, slider_width, slider_height))
            slider_position = int(slide_bar_value[i] * (slider_height - 20)) + slider_y + 10
            pygame.draw.rect(screen, BLUE, (slider_x, slider_position, slider_width, 10))

            screen.blit(text, (x, y))

        if current_time - last_flash_time >= flash_interval:
            last_flash_time = current_time
            text_flash = not text_flash

        pygame.display.flip()
        pygame.time.Clock().tick(-1)


@memoize
@lru_cache(maxsize=None)
def main_menu():
    selected_option: int = -1  # Initialize with no option selected
    options: list[str] = ["Play", "Settings", "Quit"]
    last_flash_time: int = pygame.time.get_ticks()
    flash_interval: int = 250  # Flash every 250ms
    text_flash: bool = True
    title_y: int = 10  # Initial vertical position of the title
    coins = get_player_coins()
    last_high_score: int = get_high_score()
    high_score_text: Surface | Surface = high_score_font.render(f'HIGH SCORE :{str(last_high_score)}', True,
                                                                LIGHT_BLUE)
    high_score_rect: Rect | RectType = high_score_text.get_rect()
    credits_text: Surface | Surface = credits_font.render("Credits: ", True, GREEN)
    credits_rect: Rect | RectType = credits_text.get_rect()
    credits_rect.topleft = (10, HEIGHT - 25)
    version_text: Surface | Surface = version_font.render("Version: 1.4-pre", True, BLUE)
    version_rect: Rect | RectType = version_text.get_rect()
    version_rect.topright = (WIDTH - 10, HEIGHT - 25)

    while True:
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_code(coins=coins, score=last_high_score)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_option = (selected_option - 1) % len(options)
                if event.key == pygame.K_DOWN:
                    selected_option = (selected_option + 1) % len(options)
                if selected_option == 0:
                    if event.key == pygame.K_RETURN:
                        return play_game()
                elif selected_option == 1:
                    if event.key == pygame.K_RETURN:
                        return main_settings()
                elif selected_option == 2:
                    if event.key == pygame.K_RETURN:
                        exit_code(coins=coins, score=last_high_score)
            # Handle mouse click events
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    x, y = event.pos
                    for i, option in enumerate(options):
                        text = menu_font.render(option, True, RED)
                        x_text = WIDTH // 2 - text.get_width() // 2
                        y_text = 300 + i * 60
                        option_rect = pygame.Rect(x_text - 10, y_text, text.get_width() + 20, text.get_height() + 3)
                        if option_rect.collidepoint(x, y):
                            if i == 0:
                                return play_game()
                            elif i == 1:
                                return main_settings()
                            elif i == 2:
                                exit_code(coins=coins, score=last_high_score)
        screen.fill(WHITE)
        screen.blit(background_image, (0, 0))
        title_y += 1 * numpy.sin(current_time/ 60)
        title_text = title_font.render("Chicken Cube Destroyers", True, BLUE)
        title_rect = title_text.get_rect()
        title_rect.centerx = WIDTH // 2
        title_rect.y = title_y

        # Create option rectangles for mouse interaction
        option_rects = []
        for i, option in enumerate(options):
            text_color = GREEN if i == selected_option and text_flash else RED
            text = menu_font.render(option, True, text_color)
            x = WIDTH // 2 - text.get_width() // 2
            y = 300 + i * 60

            option_rect = pygame.Rect(x - 10, y, text.get_width() + 20, text.get_height() + 3)
            option_rects.append(option_rect)

            # Check if the mouse cursor is over the option and highlight it
            list(map(lambda i: pygame.draw.rect(screen, LIGHT_BLUE, option_rect,
                                                border_radius=10) if option_rect.collidepoint(
                pygame.mouse.get_pos()) else None, [i]))

            list(map(lambda i: pygame.draw.rect(screen, BLUE, option_rect, border_radius=10,
                                                width=2) if i == selected_option else None, [i]))

            screen.blit(text, (x, y))

        if current_time - last_flash_time >= flash_interval:
            last_flash_time = current_time
            text_flash = not text_flash

        screen.blit(title_text, title_rect)
        screen.blit(credits_text, credits_rect)
        screen.blit(version_text, version_rect)
        screen.blit(high_score_text, high_score_rect)

        pygame.display.flip()


@lru_cache(maxsize=None)
def exit_code(coins: int , score: int):
    """
    :param score:
    :type coins: int
    """
    print(
        f"{Colors.OKMAGENTA}{Colors.BOLD}Saving game statistics{Colors.ENDC} {Colors.OKCYAN}{Colors.BOLD}Quitting pygame windows{Colors.ENDC}")
    update_high_score(score)
    update_player_coins(coins)
    update_player_stats(coins, high_score)
    pygame.display.quit()
    pygame.mixer.quit()
    pygame.font.quit()
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    high_score = get_high_score()
    print(high_score)
    main_menu()
