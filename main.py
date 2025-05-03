# coding=UTF-8
"""
Current Priority:
# TODO: PIXEL-PERFECT Target bullet collision
# TODO: Add waves of entities based on player level:

High Priority:
# TODO: Feature every egg a shake effect on hit
# TODO: Fix the wall bug (Re-texturing is required)
# TODO: Retexture and use proper fonts
# TODO: Use Delta Time
Medium Priority
# TODO: Extend the screen for in-game HUD (hotbar, coins, hearts)
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
# TODO: Use the icecream module for debugging (print())

# DEBUG: Interlacing is enabled for egg vector sprites; can cause problems or low quality.
"""

from collections import defaultdict
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor
import platform
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
required_libraries = ['pygame_ce', 'numpy', 'numba', 'colorama']
try:
    from numba import jit, float32, int32, njit, int8
    from numpy.random import randint
    from random import choice
    from pygame.mixer import Sound, SoundType, Channel
    from pygame import Surface, Surface, Vector2, Rect, Mask, freetype
    from pygame.rect import RectType
    from pygame.locals import *
    from pygame.font import Font, FontType
    from pygame.time import Clock
    from typing import *
    import polars as pl
    import pygame
    import numpy
except Exception as e:
    print(
        f"{Colors.FAIL}Please install the following libraries: numba, pygame_ce, numpy, polars: {Colors.ENDC + str(e)}")
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


# Collect resource_path on PyInstaller --add_data for usage inside an executable file
@memoize
@lru_cache(maxsize=None)
def resource_path(relative_path: str) -> str:
    """ Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


# Initialize Pygame, Freetype font and Mixer
pygame.freetype.init(1000, 56)
pygame.init()

try:
    max_channels = pygame.mixer.get_num_channels()  # Maximum channels depending on the system
    channels = [pygame.mixer.Channel(i) for i in range(max_channels)]
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=max_channels, buffer=512, devicename=None,
                          allowedchanges=AUDIO_ALLOW_FREQUENCY_CHANGE | AUDIO_ALLOW_CHANNELS_CHANGE)
    pygame.mixer.init(frequency=44100, size=-16, channels=max_channels, buffer=512, devicename=None,
                      allowedchanges=AUDIO_ALLOW_FREQUENCY_CHANGE | AUDIO_ALLOW_CHANNELS_CHANGE | AUDIO_ALLOW_FORMAT_CHANGE)
    executor = ThreadPoolExecutor(max_workers=max_channels)
    explosion_sound: Sound = pygame.mixer.Sound(resource_path('data/media/explosion.wav'))
    bullet_sound: Sound = pygame.mixer.Sound(resource_path('data/media/wind_bullet.mp3'))
except pygame.error as e:
    print(f"Warning: Mixer initialization failed: {e}")
    pygame.mixer = None
    executor = ThreadPoolExecutor()

# Constants
WIDTH, HEIGHT = 900, 600
HALF_WIDTH, HALF_HEIGHT = WIDTH // 2, HEIGHT // 2
PLAYER_SPEED = 4.0  # 7.5
PLAYER_HEALTH = 5  # Player starting health
PLAYER_MAX_HEALTH = 5  # Maximum health
BULLET_SPEED = 0.5  # 5.0
TARGET_SPEED = 1.0  # 3.0
MAX_TARGETS = 30  # 20 # Set the maximum number of targets
MAX_FROZEN_DURATION = 5000  # in milliseconds
MAX_HASTE_DURATION = 1000  # in milliseconds
MAX_HASTE_MULTIPLIER = 2
SPAWN_INTERVAL = numpy.sin(numpy.radians(60))  # Convert degrees to radians
SHOOT_COOLDOWN = 10  # in seconds
SPECIAL_TARGET_PROBABILITY = 80  # Percentage
DECELERATION = 0.25  # 0.25
FROZEN_TIMER = 0
FROZEN_DURATION = 1000  # in milliseconds
HASTE_TIMER = 0
HASTE_DURATION = 300  # in milliseconds
FPS = 60  # Frames Per Second
HEALTH_BAR_PADDING = 10  # Padding from the edge of the screen
HEART_SPACING = 5  # Spacing between hearts

# Colors
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
GREEN = (50, 255, 0)
DARK_GREEN = (0, 128, 0)
GREY = (250, 250, 250)
YELLOW = (255, 255, 0)
LIGHT_BLUE = (173, 216, 230)
LIGHT_COLOR = (255, 255, 200)
DARK_COLOR = (50, 50, 50)
PLAYER_COLOR = RED

# List of colors
COLORS = [
    RED,
    ORANGE,
    YELLOW,
    GREEN,
    DARK_GREEN,
    BLUE,
    (0, 255, 255),  # Cyan
    (128, 0, 128),  # Purple
    (255, 192, 203),  # Pink
    (238, 130, 238)  # Violet
]

# Create the game window
offscreen_surface: Surface = pygame.Surface((WIDTH, HEIGHT))
screen: Surface = pygame.display.set_mode((WIDTH, HEIGHT), HWSURFACE | DOUBLEBUF, depth=1)
pygame.display.set_caption("Advanced Shooting Game")
pygame.display.set_icon(pygame.image.load(resource_path('data/textures/frozen_special_egg2.png')).convert_alpha())

# Cross-platform compatibility for high-resolution timing
if platform.system() == "Windows":
    import ctypes

    hwnd = pygame.display.get_wm_info()['window']
    ctypes.windll.user32.SetForegroundWindow(hwnd)
    kernel32 = ctypes.windll.kernel32
    QueryPerformanceCounter = ctypes.c_int64()
    QueryPerformanceFrequency = ctypes.c_int64()
    kernel32.QueryPerformanceFrequency(ctypes.byref(QueryPerformanceFrequency))
    perf_frequency = QueryPerformanceFrequency.value


    def perf_counter():
        kernel32.QueryPerformanceCounter(ctypes.byref(QueryPerformanceCounter))
        return QueryPerformanceCounter.value / perf_frequency
else:
    from timeit import default_timer as perf_counter

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


    raise ImportWarning("".join([e, "\n\nPlease run setup.py for the cythonized lighting library.\n Using unoptimized lighting library. The game might be unplayable. \n\n"]))


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
                free_channel = channels[0]

            if free_channel:
                free_channel.play(sound)
            else:
                raise RuntimeError("No available mixer channel could be created.")

        # Run play_sound in a separate thread to avoid blocking
        executor.submit(play_sound)


# Main images
background_image: Surface = pygame.image.load(resource_path('data/textures/nebula2.png')).convert()
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))
coin_image: Surface = pygame.image.load(resource_path('data/textures/coin.png')).convert_alpha()
coin_image = pygame.transform.scale(coin_image, (40, 40))  # Adjust the size as needed
heart_image = pygame.transform.scale(
    pygame.image.load(resource_path(os.path.join("data", "textures", "heart.png"))).convert_alpha(), (32, 32))

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
    # Initialize with default data if file does not exist
    data = pl.DataFrame({
        "id": [1],
        "coins": [0],
        "high_score": [0]
    })
    save_player_data(data)
    return data


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

explosion_frames: list[Surface | Surface] = [
    pygame.transform.scale(pygame.image.load(resource_path('data/textures/explosion2.gif')).convert_alpha(),
                           (160, 160)),
    pygame.transform.scale(pygame.image.load(resource_path('data/textures/explosion1.gif')).convert_alpha(),
                           (160, 160))]


@njit(fastmath=True)
def move_bullets(bullets: numpy.ndarray, bullet_speed: int):
    return bullets - bullet_speed


@njit(fastmath=True)
def check_collisions(bullets: numpy.ndarray, targets: numpy.ndarray):
    hit_indices = []
    for i, target in enumerate(targets):
        hit_indices.extend(
            (i, j)
            for j, bullet in enumerate(bullets)
            if (target[0] <= bullet[0] <= target[0] + target[2])
            and (target[1] <= bullet[1] <= target[1] + target[3])
        )
    return hit_indices


def play_game(health, max_health):
    """ :return: """
    # Initialize game variables here
    bullets: list[tuple[Any, Any]] = []  # Store bullets as (x, y) tuples
    targets: list[dict[str, Rect | Surface | Surface | int | bool | Mask | tuple[int, int, int]] | dict[
        str, Rect | Surface | Surface | int | bool | tuple[
            int, int, int]]] = []  # Use 'targets' to keep track of the active targets
    explosions = []  # List to hold explosion states
    score: int = 0
    shoot_cooldown: int8 = 0  # Initialize the shoot cooldown timer
    player_speed: int8 = 0  # Initialize player speed
    max_speed: int8 = 5  # Maximum speed when controls are held
    deceleration: float = DECELERATION  # Deceleration factor for slippery movement
    iterable = range(10)
    last_spawn_time = perf_counter()
    player_img: Surface = pygame.image.load(resource_path('data/textures/chicken2.png')).convert_alpha()
    player_img = pygame.transform.rotozoom(player_img, 0, 2.0)
    player_img = pygame.transform.scale(player_img, (50, 75))
    player_rect = pygame.Rect(HALF_WIDTH - PLAYER_SIZE // 2, HALF_HEIGHT - PLAYER_SIZE // 2, PLAYER_SIZE, PLAYER_SIZE)
    player_rect: Rect | RectType = player_img.get_rect() or player_rect
    player_rect.center = (HALF_WIDTH, HEIGHT - 50)
    player_mask = pygame.mask.from_surface(player_img)
    # player_mask_img = player_mask.to_surface()
    normal_target_image: Surface = pygame.image.load(resource_path('data/textures/egg1.png')).convert_alpha()
    normal_target_image = pygame.transform.scale(normal_target_image, (37, 51))
    special_target_frozen_image: Surface = pygame.image.load(
        resource_path('data/textures/frozen_special_egg2.png')).convert_alpha()
    special_target_frozen_image = pygame.transform.scale(special_target_frozen_image, (90, 90))
    special_target_image: Surface = pygame.image.load(resource_path('data/textures/special_egg.png')).convert_alpha()
    special_target_image = pygame.transform.scale(special_target_image, (80, 80))
    clock: Clock = pygame.time.Clock()
    coins, last_high_score = initialize_player()  # Initialize player's coins
    font = pygame.freetype.Font(None, 32)
    font.antialiased = False  # Increases performance with pixelation
    font.use_bitmap_strikes = False  # FPS Optimization
    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                last_highest_score = max(last_high_score, score)
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
            mixer_play(bullet_sound)
            shoot_cooldown = SHOOT_COOLDOWN  # Set the cooldown timer

        # Respond to the display when display is not active
        while not pygame.display.get_active():
            list(map(lambda _: pygame.event.pump() or pygame.time.wait(1), iterable))

        # Move and remove bullets and assuming bullets is a list of (x, y) tuples
        bullets = list(
            map(lambda bullet: (bullet[0], bullet[1] - BULLET_SPEED), filter(lambda bullet: bullet[1] > 0.1, bullets)))

        # targets per second (TPS) = (1000 / FPS) / spawn_timer
        # Check if it's time to spawn a new target
        current_time = perf_counter()
        if current_time - last_spawn_time >= SPAWN_INTERVAL:  # 0.67 TPS.
            last_spawn_time = current_time
            if len(targets) < MAX_TARGETS:
                if randint(1, 100) < SPECIAL_TARGET_PROBABILITY:
                    # Create a special target
                    special_target = pygame.Rect(randint(0, WIDTH - 30), 0, special_target_image.get_width(),
                                                 special_target_image.get_height())
                    special_mask = pygame.mask.from_surface(special_target_image)  # Create mask for the target
                    special_color = BLUE
                    special_health = 3
                    special_frozen = True
                    # Use tags for the special target
                    targets.append({'rect': special_target, 'mask': special_mask, 'image': special_target_image,
                                    'color': special_color, 'hit': False,
                                    'health': special_health, 'score': 3, 'is_special': True,
                                    'coins': randint(3, 10), 'frozen': special_frozen})
                else:
                    # Create a normal target
                    normal_target = pygame.Rect(randint(0, WIDTH - 30), 0, normal_target_image.get_width(), normal_target_image.get_height())
                    normal_mask = pygame.mask.from_surface(normal_target_image) # Create mask for the target
                    normal_color = choice(COLORS)
                    normal_health = 1
                    # TODO: Add frozen consequences
                    normal_frozen = False
                    targets.append(
                        {'rect': normal_target, 'color': normal_color, 'mask': normal_mask, 'health': normal_health,
                         'score': 1, 'hit': False,
                         'coins': randint(1, 2), 'image': normal_target_image, 'frozen': normal_frozen,
                         'is_special': False})

        # Move and remove targets
        new_targets = []
        # Draw the targets
        for target in targets:
            target['rect'].y += TARGET_SPEED
            if target.get('is_special', False):
                if target.get('frozen', True):
                    # Apply a frozen spinning animation to the special target
                    target['image'] = special_target_frozen_image
                    target['rotation_angle'] = (target.get('rotation_angle', 0) + 2) % 360
                    rotated_surface = pygame.transform.rotate(target['image'], target['rotation_angle']).convert_alpha()
                    target['rect'] = rotated_surface.get_rect(center=target['rect'].center)
                    screen.blit(rotated_surface, target['rect'].topleft)
                else:
                    # Apply a spinning animation to the special target
                    target['rotation_angle'] = (target.get('rotation_angle', 0) + 1) % 360
                    rotated_surface = pygame.transform.rotate(target['image'], target['rotation_angle']).convert_alpha()
                    target['rect'] = rotated_surface.get_rect(center=target['rect'].center)
                    screen.blit(rotated_surface, target['rect'].topleft)
            else:
                # pygame.draw.rect(screen, target['color'], target['rect'])
                screen.blit(target['image'], target['rect'])


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
                    target['health'] -= 1
                    if target['health'] == 0:  # TODO: Use <= instead of == when damage is larger than 1
                        if target.get('is_special', False):
                            explosions.append({
                                'frame': 0,
                                'delay': 0.375,
                                'active': True,
                                'rect': pygame.Rect(0, 0, explosion_frames[0].get_width(),
                                                    explosion_frames[0].get_height()),
                                'last_time': perf_counter(),
                                'accumulated_time': 0
                            })

                            # Set precise positioning at target's center
                            explosions[-1]['rect'].center = target['rect'].center
                            print(f"DEBUG: Target Rect: {target['rect']} | Center: {target['rect'].center}")
                        targets.remove(target)
                        score += target['score']
                        coins += target.get('coins', 0)
                        break

        # Using map and filter to achieve the same effect and remove bullets after the iteration
        bullets = list(map(lambda bullet: bullet, filter(lambda bullet: bullet not in bullets_to_remove,
                                                         [(x, y) for x, y in bullets if
                                                          (x, y) not in bullets_to_remove])))

        # Cache current time for performance
        current_time = perf_counter()
        active_explosions = []

        # Draw explosions
        for explosion in explosions:
            if not explosion['active']:
                continue

            # Draw the current frame of the explosion at its precise position
            screen.blit(explosion_frames[explosion['frame']], explosion['rect'].topleft)

            # Calculate time elapsed since last frame update
            elapsed_time = current_time - explosion['last_time']
            explosion['accumulated_time'] += elapsed_time
            explosion['last_time'] = current_time

            # Update frame if enough time has elapsed
            if explosion['accumulated_time'] >= explosion['delay']:
                explosion['frame'] += 1
                explosion['accumulated_time'] = 0

                # Check if the animation is complete
                if explosion['frame'] >= len(explosion_frames):
                    explosion['active'] = False  # Mark as inactive if done
                else:
                    active_explosions.append(explosion)  # Keep active explosions
            else:
                active_explosions.append(explosion)  # Keep ongoing explosions

        # Update list to only active explosions
        explosions = active_explosions

        # Pixel-perfect collision detection between player and targets
        for target in targets:
            if target['hit']: # Cancel multiple triggering for same targets
                continue

            offset = (target['rect'].x - player_rect.x, target['rect'].y - player_rect.y)
            if player_rect.colliderect(target['rect']):
                if player_mask.overlap(target['mask'], offset):
                    target['hit'] = True
                    health = health - 1
                    print("DEBUG: Pixel-perfect collision detected!")


        for i in range(max_health):
            # Calculate the position of each heart
            heart_x = WIDTH - (i + 1) * (heart_image.get_width() + HEART_SPACING) - HEALTH_BAR_PADDING
            heart_y = HEALTH_BAR_PADDING

            if i < health:
                # Draw a full heart for each health point
                screen.blit(heart_image, (heart_x, heart_y))
            else:
                empty_heart_image = heart_image.copy()
                empty_heart_image.set_alpha(25)  # Make empty hearts more transparent
                screen.blit(empty_heart_image, (heart_x, heart_y))

        # Draw bullets and targets first
        bullet_rects = list(map(lambda bullet: pygame.Rect(bullet[0] - 2, bullet[1], 4, 10), bullets))
        list(map(lambda rect: pygame.draw.rect(screen, BLUE, rect), bullet_rects))
        # screen.blit(player_img, player_rect)

        player_center: Vector2 = pygame.Vector2(player_rect.center)
        lighting_intensity: int = calculate_lighting(player_center.distance_to(pygame.Vector2(HALF_WIDTH, HALF_HEIGHT)))
        player_img_with_lighting: Surface = pygame.Surface(player_img.get_size(), pygame.SRCALPHA)
        player_img_with_lighting.fill((255, 255, 255, lighting_intensity))
        player_img_with_lighting.blit(player_img, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(player_img_with_lighting, player_rect)

        # Display the score and coins without antialiasing
        screen.blit(coin_image, (10, 50))
        font.render_to(screen, (10, 10), f"Score: {str(score)}", RED)
        font.render_to(screen, (55, 50), str(coins), YELLOW)

        pygame.display.flip()
        clock.tick_busy_loop(FPS)


@memoize
@lru_cache(maxsize=None)
def main_menu():
    selected_option: int = -1
    options: list[str] = ["Play", "Settings", "Quit"]
    last_flash_time: int = pygame.time.get_ticks()
    flash_interval: int = 250
    text_flash: bool = True
    title_y: int = 10
    title_speed: float = 1.0
    title_direction = 1
    coins = get_player_coins()
    last_high_score: int = get_high_score()

    high_score_text: Surface = high_score_font.render(f'HIGH SCORE: {str(last_high_score)}', True, LIGHT_BLUE)
    high_score_rect: Rect = high_score_text.get_rect(topright=(60, 0))
    credits_text: Surface = credits_font.render("Credits: ", True, GREEN)
    credits_rect: Rect = credits_text.get_rect(topleft=(10, HEIGHT - 25))
    version_text: Surface = version_font.render("Version: 1.5-pre-alpha", True, BLUE)
    version_rect: Rect = version_text.get_rect(topright=(WIDTH - 10, HEIGHT - 25))

    while True:
        delta_time = pygame.time.get_ticks() - last_flash_time
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_code(coins=coins, score=last_high_score)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_option = (selected_option - 1) % len(options)
                if event.key == pygame.K_DOWN:
                    selected_option = (selected_option + 1) % len(options)
                if event.key == pygame.K_RETURN:
                    if selected_option == 0:
                        return play_game(PLAYER_HEALTH, PLAYER_MAX_HEALTH)
                    elif selected_option == 1:
                        return play_game(PLAYER_HEALTH, PLAYER_MAX_HEALTH)
                    elif selected_option == 2:
                        exit_code(coins=coins, score=last_high_score)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                for i, option in enumerate(options):
                    text = menu_font.render(option, True, RED)
                    x_text = WIDTH // 2 - text.get_width() // 2
                    y_text = 300 + i * 60
                    option_rect = pygame.Rect(x_text - 10, y_text, text.get_width() + 20, text.get_height() + 3)
                    if option_rect.collidepoint(x, y):
                        if i == 0:
                            return play_game(PLAYER_HEALTH, PLAYER_MAX_HEALTH)
                        elif i == 1:
                            return play_game(PLAYER_HEALTH, PLAYER_MAX_HEALTH)
                        elif i == 2:
                            exit_code(coins=coins, score=last_high_score)

        screen.fill(WHITE)
        screen.blit(background_image, (0, 0))

        # Oscillate title_y slowly
        if title_y <= 10 or title_y >= 100:
            title_direction *= -1
        title_y += title_direction * title_speed

        title_text = title_font.render("Chicken Cube Destroyers", True, BLUE)
        title_rect = title_text.get_rect(centerx=WIDTH // 2, y=title_y)

        option_rects = []
        for i, option in enumerate(options):
            text_color = GREEN if i == selected_option and text_flash else RED
            text = menu_font.render(option, True, text_color)
            x = WIDTH // 2 - text.get_width() // 2
            y = 300 + i * 60
            option_rect = pygame.Rect(x - 10, y, text.get_width() + 20, text.get_height() + 3)
            option_rects.append(option_rect)

            pygame.draw.rect(screen, LIGHT_BLUE, option_rect, border_radius=10) if option_rect.collidepoint(
                pygame.mouse.get_pos()) else None
            pygame.draw.rect(screen, BLUE, option_rect, border_radius=10, width=2) if i == selected_option else None
            screen.blit(text, (x, y))

        if delta_time >= flash_interval:
            last_flash_time += flash_interval
            text_flash = not text_flash

        screen.blit(title_text, title_rect)
        screen.blit(high_score_text, high_score_rect)
        screen.blit(credits_text, credits_rect)
        screen.blit(version_text, version_rect)

        pygame.display.flip()


@lru_cache(maxsize=None)
def exit_code(coins: int, score: int):
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
