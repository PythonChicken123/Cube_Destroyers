# coding= UTF-8
"""
This code has been previously reverted to an old version due to bug fixes
# TODO: Color printing coding
"""

from functools import wraps, lru_cache
import importlib
import subprocess
import sys
import os
os.environ['SDL_HINT_WINDOWS_ENABLE_MESSAGELOOP'] = "0"

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
required_libraries = ['pygame', 'numpy', 'PyMySQL', 'asyncio']

# Check if required libraries are installed and install them if missing
missing_libraries = []
for lib in required_libraries:
    try:
        if sys.platform.startswith('win'):
            import ctypes

            # Configuration on the Windows file
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                u'CompanyName.ProductName.SubProduct.VersionInformation')  # Arbitrary string
        from random import randint, choice
        from pygame.mixer import Sound, SoundType, Channel
        from pygame import Surface, SurfaceType, Vector2, Rect
        from pygame.rect import RectType
        from pygame.locals import *
        from pygame.font import Font
        from pygame.time import Clock
        from sqlite3 import Connection, Cursor
        from typing import *
        import pygame
        import numpy
        import sqlite3
        import asyncio

        importlib.import_module(lib)
    except ModuleNotFoundError:
        missing_libraries.append(lib)

if missing_libraries:
    print(f"{Colors.OKGREEN}The following required libraries are missing and will be installed:{Colors.ENDC}")
    for lib in missing_libraries:
        print(Colors.BOLD + Colors.OKORANGE + lib + Colors.ENDC)
    try:
        for lib in missing_libraries:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', lib])
        print(f"{Colors.BOLD + Colors.OKGREEN}Installation complete.{Colors.ENDC}")
        if sys.platform.startswith('win'):
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                u'CompanyName.ProductName.SubProduct.VersionInformation')  # Arbitrary string
        from random import randint, choice
        from pygame.mixer import Sound, SoundType, Channel
        from pygame import Surface, SurfaceType, Vector2, Rect
        from pygame.rect import RectType
        from pygame.locals import *
        from pygame.font import Font
        from pygame.time import Clock
        from sqlite3 import Connection, Cursor
        from typing import *
        import pygame
        import numpy
        import sqlite3
        import asyncio
    except Exception as e:
        print(f"{Colors.FAIL}An error occurred while installing the required libraries: {Colors.ENDC + str(e)}")
        sys.exit(1)

# Initialize Pygame and Mixer
pygame.init()
pygame.mixer.pre_init(frequency=44100, size=-16, channels=16, buffer=512, devicename=None,
                      allowedchanges=AUDIO_ALLOW_FREQUENCY_CHANGE | AUDIO_ALLOW_CHANNELS_CHANGE)
pygame.mixer.init(frequency=44100, size=-16, channels=16, buffer=512, devicename=None,
                  allowedchanges=AUDIO_ALLOW_FREQUENCY_CHANGE | AUDIO_ALLOW_CHANNELS_CHANGE | AUDIO_ALLOW_FORMAT_CHANGE)
pygame.mixer.get_init()

# Constants
HEIGHT: int
WIDTH: int
WIDTH, HEIGHT = 900, 600
HALF_WIDTH = WIDTH // 2
HALF_HEIGHT = HEIGHT // 2
PLAYER_SPEED = 7.5
BULLET_SPEED = 5
TARGET_SPEED = 3
MAX_TARGETS = 15
MAX_FROZEN_DURATION = 5000
MAX_HASTE_DURATION = 1000
MAX_HASTE_MULTIPLIER = 2
SPAWN_INTERVAL = numpy.sin(60)
SHOOT_COOLDOWN = 3
SPECIAL_TARGET_PROBABILITY = 10
DECELERATION = 0.25
FROZEN_TIMER = 0
FROZEN_DURATION = 1000
HASTE_TIMER = 0
HASTE_DURATION = 300
FPS: int = 60  # Frames Per Second
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (50, 255, 0)
GREY = (250, 250, 250)
YELLOW = (255, 255, 0)
LIGHT_BLUE = (173, 216, 230)
LIGHT_COLOR = (255, 255, 200)
DARK_COLOR = (50, 50, 50)
PLAYER_COLOR = (255, 0, 0)
COLORS: list[tuple[int, int, int]] = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 128, 0),
                                      (0, 0, 255), (0, 255, 255), (128, 0, 128), (255, 192, 203), (238, 130, 238)]


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


# Initialize player and light
@lru_cache(maxsize=None)
def calculate_lighting(distance):
    try:
        max_light: int
        min_light: int
        max_light, min_light = 255, 225
        attenuation: float = 0.01  # Adjust this value for different lighting effects
        # Ensure that distance is a positive value to prevent division by zero
        intensity: int = max_light / (1 + attenuation * distance)
        return max(min_light, intensity)
    except Exception as exception:
        raise AttributeError(
            f"Error in calculate_lighting: {str(exception)}"
        ) from exception


async def mixer_play(relative_path: Sound):
    """
    :rtype: Sound
    :param relative_path:
    """
    if relative_path:
        # Find a Channel to play multiple sounds with no buffer
        channel: Channel = relative_path.play()
        if not channel:
            try:
                channel = pygame.mixer.find_channel(force=True)
                channel.play(relative_path)
            except Exception as exception:
                raise FileNotFoundError(
                    f"An error occurred while searching for the following file '{relative_path}' : {exception}"
                ) from exception
        if channel:
            relative_path.play()


# Collect resource_path on PyInstaller --add_data for usage inside a executable file
@memoize
@lru_cache(maxsize=None)
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


# Create the game window
offscreen_surface: Surface = pygame.Surface((WIDTH, HEIGHT))
screen: Surface | SurfaceType = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Advanced Shooting Game")
pygame.display.set_icon(pygame.image.load(resource_path('data\\image\\frozen_special_egg2.png')).convert_alpha())

# Main images
background_image: Surface = pygame.image.load(resource_path('data\\image\\nebula2.png')).convert_alpha()
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))
coin_image: Surface = pygame.image.load(resource_path('data\\image\\coin.png')).convert_alpha()
coin_image = pygame.transform.scale(coin_image, (40, 40))  # Adjust the size as needed

# Main fonts
menu_font: Font = pygame.font.Font(resource_path('data\\fonts\\OpenSans-Semibold.ttf'), 36)
version_font: Font = pygame.font.Font(resource_path('data\\fonts\\OpenSans-Regular.ttf'), 12)
credits_font: Font = pygame.font.Font(resource_path('data\\fonts\\OpenSans-Bold.ttf'), 12)
title_font: Font = pygame.font.Font(resource_path('data\\fonts\\OpenSans-ExtraBold.ttf'), 60)
big_message_font: Font = pygame.font.Font(resource_path('data\\fonts\\OpenSans-Bold.ttf'), 42)
high_score_font: Font = pygame.font.Font(resource_path('data\\fonts\\Pixel.otf'), 12)


# ___________________________________________ DATA BASE ________________________________________________________________

@lru_cache(maxsize=None)
def create_players_table():
    connection: Connection = sqlite3.connect(os.path.join("data\\database\\game_data.db"))
    cursor: Cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY,
            coins INTEGER,
            high_score INTEGER
        )
    ''')
    connection.commit()
    connection.close()


create_players_table()


# Function to get the player's coins
@memoize
@lru_cache(maxsize=None)
def get_player_coins():
    connection: Connection = sqlite3.connect(os.path.join("data\\database\\game_data.db"))
    cursor: Cursor = connection.cursor()
    cursor.execute("SELECT coins FROM players WHERE id=1")
    coins: tuple[list[int]] = cursor.fetchone()
    connection.close()

    return coins[0] if coins else 0


# Function to update the player's coins
@memoize
@lru_cache(maxsize=None)
def update_player_coins(coins):
    connection: Connection = sqlite3.connect(os.path.join("data\\database\\game_data.db"))
    cursor: Cursor = connection.cursor()
    cursor.execute("UPDATE players SET coins = ? WHERE id = 1", (coins,))
    connection.commit()
    connection.close()


# Function to update player statistics (coins and high score)
@memoize
@lru_cache(maxsize=None)
def update_player_stats(coins, highest_score):
    connection: Connection = sqlite3.connect(os.path.join("data\\database\\game_data.db"))
    cursor: Cursor = connection.cursor()
    cursor.execute("UPDATE players SET coins = ?, high_score = ? WHERE id = 1", (coins, highest_score))
    connection.commit()
    connection.close()


# Initialize player
@memoize
@lru_cache(maxsize=None)
def initialize_player():
    coins: tuple[list[int]] = get_player_coins()
    if coins is None:
        update_player_coins(0)
    highest_score = get_high_score()
    if highest_score is None:
        update_high_score(0)
    return coins, highest_score


# Function to save the player's coins
@memoize
@lru_cache(maxsize=None)
def save_player_coins(coins):
    update_player_coins(coins)

    # Return the updated coins from the database
    return get_player_coins()


# Function to get the high score from the database
@memoize
@lru_cache(maxsize=None)
def get_high_score():
    connection = sqlite3.connect(os.path.join("data\\database\\game_data.db"))
    cursor = connection.cursor()
    cursor.execute("SELECT high_score FROM players WHERE id=1")
    highest_score = cursor.fetchone()
    connection.close()

    return highest_score[0] if highest_score else 0


# Function to update the high score in the database
@memoize
@lru_cache(maxsize=None)
def update_high_score(new_high_score):
    connection = sqlite3.connect(os.path.join("data\\database\\game_data.db"))
    cursor = connection.cursor()
    cursor.execute("UPDATE players SET high_score = ? WHERE id = 1", (new_high_score,))
    connection.commit()
    connection.close()


# ______________________________________________ DATA BASE: END ________________________________________________________
PLAYER_SIZE: int = 30

walls: list[Rect] = [
    pygame.Rect(100, 100, 20, 200),
    pygame.Rect(300, 50, 20, 150),
    pygame.Rect(500, 200, 20, 100),
    pygame.Rect(700, 100, 20, 200),
]


@memoize
async def play_game():  # sourcery skip: low-code-quality
    """ :return: """
    # Initialize game variables here
    bullets: list[tuple[Any, Any]] = []  # Store bullets as (x, y) tuples
    targets: list[dict[str, Rect | Surface | SurfaceType | int | bool | tuple[int, int, int]] | dict[
        str, Rect | Surface | SurfaceType | int | bool | tuple[
            int, int, int]]] = []  # Use 'targets' to keep track of the active targets
    score: int = 0
    shoot_cooldown: int = 0  # Initialize the shoot cooldown timer
    player_speed: int = 0  # Initialize player speed
    max_speed: int = 15  # Maximum speed when controls are held
    deceleration: float = 0.25  # Deceleration factor for slippery movement
    special_egg_destroyed: bool = False
    iterable = range(10)

    player_img: Surface = pygame.image.load(resource_path('data\\image\\chicken2.png')).convert_alpha()
    player_img = pygame.transform.rotozoom(player_img, 0, 2.0)
    player_img = pygame.transform.scale(player_img, (50, 75))
    player_rect = pygame.Rect(WIDTH // 2 - PLAYER_SIZE // 2, HEIGHT // 2 - PLAYER_SIZE // 2, PLAYER_SIZE, PLAYER_SIZE)
    player_rect: Rect | RectType = player_img.get_rect() or player_rect
    player_rect.center = (WIDTH // 2, HEIGHT - 50)
    normal_target_image: Surface = pygame.image.load(resource_path('data\\image\\egg.png')).convert_alpha()
    normal_target_image = pygame.transform.scale(normal_target_image, (40, 40))
    special_target_frozen_image: Surface = pygame.image.load(
        resource_path('data\\image\\frozen_special_egg2.png')).convert_alpha()
    special_target_frozen_image = pygame.transform.scale(special_target_frozen_image, (90, 90))
    special_target_image: Surface = pygame.image.load(resource_path('data\\image\\special_egg.png')).convert_alpha()
    special_target_image = pygame.transform.scale(special_target_image, (80, 80))
    explosion_frames: list[Surface | SurfaceType] = [
        pygame.transform.scale(pygame.image.load(os.path.join('data', 'image', 'explosion2.gif')), (160, 160)),
        pygame.transform.scale(pygame.image.load(os.path.join('data', 'image', 'explosion1.gif')), (160, 160))]
    explosion_sound: Sound = pygame.mixer.Sound(resource_path('data\\media\\explosion.wav'))
    explosion_frame_index: int = 0
    explosion_frame_delay: int = 10
    explosion_frame_counter: int = 0
    explosion_rect: Rect = pygame.Rect(0, 0, 80, 80)

    clock: Clock = pygame.time.Clock()
    running: bool = True
    spawn_timer: int = 0  # Initialize a timer for target spawning
    max_targets: int = MAX_TARGETS  # Set the maximum number of targets

    coins, last_highest_score = initialize_player()  # Initialize player's coins

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                exit_code(coins=coins, high_score=last_highest_score)

        keys = pygame.key.get_pressed()

        # Increase speed when holding controls
        if keys[pygame.K_LEFT]:
            if player_speed > -max_speed:
                player_speed -= 0.5
        elif keys[pygame.K_RIGHT]:
            if player_speed < max_speed:
                player_speed += 0.5
        elif player_speed > 0:
            player_speed -= deceleration
        elif player_speed < 0:
            player_speed += deceleration

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
            pygame.mixer.Sound(os.path.join('data/media/wind_bullet.mp3')).play()
            shoot_cooldown = SHOOT_COOLDOWN  # Set the cooldown timer

        # Respond to the display when display is not active
        while not pygame.display.get_active():
            list(map(lambda _: pygame.event.pump() or pygame.time.wait(1), iterable))

        # Move and remove bullets and assuming bullets is a list of (x, y) tuples
        bullets = list(
            map(lambda bullet: (bullet[0], bullet[1] - BULLET_SPEED), filter(lambda bullet: bullet[1] > 0.1, bullets)))

        list(map(lambda wall: pygame.draw.rect(screen, WHITE, wall), walls))

        player_center = player_rect.center
        angle = numpy.radians(45)  # Example angle, you can change this
        ray_length = 200

        end_point = (
            player_center[0] + ray_length * numpy.cos(angle),
            player_center[1] + ray_length * numpy.sin(angle)
        )

        # Draw the ray
        pygame.draw.line(screen, RED, player_center, end_point, 2)

        # Check if it's time to spawn a new target
        spawn_timer += 1
        if spawn_timer >= numpy.sin(160):  # You can adjust this value to control target spawn frequency
            if len(targets) < max_targets:
                if randint(1, 100) < SPECIAL_TARGET_PROBABILITY:
                    # Create a special target
                    special_target = pygame.Rect(randint(0, WIDTH - 30), 0, 30, 30)
                    special_color = BLUE
                    special_health = 3
                    special_frozen = True
                    # Use the loaded special_target_image for the special target
                    targets.append({'rect': special_target, 'image': special_target_image, 'color': special_color,
                                    'health': special_health, 'score': 3, 'is_special': True,
                                    'coins': randint(3, 10), 'frozen': special_frozen})
                else:
                    # Create a normal target
                    normal_target = pygame.Rect(randint(0, WIDTH - 30), 0, 30, 30)
                    normal_color = choice(COLORS)
                    normal_health = 1
                    # TODO: Add frozen consequences
                    normal_frozen = False
                    targets.append(
                        {'rect': normal_target, 'color': normal_color, 'health': normal_health, 'score': 1,
                         'coins': randint(1, 2), 'image': normal_target_image, 'frozen': normal_frozen})
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
                pygame.draw.rect(screen, target['color'], target['rect'])

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
                        special_egg_position = target['rect'].topright
                        special_egg_destroyed = True
                        explosion_rect.topleft = special_egg_position
                        explosion_frame_index = 0
                    target['health'] -= 1
                    if target['health'] == 0:
                        targets.remove(target)
                        score += target['score']
                        coins += target.get('coins', 0)

        # Remove bullets after the iteration
        bullets = [(x, y) for x, y in bullets if (x, y) not in bullets_to_remove]

        # Using map and filter to achieve the same effect
        bullets = list(map(lambda bullet: bullet, filter(lambda bullet: bullet not in bullets_to_remove, bullets)))

        # Draw everything
        screen.blit(background_image, (0, 0))  # Set the background image

        # Draw bullets and targets first
        bullet_rects = list(map(lambda bullet: pygame.Rect(bullet[0] - 2, bullet[1], 4, 10), bullets))
        list(map(lambda rect: pygame.draw.rect(screen, RED, rect), bullet_rects))

        async def perform_explosion():
            nonlocal explosion_frame_index, explosion_frame_counter, special_egg_destroyed

            screen.blit(explosion_frames[explosion_frame_index], explosion_rect.topleft)
            explosion_frame_counter += 1
            await mixer_play(explosion_sound)

            if explosion_frame_counter >= explosion_frame_delay:
                explosion_frame_index += 1
                explosion_frame_counter = 0

        def reset_explosion():
            nonlocal explosion_frame_index, special_egg_destroyed
            special_egg_destroyed = False
            explosion_frame_index = -1

        conditions = [
            special_egg_destroyed and 0 <= explosion_frame_index < len(explosion_frames),
            explosion_frame_index >= len(explosion_frames)
        ]

        actions = [perform_explosion, reset_explosion]

        # Perform the explosions after the special_target is removed
        list(map(lambda args: args[0]() if args[1] else None, zip(actions, conditions)))

        screen.blit(player_img, player_rect)

        # Draw the targets
        for target in targets:
            if target.get('is_special', False):
                if target.get('frozen', True):
                    # Apply a frozen spinning animation to the special target
                    target['image'] = special_target_frozen_image
                    target['rotation_angle'] = (target.get('rotation_angle', 0) + 0.25) % 360
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
                    pygame.draw.rect(screen, target['color'], target['rect'])
                else:
                    pygame.draw.rect(screen, target['color'], target['rect'])

        player_center: Vector2 = pygame.Vector2(player_rect.center)
        lighting_intensity: int = calculate_lighting(player_center.distance_to(pygame.Vector2(WIDTH // 2, HEIGHT // 2)))
        player_img_with_lighting: Surface = pygame.Surface(player_img.get_size(), pygame.SRCALPHA)
        player_img_with_lighting.fill((255, 255, 255, lighting_intensity))
        player_img_with_lighting.blit(player_img, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(player_img_with_lighting, player_rect)

        # Draw everything to the off-screen surface
        offscreen_surface.fill((0, 0, 0))  # Clear the off-screen surface

        # Display the score and coins
        font: Font = pygame.font.Font(None, 36)
        score_text: Surface | SurfaceType = font.render(f"Score: {str(score)}", True, RED)
        coins_text: Surface | SurfaceType = font.render(str(coins), True, YELLOW)
        screen.blit(coin_image, (10, 50))
        screen.blit(score_text, (10, 10))
        screen.blit(coins_text, (55, 50))

        pygame.display.flip()
        clock.tick(60)

    if score > last_highest_score:
        last_highest_score = score

    # Update coins and high score in the database
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
    options = ["3D Settings", "Sound Settings", "More Settings", "Back"]
    slide_bar_value = [0.5, 0.7, 0.3, 0.8]

    while True:
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_code(coins=get_player_coins(), high_score=get_high_score())
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
                        x_text = WIDTH // 2 - text.get_width() // 2
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
        title_rect.centerx = WIDTH // 2
        title_rect.y = 50
        screen.blit(title_text, title_rect)

        # Create option rectangles for mouse interaction
        option_rects = []
        for i, option in enumerate(zip(options, slide_bar_value)):
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
        pygame.time.Clock().tick(60)


@memoize
@lru_cache(maxsize=None)
def main_menu():
    selected_option: int = -1  # Initialize with no option selected
    options: list[str] = ["Play", "Settings", "Quit"]
    last_flash_time: int = pygame.time.get_ticks()
    flash_interval: int = 250  # Flash every 250ms
    text_flash: bool = True
    title_y: int = 100  # Initial vertical position of the title
    coins = get_player_coins()
    last_high_score: object = get_high_score()
    high_score_text: Surface | SurfaceType = high_score_font.render(f'HIGH SCORE :{str(last_high_score)}', True,
                                                                    LIGHT_BLUE)
    high_score_rect: Rect | RectType = high_score_text.get_rect()
    credits_text: Surface | SurfaceType = credits_font.render("Credits: ", True, GREEN)
    credits_rect: Rect | RectType = credits_text.get_rect()
    credits_rect.topleft = (10, HEIGHT - 25)
    version_text: Surface | SurfaceType = version_font.render("Version: 1.3", True, BLUE)
    version_rect: Rect | RectType = version_text.get_rect()
    version_rect.topright = (WIDTH - 10, HEIGHT - 25)

    while True:
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_code(coins=coins, high_score=last_high_score)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_option = (selected_option - 1) % len(options)
                if event.key == pygame.K_DOWN:
                    selected_option = (selected_option + 1) % len(options)
                if selected_option == 0:
                    if event.key == pygame.K_RETURN:
                        return asyncio.run(play_game())
                elif selected_option == 1:
                    if event.key == pygame.K_RETURN:
                        return main_settings()
                elif selected_option == 2:
                    if event.key == pygame.K_RETURN:
                        exit_code(coins=coins, high_score=last_high_score)
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
                                return asyncio.run(play_game())
                            elif i == 1:
                                return main_settings()
                            elif i == 2:
                                exit_code(coins=coins, high_score=last_high_score)
        screen.fill(WHITE)
        screen.blit(background_image, (0, 0))
        title_y += 1 * numpy.sin(current_time / 60)
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
def exit_code(coins, high_score):
    print(
        f"{Colors.OKMAGENTA}{Colors.BOLD}Saving game statistics{Colors.ENDC} {Colors.OKCYAN}{Colors.BOLD}Quitting pygame windows{Colors.ENDC}")
    update_high_score(high_score)
    update_player_coins(coins)
    update_player_stats(coins, high_score)
    pygame.display.quit()
    pygame.mixer.quit()
    pygame.font.quit()
    pygame.quit()
    sys.exit()


# Initialize Pygame
pygame.init()

# Constants
pygame.display.set_caption("Simple Shooting Game")

if __name__ == '__main__':
    high_score = get_high_score()
    print(high_score)
    main_menu()
