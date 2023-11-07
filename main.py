from random import randint, choice
from pygame import Surface, SurfaceType
import pygame
import numpy
import sys
import sqlite3

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 900, 600
HALF_WIDTH = WIDTH // 2
HALF_HEIGHT = HEIGHT // 2
PLAYER_SPEED = 7.5
BULLET_SPEED = 5
TARGET_SPEED = 3
MAX_TARGETS = 10
SHOOT_COOLDOWN = 3
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (50, 255, 0)
GREY = (250, 250, 250)
YELLOW = (255, 255, 0)
LIGHT_BLUE = (173, 216, 230)
COLORS = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 128, 0),
          (0, 0, 255), (0, 255, 255), (128, 0, 128), (255, 192, 203), (238, 130, 238)]

# Create the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Shooting Game")

# Load images
background_image = pygame.image.load('data\\image\\nebula.png').convert_alpha()
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))
player_img = pygame.image.load('data\\image\\chicken2.png').convert_alpha()
player_img = pygame.transform.scale(player_img, (50, 75))
special_target_image = pygame.image.load('data\\image\\special_egg.png').convert_alpha()
special_target_image = pygame.transform.scale(special_target_image, (80, 80))
coin_image = pygame.image.load('data\\image\\coin.png').convert_alpha()
coin_image = pygame.transform.scale(coin_image, (40, 40))  # Adjust the size as needed
explosion_frames = [
    pygame.transform.scale(pygame.image.load(f'data\\image\\explosion{frame}.gif').convert_alpha(), (160, 160))
    for frame in range(1, 3)]

# Load fonts
menu_font = pygame.font.Font('data\\fonts\\OpenSans-Semibold.ttf', 36)
version_font = pygame.font.Font('data\\fonts\\OpenSans-Regular.ttf', 12)
credits_font = pygame.font.Font('data\\fonts\\OpenSans-Bold.ttf', 12)
title_font = pygame.font.Font('data\\fonts\\OpenSans-ExtraBold.ttf', 60)
big_message_font = pygame.font.Font('data\\fonts\\OpenSans-Bold.ttf', 42)
high_score_font = pygame.font.Font('data\\fonts\\Pixel.otf', 12)

# Initialize Pygame mixer for sound
pygame.mixer.init()


# ___________________________________________ DATA BASE ________________________________________________________________

def create_players_table():
    connection = sqlite3.connect("data\\database\\game_data.db")
    cursor = connection.cursor()
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
def get_player_coins():
    connection = sqlite3.connect("data\\database\\game_data.db")
    cursor = connection.cursor()
    cursor.execute("SELECT coins FROM players WHERE id=1")
    coins = cursor.fetchone()
    connection.close()

    if coins:
        return coins[0]
    else:
        return 0


# Function to update the player's coins
def update_player_coins(coins):
    connection = sqlite3.connect("data\\database\\game_data.db")
    cursor = connection.cursor()
    cursor.execute("UPDATE players SET coins = ? WHERE id = 1", (coins,))
    connection.commit()
    connection.close()


# Function to update player statistics (coins and high score)
def update_player_stats(coins, high_score):
    connection = sqlite3.connect("data\\database\\game_data.db")
    cursor = connection.cursor()
    cursor.execute("UPDATE players SET coins = ?, high_score = ? WHERE id = 1", (coins, high_score))
    connection.commit()
    connection.close()


# Initialize player
def initialize_player():
    coins = get_player_coins()
    if coins is None:
        update_player_coins(0)
    high_score = get_high_score()
    if high_score is None:
        update_high_score(0)
    return coins, high_score


# Function to save the player's coins
def save_player_coins(coins):
    update_player_coins(coins)

    # Return the updated coins from the database
    return get_player_coins()


# Function to get the high score from the database
# Function to get the high score from the database
def get_high_score():
    connection = sqlite3.connect("data\\database\\game_data.db")
    cursor = connection.cursor()
    cursor.execute("SELECT high_score FROM players WHERE id=1")
    high_score = cursor.fetchone()
    connection.close()

    if high_score:
        return high_score[0]
    else:
        return 0


# Function to update the high score in the database
def update_high_score(new_high_score):
    connection = sqlite3.connect("data\\database\\game_data.db")
    cursor = connection.cursor()
    cursor.execute("UPDATE players SET high_score = ? WHERE id = 1", (new_high_score,))
    connection.commit()
    connection.close()


# ______________________________________________ DATA BASE: END ________________________________________________________
def play_game():
    # Initialize game variables here
    bullets = []  # Store bullets as (x, y) tuples
    targets = []  # Use 'targets' to keep track of the active targets
    score = 0
    shoot_cooldown = 0  # Initialize the shoot cooldown timer
    player_speed = 0  # Initialize player speed
    max_speed = 15  # Maximum speed when controls are held
    deceleration = 0.25  # Deceleration factor for slippery movement
    current_explosion_frame = -1
    special_egg_destroyed = False

    # Load player image
    player_img: Surface = pygame.image.load('data/image/chicken2.png').convert_alpha()
    player_img = pygame.transform.rotozoom(player_img, 0, 2.0)
    player_img = pygame.transform.scale(player_img, (50, 75))
    player_rect = player_img.get_rect()
    player_rect.center = (WIDTH // 2, HEIGHT - 50)
    normal_target_image: Surface = pygame.image.load('data/image/egg.png').convert_alpha()
    normal_target_image = pygame.transform.scale(normal_target_image, (40, 40))
    special_target_image: Surface = pygame.image.load('data/image/special_egg.png').convert_alpha()
    special_target_image = pygame.transform.scale(special_target_image, (80, 80))
    explosion_frames: list[Surface | SurfaceType] = [
        pygame.transform.scale(pygame.image.load('data\\image\\explosion1.gif'), (160, 160)),
        pygame.transform.scale(pygame.image.load('data\\image\\explosion2.gif'), (160, 160))]
    explosion_sound = pygame.mixer.Sound('data\\media\\explosion.wav')
    explosion_frame_index = 0
    explosion_frame_delay = 10
    explosion_frame_counter = 0
    explosion_rect = pygame.Rect(0, 0, 80, 80)

    clock = pygame.time.Clock()
    running = True
    spawn_timer = 0  # Initialize a timer for target spawning
    max_targets = MAX_TARGETS  # Set the maximum number of targets

    coins, high_score = initialize_player()  # Initialize player's coins

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # Increase speed when holding controls
        if keys[pygame.K_LEFT]:
            if player_speed > -max_speed:
                player_speed -= 0.5
        elif keys[pygame.K_RIGHT]:
            if player_speed < max_speed:
                player_speed += 0.5
        else:
            # Apply deceleration when controls are released
            if player_speed > 0:
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
            pygame.mixer.Sound('data/media/wind_bullet.mp3').play()
            shoot_cooldown = SHOOT_COOLDOWN  # Set the cooldown timer

        # Move and remove bullets
        bullets = [(x, y - BULLET_SPEED) for x, y in bullets if y > 0]

        # Check if it's time to spawn a new target
        spawn_timer += 1
        if spawn_timer >= numpy.sin(60):  # You can adjust this value to control target spawn frequency
            if len(targets) < max_targets:
                if randint(1, 100) < 2:
                    # Create a special target
                    special_target = pygame.Rect(randint(0, WIDTH - 30), 0, 30, 30)
                    special_color = BLUE
                    special_health = 3
                    # Use the loaded special_target_image for the special target
                    targets.append({'rect': special_target, 'image': special_target_image, 'color': special_color,
                                    'health': special_health, 'score': 3, 'is_special': True,
                                    'coins': randint(3, 10)})
                else:
                    # Create a normal target
                    normal_target = pygame.Rect(randint(0, WIDTH - 30), 0, 30, 30)
                    normal_color = choice(COLORS)
                    normal_health = 1
                    targets.append(
                        {'rect': normal_target, 'color': normal_color, 'health': normal_health, 'score': 1,
                         'coins': randint(1, 2), 'image': normal_target_image})
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

            # Only keep targets that are within the window
            if target['rect'].bottom <= HEIGHT:
                new_targets.append(target)

        targets = new_targets

        # Check for bullet-target collisions and update target health
        for i, bullet in enumerate(bullets):
            for target in targets:
                if target['rect'].colliderect(
                        pygame.Rect(bullet[0] - 2, bullet[1], 4, 10)):  # Create a temporary rect for the bullet
                    bullets.pop(i)
                    if target.get('is_special', False):
                        special_egg_position = target['rect'].topleft
                        special_egg_destroyed = True
                        explosion_rect.topleft = special_egg_position
                        explosion_frame_index = 0
                    target['health'] -= 1
                    if target['health'] == 0:
                        targets.remove(target)
                        score += target['score']
                        coins += target.get('coins', 0)

        # Draw everything
        screen.blit(background_image, (0, 0))  # Set the background image

        # Draw bullets and targets first
        for bullet in bullets:
            pygame.draw.rect(screen, RED, pygame.Rect(bullet[0] - 2, bullet[1], 4, 10))

        if special_egg_destroyed and 0 <= explosion_frame_index < len(explosion_frames):
            screen.blit(explosion_frames[explosion_frame_index], explosion_rect.topleft)
            explosion_frame_counter += 1
            explosion_sound.play()
            if explosion_frame_counter >= explosion_frame_delay:
                explosion_frame_index += 1
                explosion_frame_counter = 0
        elif explosion_frame_index >= len(explosion_frames):
            special_egg_destroyed = False
            explosion_frame_index = -1

        screen.blit(player_img, player_rect)

        # Draw the targets
        for target in targets:
            if target.get('is_special', False):
                # Apply a spinning animation to the special target
                target['rotation_angle'] = (target.get('rotation_angle', 0) + 2) % 360
                rotated_surface = pygame.transform.rotate(target['image'], target['rotation_angle'])
                target['rect'] = rotated_surface.get_rect(center=target['rect'].center)
                screen.blit(rotated_surface, target['rect'].topleft)
            else:
                pygame.draw.rect(screen, target['color'], target['rect'])

        # Display the score and coins
        font = pygame.font.Font(None, 36)
        score_text = font.render("Score: " + str(score), True, RED)
        coins_text = font.render(str(coins), True, YELLOW)
        screen.blit(coin_image, (10, 50))
        screen.blit(score_text, (10, 10))
        screen.blit(coins_text, (55, 50))

        pygame.display.flip()
        clock.tick(60)

    if score > high_score:
        high_score = score

    # Update coins and high score in the database
    update_high_score(high_score)
    update_player_stats(coins, high_score)
    update_high_score(high_score)

    # Return the targets and bullets for the next game
    return targets, bullets


def main_menu():
    selected_option = -1  # Initialize with no option selected
    options = ["Play", "Quit"]
    last_flash_time = pygame.time.get_ticks()
    flash_interval = 250  # Flash every 250ms
    text_flash = True
    title_y = 100  # Initial vertical position of the title
    last_high_score = get_high_score()
    high_score_text = high_score_font.render(f'HIGH SCORE :{str(last_high_score)}', True, LIGHT_BLUE)
    high_score_rect = high_score_text.get_rect()
    credits_text = credits_font.render("Credits: ", True, GREEN)
    credits_rect = credits_text.get_rect()
    credits_rect.topleft = (10, HEIGHT - 25)
    version_text = version_font.render("Version: 1.0", True, BLUE)
    version_rect = version_text.get_rect()
    version_rect.topright = (WIDTH - 10, HEIGHT - 25)

    while True:
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_option = (selected_option - 1) % len(options)
                if event.key == pygame.K_DOWN:
                    selected_option = (selected_option + 1) % len(options)
                if event.key == pygame.K_RETURN:
                    if selected_option == 0:
                        return play_game()
                    elif selected_option == 1:
                        pygame.quit()
                        sys.exit()
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
                                # Play the game
                                return play_game()
                            elif i == 1:
                                pygame.quit()
                                sys.exit()

        screen.fill(WHITE)
        screen.blit(background_image, (0, 0))
        title_y += 1 * numpy.sin(current_time / 60)
        title_text = title_font.render("Chicken Cube Destroyers", True, RED)
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
            if option_rect.collidepoint(pygame.mouse.get_pos()):
                pygame.draw.rect(screen, LIGHT_BLUE, option_rect, border_radius=10)

            if i == selected_option:
                pygame.draw.rect(screen, BLUE, option_rect, border_radius=10, width=2)

            screen.blit(text, (x, y))

        if current_time - last_flash_time >= flash_interval:
            last_flash_time = current_time
            text_flash = not text_flash

        screen.blit(title_text, title_rect)
        screen.blit(credits_text, credits_rect)
        screen.blit(version_text, version_rect)
        screen.blit(high_score_text, high_score_rect)

        pygame.display.flip()


# Initialize Pygame
pygame.init()

# Constants
pygame.display.set_caption("Simple Shooting Game")

if __name__ == '__main__':
    high_score = get_high_score()
    print(high_score)
    main_menu()