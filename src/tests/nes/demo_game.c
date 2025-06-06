#include "game.h"
#include "linear.h"
#include "my_print.h"
#include "neslib.h"
#include "stdint.h"
#include "nes_random.h"
#include "demo_game_assets.h"
#include <stdint.h>
#include <string.h>
#include "dqn.h"
#include <ines.h>

#define asm __asm__

MAPPER_PRG_ROM_KB(32);
MAPPER_PRG_RAM_KB(8);
MAPPER_CHR_ROM_KB(8);

DQNAgent* agent;

static DatasetObject s_t_observation;      // State at time t (before AI action)
static DatasetObject s_t_plus_1_observation; // State at time t+1 (after AI action and game step)
static uint16_t train_counter = 0; // Counter to trigger training periodically

// TODO Remove
static const char msg[] = "HELLO WORLD";

typedef struct Bullet {
    uint8_t x;
    uint8_t y;
    uint8_t dir;
    uint8_t alive;
} Bullet;

typedef struct Player {
    uint8_t x;
    uint8_t y;
    uint8_t dir;
    uint8_t score;
    char score_str[4];
    Bullet bullet;
} Player;

Player player;
Player enemy;

#define SPRITE_SCREEN_WIDTH 255 - 8
#define SPRITE_SCREEN_HEIGHT 239 - 8

struct PlayerMetatilePart {
    uint8_t offset_x;
    uint8_t offset_y;
    uint8_t tile;
    uint8_t palette;
};

struct PlayerMetatile {
    struct PlayerMetatilePart parts[4];
    uint8_t end;
} playerMetatile = {.parts[0] = {.offset_x = 0, .offset_y = 0, .tile = 0, .palette = 0},
                     .parts[1] = {.offset_x = 8, .offset_y = 0, .tile = 0, .palette = 0},
                     .parts[2] = {.offset_x = 0, .offset_y = 8, .tile = 0, .palette = 0},
                     .parts[3] = {.offset_x = 8, .offset_y = 8, .tile = 0, .palette = 0},
                     .end = 128};

void spawn_player(Player* player) {
    player->x = nes_rand() % SPRITE_SCREEN_WIDTH;
    player->y = nes_rand() % SPRITE_SCREEN_HEIGHT;
    player->dir = 1;
    player->score = 0;
    memcpy(player->score_str, "000", 4);
    player->bullet.alive = 0;
}

void kill_other(Player* killer, Player* killed) {
    spawn_player(killed);
    uint8_t size = ll_to_str(++killer->score, killer->score_str);
    for (uint8_t i = size; i > 0; --i) {
        killer->score_str[i + 3 - size - 1] = killer->score_str[i - 1];
    }
    for (uint8_t i = 0; i < 3 - size; ++i) {
        killer->score_str[i] = '0';
    }
    killer->score_str[3] = '\0';
}

#define SCREEN_WIDTH_BLOCKS 32
#define SCREEN_HEIGHT_BLOCKS 30

#define BULLET_SPEED 4
#define PLAYER_SPEED 1

void ppu_puts(unsigned x, unsigned y, const char *s) {
    unsigned addr = NTADR_A(x,y);
    vram_adr(addr);
    while (*s) {
        unsigned char tile = *s;
        vram_put(tile);
        ++s;
    }
}

void init_game(void) {
    spawn_player(&player);
    spawn_player(&enemy);
    
    ppu_off();

    bank_bg(0);

    pal_spr(SPRITE_PAL);
    pal_bg(BG_PAL);

    vram_adr(NAMETABLE_A + 0x03C0);
    vram_fill(0b11111111, 64);

    vram_adr(NAMETABLE_A);
    vram_fill(' ', 32 * 30);

    ppu_puts(10, 10, msg);

    ppu_on_all();
}

void draw_score(uint8_t x, uint8_t y, Player* player) {
    ppu_puts(x, y, player->score_str);
}

#define DIR_UP 0
#define DIR_RIGHT 1
#define DIR_DOWN 2
#define DIR_LEFT 3

#define BULLET_DOWN_SPR 0xa5
#define BULLET_LEFT_SPR 0xa6
#define BULLET_UP_SPR 0xa7
#define BULLET_RIGHT_SPR 0xa8

#define PLAYER_SIZE 16
#define PLAYER_HALF_SIZE PLAYER_SIZE / 2

void check_bullet(Player* p1, Player* p2) {
    if (p1->bullet.alive) {
        // if (abs(p1->bullet.x - p2->x) <= PLAYER_HALF_SIZE &&
        //     abs(p1->bullet.y - p2->y) <= PLAYER_HALF_SIZE) {
        //     p1->bullet.alive = 0;
        //     kill_other(p1, p2);
        // }
        if (p1->bullet.x >= p2->x - PLAYER_HALF_SIZE &&
            p1->bullet.x <= p2->x + PLAYER_HALF_SIZE &&
            p1->bullet.y >= p2->y - PLAYER_HALF_SIZE &&
            p1->bullet.y <= p2->y + PLAYER_HALF_SIZE) {
            p1->bullet.alive = 0;
            kill_other(p1, p2);
        }
    }
}

void check_bullets(Player* p1, Player* p2) {
    check_bullet(p1, p2);
    check_bullet(p2, p1);
}

// Helper to populate DatasetObject for the DQN
void get_current_game_state(DatasetObject* ds_obj) {
    ds_obj->player_x = player.x;
    ds_obj->player_y = player.y;
    ds_obj->enemy_x = enemy.x; // AI is the 'enemy' player
    ds_obj->enemy_y = enemy.y;
    ds_obj->player_bullet_alive = player.bullet.alive;
    ds_obj->player_bullet_x = player.bullet.x;
    ds_obj->player_bullet_y = player.bullet.y;
    ds_obj->enemy_bullet_alive = enemy.bullet.alive; // AI's bullet
    ds_obj->enemy_bullet_x = enemy.bullet.x;
    ds_obj->enemy_bullet_y = enemy.bullet.y;
}

// Helper to apply AI's chosen action
void apply_ai_action(uint8_t action, Player* ai_player_obj) {
    // Actions: 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT, 4:SHOOT, 5:NOOP
    uint8_t pressed_action = 0; // To set direction correctly if moving
    switch (action) {
        case 0: // UP
            if (ai_player_obj->y >= 1 + PLAYER_SPEED + PLAYER_HALF_SIZE) ai_player_obj->y -= PLAYER_SPEED;
            if (!pressed_action) ai_player_obj->dir = DIR_UP;
            pressed_action = 1;
            break;
        case 1: // DOWN
            if (ai_player_obj->y <= 239 - (PLAYER_SPEED + PLAYER_HALF_SIZE)) ai_player_obj->y += PLAYER_SPEED;
            if (!pressed_action) ai_player_obj->dir = DIR_DOWN;
            pressed_action = 1;
            break;
        case 2: // LEFT
            if (ai_player_obj->x >= (PLAYER_SPEED + PLAYER_HALF_SIZE)) ai_player_obj->x -= PLAYER_SPEED;
            if (!pressed_action) ai_player_obj->dir = DIR_LEFT;
            pressed_action = 1;
            break;
        case 3: // RIGHT
            if (ai_player_obj->x <= 255 - (PLAYER_SPEED + PLAYER_HALF_SIZE)) ai_player_obj->x += PLAYER_SPEED;
            if (!pressed_action) ai_player_obj->dir = DIR_RIGHT;
            pressed_action = 1;
            break;
        case 4: // SHOOT
            if (!ai_player_obj->bullet.alive) {
                ai_player_obj->bullet.x = ai_player_obj->x + ((ai_player_obj->dir < 2) << 3) - PLAYER_HALF_SIZE;
                ai_player_obj->bullet.y = ai_player_obj->y + ((ai_player_obj->dir < 2) << 3) - PLAYER_HALF_SIZE;
                ai_player_obj->bullet.dir = 0xa5 + ((ai_player_obj->dir + 2) % 4); // Sprite for bullet direction
                ai_player_obj->bullet.alive = 1;
            }
            break;
        case 5: // NOOP
            // Do nothing
            break;
    }
}

uint8_t game_started = 0;
__attribute__((section(".data"))) volatile uint8_t step_status = 0; // 0 = Not stated, 1 = computing, 2 = ready
uint8_t ai_action;

void game_step_after_nmi(void) {
}

void game_step(void) {
    int8_t reward = 0;
    uint8_t done = 0;

    const char pad_state = pad_poll(0);
    uint8_t pressed = 0;

    if (pad_state & PAD_UP) {
        if (player.y >= 1 + PLAYER_SPEED + PLAYER_HALF_SIZE) player.y -= PLAYER_SPEED;
        if (!pressed) player.dir = DIR_UP;
        pressed = 1;
    } else if (pad_state & PAD_DOWN) {
        if (player.y <= 239 - (PLAYER_SPEED + PLAYER_HALF_SIZE)) player.y += PLAYER_SPEED;
        if (!pressed) player.dir = DIR_DOWN;
        pressed = 1;
    }

    if (pad_state & PAD_RIGHT) {
        if (player.x <= 255 - (PLAYER_SPEED + PLAYER_HALF_SIZE)) player.x += PLAYER_SPEED;
        if (!pressed) player.dir = DIR_RIGHT;
        pressed = 1;
    } else if (pad_state & PAD_LEFT) {
        if (player.x >= (PLAYER_SPEED + PLAYER_HALF_SIZE)) player.x -= PLAYER_SPEED;
        if (!pressed) player.dir = DIR_LEFT;
        pressed = 1;
    }

    if ((pad_state & PAD_A) && !player.bullet.alive) {
        player.bullet.x = player.x + ((player.dir < 2) << 3) - PLAYER_HALF_SIZE;
        player.bullet.y = player.y + ((player.dir < 2) << 3) - PLAYER_HALF_SIZE;
        // player.bullet.dir = player.dir;
        player.bullet.dir = 0xa5 + ((player.dir + 2) % 4);
        player.bullet.alive = 1;
    }

    // 1. Observe current state for AI (enemy)
    get_current_game_state(&s_t_observation);

    // 2. AI (enemy) selects action
    // ai_action = dqn_select_action(agent, &s_t_observation);
    ai_action = dqn_select_action(agent, &s_t_observation);

    oam_clear();

    if (player.bullet.alive) {
        if (player.bullet.dir == BULLET_DOWN_SPR) {
            if (player.bullet.y >= BULLET_SPEED) player.bullet.y += BULLET_SPEED;
            else player.bullet.alive = 0;
        } else if (player.bullet.dir == BULLET_UP_SPR) {
            if (player.bullet.y <= 239 - 8 - BULLET_SPEED) player.bullet.y -= BULLET_SPEED;
            else player.bullet.alive = 0;
        } else if (player.bullet.dir == BULLET_RIGHT_SPR) {
            if (player.bullet.x <= (255 - 8 - BULLET_SPEED)) player.bullet.x += BULLET_SPEED;
            else player.bullet.alive = 0;
        } else if (player.bullet.dir == BULLET_LEFT_SPR) {
            if (player.bullet.x >= BULLET_SPEED) player.bullet.x -= BULLET_SPEED;
            else player.bullet.alive = 0;
        }
    }

    // Enemy Logic (Dummy)

    // uint8_t enemy_dx = nes_rand();
    // uint8_t enemy_dy = nes_rand();

    // if (enemy_dy & 1) {
    //     if (enemy.y > 1 + (PLAYER_SPEED + PLAYER_HALF_SIZE)) enemy.y -= PLAYER_SPEED;
    //     enemy.dir = DIR_UP;
    // } else {
    //     if (enemy.y < 239 - (PLAYER_SPEED + PLAYER_HALF_SIZE)) enemy.y += PLAYER_SPEED;
    //     enemy.dir = DIR_DOWN;
    // }

    // if (enemy_dx & 1) {
    //     if (enemy.x < 255 - (PLAYER_SPEED + PLAYER_HALF_SIZE)) enemy.x += 1;
    //     enemy.dir = DIR_RIGHT;
    // } else {
    //     if (enemy.x > (PLAYER_SPEED + PLAYER_HALF_SIZE)) enemy.x -= 1;
    //     enemy.dir = DIR_LEFT;
    // }

    // 3. Apply AI's action to 'enemy' player
    apply_ai_action(ai_action, &enemy);

    // Update AI (enemy) bullet
    if (enemy.bullet.alive) {
        if (enemy.bullet.dir == BULLET_DOWN_SPR) {
            if (enemy.bullet.y >= BULLET_SPEED) enemy.bullet.y += BULLET_SPEED;
            else enemy.bullet.alive = 0;
        } else if (enemy.bullet.dir == BULLET_UP_SPR) {
            if (enemy.bullet.y <= 239 - 8 - BULLET_SPEED) enemy.bullet.y -= BULLET_SPEED;
            else enemy.bullet.alive = 0;
        } else if (enemy.bullet.dir == BULLET_RIGHT_SPR) {
            if (enemy.bullet.x <= (255 - 8 - BULLET_SPEED)) enemy.bullet.x += BULLET_SPEED;
            else enemy.bullet.alive = 0;
        } else if (enemy.bullet.dir == BULLET_LEFT_SPR) {
            if (enemy.bullet.x >= BULLET_SPEED) enemy.bullet.x -= BULLET_SPEED;
            else enemy.bullet.alive = 0;
        }
    }

    // 6. Check for collisions and determine reward/done
    uint8_t old_player_score = player.score;
    uint8_t old_enemy_score = enemy.score;

    check_bullets(&player, &enemy);

    if (player.score > old_player_score) { // Player killed AI (enemy)
        reward = -20; // Negative reward for AI
        done = 1;     // Episode ends for AI
    }
    if (enemy.score > old_enemy_score) { // AI (enemy) killed Player
        reward = +20; // Positive reward for AI
        done = 1;     // Episode ends for AI (from its perspective)
    }

    if (!done) { // Small penalty for existing if game not over
        reward -= 1;
    }

    // 7. Observe next state for AI
    get_current_game_state(&s_t_plus_1_observation);
    dqn_store_experience(agent, &s_t_observation, ai_action, reward, &s_t_plus_1_observation, done);

    if (player.bullet.alive) {
        oam_spr(player.bullet.x, player.bullet.y, player.bullet.dir, 0x3);
    }
    if (enemy.bullet.alive) {
        oam_spr(enemy.bullet.x, enemy.bullet.y, enemy.bullet.dir, 0x3);
    }

    uint8_t base_tile = 0x80 + (player.dir << 1);

    for (uint8_t i = 0; i < 4; ++i) {
        uint8_t is_down = i > 1;
        playerMetatile.parts[i].tile = base_tile + (0x10 * is_down) + (i & 1);
        playerMetatile.parts[i].palette = 0b0000010;
    }

    oam_meta_spr(player.x - PLAYER_HALF_SIZE, player.y - PLAYER_HALF_SIZE, &playerMetatile);

    base_tile = 0x80 + (enemy.dir << 1);

    for (uint8_t i = 0; i < 4; ++i) {
        uint8_t is_down = i > 1;
        playerMetatile.parts[i].tile = base_tile + (0x10 * is_down) + (i & 1);
        playerMetatile.parts[i].palette = 0b00000001;
    }

    oam_meta_spr(enemy.x, enemy.y, &playerMetatile);
}

void game_step_before_nmi(void) {
    if (!game_started) return;
    draw_score(1, 1, &player);
    draw_score(SCREEN_WIDTH_BLOCKS - 4, 1, &enemy);
}

extern void __set_heap_limit(size_t limit);

#define LEARNING_RATE_MU 4

int main(void) {
    nes_srand(1);
    __set_heap_limit(1024 * 7);
    set_mu(LEARNING_RATE_MU);
    init_pools();
    init_game();

    agent = init_dqn();

    game_started = 1;

    if (!agent) {
        println("DQN INIT FAIL");
    }

    println("Start loop");

    while(1) {
        // __asm__ volatile (
        //     "pha\n"
        //     "lda #$01\n"
        //     "sta VRAM_UPDATE\n"
        //     "pla\n"
        // );
        ppu_wait_nmi();
        game_step();
        ++train_counter;
        if (agent->replay_buffer.size >= BATCH_SIZE && (train_counter % 4 == 0)) {
            dqn_step(agent);
        }
    }

    return 0;
}