#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "base.h"
#include "dataset_bench_int.h"
#include "nes_random.h"
#include "network.h"
#include "demo_game_assets.h"
#include "iris-data-shuffled.h"
#include <stdlib.h>

#include "neslib.h"
#include <ines.h>
#include "game.h"

#ifdef __NES__
#include <ines.h>

#define asm __asm__

MAPPER_PRG_ROM_KB(32);
MAPPER_PRG_RAM_KB(8);
MAPPER_CHR_ROM_KB(8);
#endif

// --- Dataset Constants ---
#define IRIS_INPUT_SIZE 4       // Sepal Length, Sepal Width, Petal Length, Petal Width
#define IRIS_NUM_CLASSES 3      // Iris-setosa, Iris-versicolor, Iris-virginica
#define IRIS_TOTAL_SAMPLES 150
#define MAX_LINE_LENGTH 255     // Reduced, but still generous
#define TRAIN_SPLIT_RATIO 80   // for training, in %

// Calculate split sizes
#define IRIS_MAX_TRAIN_SAMPLES (int) (IRIS_TOTAL_SAMPLES * TRAIN_SPLIT_RATIO)
#define IRIS_MAX_TEST_SAMPLES (IRIS_TOTAL_SAMPLES - IRIS_MAX_TRAIN_SAMPLES)

// --- Hyperparameters (Adjusted for Iris) ---
#define BATCH_SIZE 4           // Smaller batch size for smaller dataset
#define HIDDEN_NEURONS 16       // Reduced hidden layer size
#define NUM_EPOCHS 10           // May need more epochs for convergence
#define LEARNING_RATE_MU 4      // Keep for now, might need tuning

// --- Global storage for float data ---
// Store data as floats first for adaptive quantization per batch
int8_t** X_train_float = NULL;
int8_t** X_test_float = NULL;
Vector8 Y_train_full; // Keep labels as int8
Vector8 Y_test_full;
lsize_t num_train_samples_loaded = 0;
lsize_t num_test_samples_loaded = 0;



const char pal[] = {0x0f, 0x10, 0x20, 0x30};

const unsigned char palette_didkotoy2_a[16]={ 0x0f,0x00,0x10,0x30,0x0f,0x0c,0x21,0x32,0x0f,0x05,0x16,0x27,0x0f,0x0b,0x1a,0x37 };

static const char msg[] = "HELLO WORLD";

#define VBLANK_FLAG 0x80

// asm (
//     ".section .nmi.009,\"ax\",@progbits\n"
//     "jsr game_step_before_nmi\n"
// );

// asm (
//     ".section .nmi.400,\"ax\",@progbits\n"
//     "jsr game_step_after_nmi\n"
// );

typedef struct Bullet {
    uint8_t x, y;
    uint8_t dir;
    uint8_t alive;
} Bullet;

typedef struct Player {
    uint8_t x, y;
    uint8_t dir;
    uint8_t score;
    char score_str[4];
    // char* score_str;
    Bullet bullet;
} Player;

volatile Player player;
volatile Player enemy;

// char x, y = 50;
// uint8_t bullet_x, bullet_y, bullet_dir, bullet_alive = 0;

// uint8_t color = 0;

// uint8_t dir = 1;

void split_iris(int8_t*** X_train, Vector8* Y_train, lsize_t* train_count, int8_t*** X_test, Vector8* Y_test, lsize_t* test_count) {
    *train_count = (int16_t) IRIS_SAMPLES * (int16_t) TRAIN_SPLIT_RATIO / (int16_t) 100;
    *test_count = IRIS_SAMPLES - *train_count;

    // TODO TESTS
    // *X_train = (int8_t**) iris_X;
    *X_train = malloc(*train_count * sizeof(int8_t*));
    for (lsize_t i = 0; i < *train_count; ++i) {
        // Cast to (int8_t*) to discard const/volatile for the assignment,
        // assuming read-only access through X_train_float.
        (*X_train)[i] = (int8_t*)iris_X[i]; // If using shuffled indices
        // Or, for a simple sequential split:
        // (*X_train)[i] = (int8_t*)iris_X[i];
    }
    
    // *X_test = iris_X + *train_count;
    *X_test = malloc(*test_count * sizeof(int8_t*));
    for (lsize_t i = 0; i < *test_count; ++i) {
        // Cast to (int8_t*) to discard const/volatile for the assignment,
        // assuming read-only access through X_train_float.
        (*X_test)[i] = (int8_t*)iris_X[i + *train_count]; // If using shuffled indices
        // Or, for a simple sequential split:
        // (*X_train)[i] = (int8_t*)iris_X[i];
    }

    *Y_train = init_v8(*train_count);
    // Y_train->vector = iris_Y;
    // free(Y_train->vector);
    // Y_train->vector = (int8_t*) iris_Y;
    if (Y_train->vector) { // Check if init_v8 succeeded
        memcpy(Y_train->vector, iris_Y, *train_count * sizeof(int8_t));
    }

    *Y_test = init_v8(*test_count);
    // free(Y_test->vector);
    // Y_test->vector = (int8_t*) (iris_Y + *train_count);
    if (Y_test->vector) { // Check if init_v8 succeeded
        memcpy(Y_test->vector, iris_Y + *train_count, *test_count * sizeof(int8_t));
    }
}

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

void spawn_player(volatile Player* player) {
    // player->x = rand8();
    // player->y = rand8();
    // player->x = nes_rand() % SPRITE_SCREEN_WIDTH;
    // player->y = nes_rand() % SPRITE_SCREEN_HEIGHT;
    player->x = 100;
    player->y = 100;
    player->dir = 1;
    if (player->score_str[0] == 0) {
        player->score = 0;
        // player->score_str = malloc(4 * sizeof(char));
        for (uint8_t i = 0; i < 3; ++i) {
            player->score_str[i] = '0';
        }
        player->score_str[3] = '\0';
    }
    // player->score = 0;
    // memcpy(player->score_str, "000", 4);
    // for (uint8_t i = 0; i < 3; ++i) {
    //     player->score_str[i] = '0';
    // }
    // player->score_str[3] = '\0';
    player->bullet.alive = 0;
}

void kill_other(volatile Player* killer, volatile Player* killed) {
    spawn_player(killed);
    // uint8_t size = ll_to_str(++killer->score, killer->score_str);
    // ++killer->score;
    // println("1");
    killer->score;
    // killer->score = 0;
    // killer->score = killer->score + 1;
    // killer->score_str[0] = killer->score / 100 + '0';
    // killer->score_str[1] = (killer->score / 10) % 10 + '0';
    // killer->score_str[2] = killer->score % 10 + '0';
    // killer->score_str[0] = killer->score + '0';
    // killer->score_str[1] = killer->score + '0';
    // killer->score_str[2] = killer->score + '0';
    // uint8_t tmp = 0;
    // for (uint16_t i = killer->score; i >= 100; i -= 100, ++tmp);
    // killer->score_str[0] = tmp + '0';
    // tmp = 0;
    // for (uint16_t i = killer->score; i >= 10; i -= 10, ++tmp);
    // killer->score_str[1] = tmp + '0';
    // tmp = 0;
    // for (uint16_t i = killer->score; i >= 1; i -= 1, ++tmp);
    // killer->score_str[2] = tmp + '0';
    // killer->score_str[3] = '\0';

    // uint8_t size = 3;
    // for (uint8_t i = size; i > 0; --i) {
    //     killer->score_str[i + 3 - size - 1] = killer->score_str[i - 1];
    // }
    // for (uint8_t i = 0; i < 3 - size; ++i) {
    //     killer->score_str[i] = '0';
    // }
    // killer->score_str[3] = '\0';
}

#define SCREEN_WIDTH_BLOCKS 32
#define SCREEN_HEIGHT_BLOCKS 30

#define BULLET_SPEED 4
#define PLAYER_SPEED 1

void ppu_puts(unsigned x, unsigned y, volatile const char *s) {
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
    // player.x = nes_rand() % SPRITE_SCREEN_WIDTH;
    // player.y = nes_rand() % SPRITE_SCREEN_HEIGHT;
    // player.dir = 1;
    // player.score = 0;

    // enemy.x = nes_rand() % SPRITE_SCREEN_WIDTH;
    // enemy.y = nes_rand() % SPRITE_SCREEN_HEIGHT;
    // enemy.dir = 1;
    // enemy.score = 0;

    // srand32(1);
}

void draw_score(uint8_t x, uint8_t y, volatile Player* player) {
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

extern void __set_heap_limit(size_t size);

int main(void) {
    #ifdef __NES__
    __set_heap_limit(1024 * 7);
    #endif
    init_pools();
    set_mu(LEARNING_RATE_MU); // Set global learning rate parameter
    // srand(time(NULL)); // Seeding is now done before shuffling in load function

    init_game();

    ppu_off();

    bank_bg(0);

    // pal_bg(palette_didkotoy2_a);
    // pal_spr(palette_didkotoy2_a);
    pal_spr(SPRITE_PAL);
    pal_bg(BG_PAL);

    vram_adr(NAMETABLE_A + 0x03C0);
    vram_fill(0b11111111, 64);

    vram_adr(NAMETABLE_A);
    vram_fill(' ', 32 * 30);

    // vram_adr(NTADR_A(10, 10));
    // vram_write(msg, sizeof(msg) - 1);
    ppu_puts(10, 10, msg);

    ppu_on_all();

    // println("--- C Implementation Configuration (Iris Dataset) ---");
    // println("Input Size: %d", IRIS_INPUT_SIZE);
    // println("Number of Classes: %d", IRIS_NUM_CLASSES);
    // println("Using learning rate parameter mu = %d", LEARNING_RATE_MU);
    // println("Batch Size: %d", BATCH_SIZE);
    // println("Target BITWIDTH = %d", BITWIDTH);
    // println("Hidden Neurons: %d", HIDDEN_NEURONS);
    // println("Epochs: %d", NUM_EPOCHS);
    // println("Train/Test Split: %d%% / %d%%", TRAIN_SPLIT_RATIO, 100 - TRAIN_SPLIT_RATIO);
    // println("----------------------------------------------------");
    // println("Initializing Iris Classification Test (C)...");
    // println("Allocating memory for labels...");

    split_iris(&X_train_float, &Y_train_full, &num_train_samples_loaded,
        &X_test_float, &Y_test_full, &num_test_samples_loaded);

    if (num_train_samples_loaded == 0 || num_test_samples_loaded == 0) {
        free_v8(&Y_train_full);
        free_v8(&Y_test_full);
        cleanup(NULL, &X_train_float, &Y_train_full, num_train_samples_loaded,
                    &X_test_float, &Y_test_full, num_test_samples_loaded);
        fatal("Error: Failed to load or split data. Train: %d, Test: %d", num_train_samples_loaded, num_test_samples_loaded);
    }

    Network* network = create_network(3,
                                    (LayerType[]) {LINEAR, RELU, LINEAR},
                                    (lsize_t[]) {IRIS_INPUT_SIZE, HIDDEN_NEURONS, HIDDEN_NEURONS, IRIS_NUM_CLASSES}, // Layer sizes: Input, Hidden, Output
                                    BATCH_SIZE);

    if (!network) {
        cleanup(NULL, &X_train_float, &Y_train_full, num_train_samples_loaded,
                    &X_test_float, &Y_test_full, num_test_samples_loaded);
        fatal("Error: Failed to create network.");
    }

    println("Network created successfully.");

    train_network(network,
                &X_train_float, &Y_train_full, num_train_samples_loaded,
                &X_test_float, &Y_test_full, num_test_samples_loaded,
                NUM_EPOCHS);
    free(X_train_float);
    free(X_test_float);
    free_network(network);

    println("Iris Classification Test Finished (C).");

    print_metrics();
    lin_cleanup();

    // for(;;) ppu_wait_frame();

    return 0;
}

#define PUSH(reg) PUSH_##reg
#define PUSH_x "txa\n" PUSH_a
#define PUSH_y "tya\n" PUSH_a
#define PUSH_a "pha\n"

#define POP(reg) POP_##reg
#define POP_x POP_a "tax\n"
#define POP_y POP_a "tay\n"
#define POP_a "pla\n"

// volatile unsigned char rcx_backup[32] __attribute__((section(".zp"), used));

// void copy_zero_page_to_buffer() {
//     __asm__ volatile (
//         PUSH(a)
//         PUSH(x)
//         PUSH(y)

//         // Save __rcX registers into $80..$9F
//         "ldx #$00\n"
//         "push_zp:\n"
//         "lda $00, x\n"
//         "sta rcx_backup, x\n"
//         "inx\n"
//         "cpx #$20\n"
//         "bne push_zp\n"
        

//         // "ldx #$00\n"
//         // "ldy #$00\n"
//         // "push_zp:\n"
//         // "lda $00,x\n"
//         // "sta ($fe),y\n"
//         // "inx\n"
//         // "iny\n"
//         // "cpx #$20\n"
//         // "bne push_zp\n"

//         POP(y)
//         POP(x)
//         POP(a)
//     );
// }

// void copy_buffer_to_zero_page() {
//     __asm__ volatile (
//         PUSH(a)
//         PUSH(x)
//         PUSH(y)
        
//         // Restore __rcX registers from $80..$9F
//         "ldx #$00\n"
//         "pop_zp:\n"
//         "lda rcx_backup, x\n"
//         "sta $00, x\n"
//         "inx\n"
//         "cpx #$20\n"
//         "bne pop_zp\n"
        


//         // "ldx #$00\n"
//         // "ldy #$00\n"
//         // "pop_zp:\n"
//         // "lda ($fe),y\n"   // Load from buffer
//         // "sta $00,x\n"     // Store to zero page
//         // "inx\n"
//         // "iny\n"
//         // "cpx #$20\n"      // 32 bytes
//         // "bne pop_zp\n"

//         POP(y)
//         POP(x)
//         POP(a)
//     );
// }

void check_bullet(volatile Player* p1, volatile Player* p2) {
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

void check_bullets(volatile Player* p1, volatile Player* p2) {
    check_bullet(p1, p2);
    check_bullet(p2, p1);
}

volatile int8_t just_pressed_a = 0;

void game_step_after_nmi(void) {
    // copy_zero_page_to_buffer();
    // asm volatile (
    //     "lda #$01\n"
    //     "sta VRAM_UPDATE\n"
    // );

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

    if (!just_pressed_a && (pad_state & PAD_A) && !player.bullet.alive) {
    // if ((pad_state & PAD_A) && !player.bullet.alive) {
        player.bullet.x = player.x + ((player.dir < 2) << 3) - PLAYER_HALF_SIZE;
        player.bullet.y = player.y + ((player.dir < 2) << 3) - PLAYER_HALF_SIZE;
        // player.bullet.dir = player.dir;
        player.bullet.dir = 0xa5 + ((player.dir + 2) % 4);
        player.bullet.alive = 1;
    }

    just_pressed_a = (pad_state & PAD_A);

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

    uint8_t enemy_dx = nes_rand();
    uint8_t enemy_dy = nes_rand();
    
    if (enemy_dy & 1) {
        if (enemy.y > 1 + (PLAYER_SPEED + PLAYER_HALF_SIZE)) enemy.y -= PLAYER_SPEED;
        enemy.dir = DIR_UP;
    } else {
        if (enemy.y < 239 - (PLAYER_SPEED + PLAYER_HALF_SIZE)) enemy.y += PLAYER_SPEED;
        enemy.dir = DIR_DOWN;
    }

    if (enemy_dx & 1) {
        if (enemy.x < 255 - (PLAYER_SPEED + PLAYER_HALF_SIZE)) enemy.x += 1;
        enemy.dir = DIR_RIGHT;
    } else {
        if (enemy.x > (PLAYER_SPEED + PLAYER_HALF_SIZE)) enemy.x -= 1;
        enemy.dir = DIR_LEFT;
    }

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

    check_bullets(&player, &enemy);

    // 

    if (player.bullet.alive) {
        oam_spr(player.bullet.x, player.bullet.y, 0x91 + ((player.bullet.dir - 1) >> 1), 0x3);
    }
    if (enemy.bullet.alive) {
        oam_spr(enemy.bullet.x, enemy.bullet.y, 0x91 + ((enemy.bullet.dir - 1) >> 1), 0x3);
    }

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

    // oam_spr(player.x, player.y, 0x7F + player.dir, 0x3);
    // oam_spr(enemy.x, enemy.y, 0x7F + enemy.dir, 0x3);

    // copy_buffer_to_zero_page();
}

// void __attribute__((interrupt)) nmi(void) {
//     println("AAA");
// }

void game_step_before_nmi(void) {
    draw_score(1, 1, &player);
    draw_score(SCREEN_WIDTH_BLOCKS - 4, 1, &enemy);
    // vram_adr(NTADR_A(bullet_x, bullet_y));
    // vram_put(0x80);

    // vram_adr(NTADR_A(x, y));
    // vram_put(0x7F + dir)

    // oam_spr(x, y, 0x7F + dir, 0x3);

    // oam_spr(x, y, 0x80, 0x4);
}

// int main(void) {
    

//     for (;;) {
//         ppu_wait_frame();
//     }
//     return 0;
// }