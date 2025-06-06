#ifndef DQN
#define DQN

#include "base.h"
#include "network.h"
#include <stdint.h>

typedef struct DatasetObject {
    uint8_t player_x;
    uint8_t player_y;
    uint8_t enemy_x;
    uint8_t enemy_y;
    uint8_t player_bullet_alive;
    uint8_t enemy_bullet_alive;
    uint8_t player_bullet_x;
    uint8_t player_bullet_y;
    uint8_t enemy_bullet_x;
    uint8_t enemy_bullet_y;
} DatasetObject;

#define MEMORY_CAPACITY 100
#define BATCH_SIZE 8
#define TARGET_UPDATE_FREQUENCY 10
#define INPUT_FEATURES sizeof(DatasetObject)
#define HIDDEN_NEURONS 16
#define NUM_ACTIONS 5
#define EPSILON_START 100
#define EPSILON_END 1
#define EPSILON_DECAY 1

typedef struct Experience {
    DatasetObject state;
    uint8_t action;
    uint8_t reward;
    DatasetObject next_state;
    uint8_t done;
} Experience;

typedef struct ReplayBuffer {
    Experience experiences[MEMORY_CAPACITY];
    lsize_t capacity;
    lsize_t size;
    lsize_t current_index;
} ReplayBuffer;

typedef struct DQNAgent {
    Network* main_network;
    // Network* target_network;
    ReplayBuffer replay_buffer;

    uint16_t steps_done;
    uint8_t current_epsilon;
    lsize_t batch_size;
} DQNAgent;

DQNAgent* init_dqn(void);
void free_dqn(DQNAgent* agent);
void dqn_step(DQNAgent* agent);
uint8_t dqn_select_action(DQNAgent* agent, const DatasetObject* state_obj);
void dqn_store_experience(DQNAgent* agent, const DatasetObject* state, uint8_t action, int8_t reward, const DatasetObject* next_state, uint8_t done);

#endif