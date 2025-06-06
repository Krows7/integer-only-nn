#include "dqn.h"
#include "base.h"
#include "linear.h"
#include "network.h"
#include "dataset_bench_int.h"
#include "string.h"

// Helper to convert a single DatasetObject state to a 1xN Matrix8
void state_to_matrix8(const DatasetObject* state_obj, Matrix8* out) {
    *out = init_m8(1, INPUT_FEATURES); // batch_size = 1
    // Directly copy, assuming DatasetObject is contiguous uint8_t
    memcpy(out->matrix[0], state_obj, INPUT_FEATURES);
    out->scale = 0; // Assuming raw pixel/coordinate data, no initial scaling
    // return state_matrix;
}

DQNAgent* init_dqn(void) {
    DQNAgent* agent = malloc(sizeof(DQNAgent));
    assert_fatal(agent, "Failed to allocate DQNAgent");

    agent->current_epsilon = EPSILON_START;
    agent->main_network = create_network(3, (LayerType[]) {LINEAR, RELU, LINEAR}, (lsize_t[]) {sizeof(DatasetObject), HIDDEN_NEURONS, HIDDEN_NEURONS, NUM_ACTIONS}, BATCH_SIZE);
    // agent->target_network = create_raw_network(3, (LayerType[]) {LINEAR, RELU, LINEAR}, (lsize_t[]) {sizeof(DatasetObject), HIDDEN_NEURONS, HIDDEN_NEURONS, NUM_ACTIONS}, BATCH_SIZE);
    // agent->main_network = create_network(3, (LayerType[]) {LINEAR, RELU, LINEAR}, (lsize_t[]) {sizeof(DatasetObject), sizeof(DatasetObject), HIDDEN_NEURONS, NUM_ACTIONS}, BATCH_SIZE);
    // agent->target_network = create_raw_network(3, (LayerType[]) {LINEAR, RELU, LINEAR}, (lsize_t[]) {sizeof(DatasetObject), sizeof(DatasetObject), HIDDEN_NEURONS, NUM_ACTIONS}, BATCH_SIZE);

    // for (lsize_t i = 0; i < agent->main_network->num_layers; ++i) {
    //     if (agent->main_network->layers[i]->type == LINEAR) {
    //         m_cpy(&agent->target_network->layers[i]->weights, &agent->main_network->layers[i]->weights);
    //     }
    // }
    
    println("DQN Agent Initialized: Input=%d, Hidden=%d, Actions=%d", INPUT_FEATURES, HIDDEN_NEURONS, NUM_ACTIONS);
    print_network(agent->main_network);

    return agent;
}

void dqn_free(DQNAgent* agent) {
    if (agent) {
        free_network(agent->main_network);
        // free_network(agent->target_network);
        free(agent);
    }
}

void replay_sample(ReplayBuffer* buffer, lsize_t batch_size, Experience* batch) {
    for (lsize_t i = 0; i < batch_size; ++i) {
        lsize_t sample_idx = nes_rand() % buffer->size;
        batch[i] = buffer->experiences[sample_idx];
    }
}

void rb_push(ReplayBuffer* buffer, const DatasetObject* state, uint8_t action, int8_t reward, const DatasetObject* next_state, uint8_t done) {
    Experience* exp = &buffer->experiences[buffer->current_index];
    exp->state = *state;
    exp->action = action;
    exp->reward = reward;
    exp->next_state = *next_state;
    exp->done = done;

    buffer->current_index = (buffer->current_index + 1) % buffer->capacity;
    if (buffer->size < buffer->capacity) {
        buffer->size++;
    }
}

uint8_t dqn_select_action(DQNAgent* agent, const DatasetObject* state_obj) {
    agent->steps_done++;
    // Epsilon update (linear decay for simplicity)
    // This float math will need to be adapted for fixed-point on NES
    if (agent->current_epsilon > EPSILON_END) {
        // agent->current_epsilon -= (EPSILON_START - EPSILON_END) / EPSILON_DECAY;
        agent->current_epsilon -= EPSILON_DECAY;
        if (agent->current_epsilon < EPSILON_END) {
            agent->current_epsilon = EPSILON_END;
        }
    }

    // Epsilon-greedy
    // if ((float)rand32() / RANDOM_32_MAX < agent->current_epsilon) {
    // if ((nes_rand() % 100) < (uint8_t)(agent->current_epsilon * 100)) { // Simplified for int comparison
    if ((nes_rand() % 100) < agent->current_epsilon) {
        return nes_rand() % NUM_ACTIONS;
    } else {
        println("Inference");
        println("%d %d", agent->main_network->layers[0]->input_copy.width, agent->main_network->layers[0]->input_copy.height);
        // Matrix8 state_m = state_to_matrix8(state_obj);
        Matrix8 state_m;
        state_to_matrix8(state_obj, &state_m);
        Matrix8 q_values = network_forward(agent->main_network, &state_m);

        int8_t max_q = -128;
        uint8_t best_action = 0;
        for (uint8_t i = 0; i < NUM_ACTIONS; ++i) {
            if (q_values.matrix[0][i] > max_q) {
                max_q = q_values.matrix[0][i];
                best_action = i;
            }
        }
        println("Free");
        free_m8(&state_m);
        // free_m8(&q_values);
        return best_action;
    }
}

void dqn_store_experience(DQNAgent* agent, const DatasetObject* state, uint8_t action, int8_t reward, const DatasetObject* next_state, uint8_t done) {
    rb_push(&agent->replay_buffer, state, action, reward, next_state, done);
}

// Helper to convert a batch of DatasetObjects to a BxN Matrix8
void states_to_batch_matrix8(const DatasetObject* state_objs, lsize_t num_states, Matrix8* out_matrix) {
    if (out_matrix->width != num_states || out_matrix->height != INPUT_FEATURES) {
        if (out_matrix->matrix) free_m8(out_matrix);
        *out_matrix = init_m8(num_states, INPUT_FEATURES);
    }
    for (lsize_t i = 0; i < num_states; ++i) {
        memcpy(out_matrix->matrix[i], &state_objs[i], INPUT_FEATURES);
    }
    out_matrix->scale = 0; // Assuming raw data
}

void dqn_step(DQNAgent *agent) {
    if (agent->replay_buffer.size < BATCH_SIZE)  return;

    Experience batch[BATCH_SIZE];
    replay_sample(&agent->replay_buffer, BATCH_SIZE, batch);

    DatasetObject current_states[BATCH_SIZE];
    DatasetObject next_states[BATCH_SIZE];
    
    for (lsize_t i = 0; i < BATCH_SIZE; ++i) {
        current_states[i] = batch[i].state;
        next_states[i] = batch[i].next_state;
    }

    Matrix8 current_states_m = init_m8(0, 0);
    Matrix8 next_states_m = init_m8(0, 0);

    states_to_batch_matrix8(current_states, BATCH_SIZE, &current_states_m);
    states_to_batch_matrix8(next_states, BATCH_SIZE, &next_states_m);

    Matrix8 current_pred_all = network_forward(agent->main_network, &current_states_m);
    // Matrix8 next_target_all = network_forward(agent->target_network, &next_states_m);

    Matrix8 error = init_m8(BATCH_SIZE, 1);
    error.scale = current_pred_all.scale;

    // for (lsize_t i = 0; i < BATCH_SIZE; ++i) {
    //     int8_t reward_val = batch[i].reward;
    //     uint8_t action_val = batch[i].action;
    //     int8_t max_next_q = -128;

    //     if (!batch[i].done) {
    //         for (uint8_t a = 0; a < 1; ++a) {
    //             if (next_target_all.matrix[i][a] > max_next_q) {
    //                 max_next_q = next_target_all.matrix[i][a];
    //             }
    //         }
    //     } else {
    //         max_next_q = 0; // Terminal state, no future reward
    //     }

    //     // int16_t target_q_val_intermediate = (int16_t)reward_val + (int16_t)(GAMMA * max_next_q); // GAMMA is float, max_next_q is int8
    //     //                                                                                   // This needs proper fixed point math.
    //     // int8_t target_q_val_quantized = clamp_to_int8((float)target_q_val_intermediate); // Simplistic clamping

    //     // Initialize all errors for this sample to 0
    //     for (uint8_t a = 0; a < 1; ++a) error.matrix[i][a] = 0;

    //     // Set error for the action taken: Q_predicted - Q_target
    //     // Ensure scales are handled if Q_predicted and Q_target_quantized have different scales.
    //     // Assuming they are brought to the same scale for subtraction.
    //     // error.matrix[i][action_val] = current_pred_all.matrix[i][action_val] - target_q_val_quantized;
    // }

    // IMPORTANT: The existing network_backward expects Y (labels) and uses loss_gradient.
    // We need to bypass loss_gradient and feed our dqn_error_signal directly
    // into the backpropagation mechanism.
    // This requires either modifying network_backward or having a new function.
    // Let's assume a conceptual function: network_backward_with_error(network, initial_error_matrix)
    // For now, we'll call the existing one, knowing `Y` will be ignored if `loss` is replaced.
    // This is a HACK and needs proper refactoring of network_backward.
    // The loop inside network_backward starts with `Matrix8 loss = loss_gradient(...)`.
    // We need to replace this `loss` with `dqn_error_signal`.

    // --- HACK: Manually perform the backpropagation steps from network_backward ---
    // This bypasses the Y-based loss_gradient.
    Matrix8 current_error = error; // Start with our DQN error
    Matrix8 next_error_storage = {0}; // To store output of layer_backward_1

    for (int i = agent->main_network->num_layers - 1; i > 0; --i) {
        // layer_backward_1 needs the input that was fed to this layer during forward pass.
        // This is stored in network->layers[i-1]->activations (for layer i's input)
        // or network->layers[0]->input_copy (for layer 0's input)
        const Matrix8* layer_input_activations = (i == 0) ? &agent->main_network->layers[0]->input_copy 
                                                          : &agent->main_network->layers[i-1]->activations;
        
        // Ensure next_error_storage is appropriately sized or re-initializable by layer_backward_1
        if(next_error_storage.matrix) free_m8(&next_error_storage);
        next_error_storage = init_m8(0,0); // layer_backward_1 will init it

        layer_backward_1(agent->main_network->layers[i], layer_input_activations, &current_error, &next_error_storage);
        
        if (&current_error != &error) { // Don't free the initial dqn_error_signal if it was used directly
             free_m8(&current_error);
        }
        current_error = next_error_storage;
        next_error_storage.matrix = NULL; // Avoid double free if next iteration reuses
    }
    // Last layer (layer 0)
    if(next_error_storage.matrix) free_m8(&next_error_storage); // Should be empty if loop ran
    next_error_storage = init_m8(0,0);
    layer_backward_1(agent->main_network->layers[0], &agent->main_network->layers[0]->input_copy, &current_error, &next_error_storage);
    
    if (&current_error != &error) free_m8(&current_error);
    free_m8(&next_error_storage); // This is the error propagated to input, not used further here.
    // dqn_error_signal itself was the first error matrix, it's on stack or init_m8'd, needs free if init_m8'd.
    // Since dqn_error_signal was init_m8'd, it's freed if current_error pointed to it and then got freed.
    // If the loop didn't run (1 layer network), dqn_error_signal needs explicit free.
    // The logic above should handle freeing intermediate `current_error` matrices.
    // The original `dqn_error_signal` is implicitly handled if it becomes `current_error`.

    // --- End HACK ---

    free_m8(&current_states_m);
    free_m8(&next_states_m);
    free_m8(&current_pred_all);
    // free_m8(&next_target_all);
    // dqn_error_signal is managed by the HACK block

    // Update target network periodically
    // if (agent->steps_done % TARGET_UPDATE_FREQUENCY == 0) {
    //     log("Updating target network...");
    //     for(lsize_t i=0; i < agent->main_network->num_layers; ++i) {
    //         if (agent->main_network->layers[i]->type == LINEAR) {
    //             // This performs a deep copy of matrix data and scale
    //             m_cpy(&agent->target_network->layers[i]->weights, &agent->main_network->layers[i]->weights);
    //         }
    //         // Note: Activations and input_copy are transient and don't need copying for the target network's purpose.
    //     }
    // }
}