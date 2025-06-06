#ifndef GAME
#define GAME

void init_game(void);

void game_step_after_nmi(void);

// CAUTION: This method is computing-sensitive! Use only for displaying pre-computed data!
void game_step_before_nmi(void);

#endif