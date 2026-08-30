#include "balatro.h"
#include <time.h>

int main() {
    Env env = {0};
    env.rng = (unsigned int)time(NULL);
    Dict kwargs = {0};
    puf_init(&env, &kwargs);

    obs_t observations[OBS_SIZE];
    float actions[NUM_ATNS] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    unsigned char action_mask[ACTION_MASK_SIZE] = {0};

    memset(observations, 0, sizeof(observations));
    memset(actions, 0, sizeof(actions));
    memset(rewards, 0, sizeof(rewards));
    memset(terminals, 0, sizeof(terminals));
    memset(action_mask, 0, sizeof(action_mask));

    env.agents[0].observations = observations;
    env.agents[0].actions = actions;
    env.agents[0].rewards = rewards;
    env.agents[0].terminals = terminals;
    env.agents[0].action_mask = action_mask;
    env.agents[0].policy = 0;

    puf_reset(&env);
    puf_render(&env);

    while (!WindowShouldClose()) {
        puf_render(&env);
    }

    puf_close(&env);
    return 0;
}
