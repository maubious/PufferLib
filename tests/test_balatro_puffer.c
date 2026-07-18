#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "../ocean/balatro/balatro.h"

int main(void) {
    Env env = {0};
    Dict kwargs = {0};
    float observation[OBS_SIZE];
    float actions[NUM_ATNS] = {0};
    float reward = 0.0f;
    float terminal = 0.0f;
    unsigned char action_mask[ACTION_MASK_SIZE];

    env.agents[0].observations = observation;
    env.agents[0].actions = actions;
    env.agents[0].rewards = &reward;
    env.agents[0].terminals = &terminal;
    env.agents[0].action_mask = action_mask;
    puf_init(&env, &kwargs);
    puf_reset(&env);

    int transitions = 0;
    int terminals = 0;
    for (; transitions < 1024; ++transitions) {
        for (int i = 0; i < OBS_SIZE; ++i) assert(isfinite(observation[i]));
        BalatroLegalView view = {0};
        assert(balatro_legal_view(&env.state, &view) == BALATRO_OK);
        assert(view.group_count > 0);
        BalatroAction action = {0};
        assert(balatro_legal_group_action(&view.groups[0], 0, &action) == BALATRO_OK);
        actions[0] = (float)action.type;
        actions[1] = (float)action.primary;
        actions[2] = (float)action.selection_count;
        for (int i = 0; i < BALATRO_MAX_SELECTION; ++i)
            actions[3 + i] = (float)action.selection[i];
        actions[8] = 0.0f;
        puf_step(&env);
        assert(isfinite(reward));
        assert(isfinite(terminal));
        if (terminal > 0.5f) ++terminals;
    }
    assert(env.log.invalid_actions == 0.0f);
    assert(terminals > 0);
    printf("simulatro Puffer adapter: %d transitions, %d terminals, invalid=%.0f\n",
           transitions, terminals, env.log.invalid_actions);
    return 0;
}
