#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "../ocean/balatro/balatro.h"

static uint64_t mask_u64(const unsigned char *mask, int offset) {
    uint64_t value = 0;
    memcpy(&value, mask + offset, sizeof(value));
    return value;
}

int main(void) {
    Env env = {0};
    Dict kwargs = {0};
    unsigned char observation[OBS_SIZE];
    float actions[NUM_ATNS] = {0};
    float reward = 0.0f;
    float terminal = 0.0f;
    unsigned char action_mask[ACTION_MASK_SIZE];

    DictItem *timeout = dict_item(&kwargs, "max_episode_steps");
    timeout->value = 1.0;
    env.agents[0].observations = observation;
    env.agents[0].actions = actions;
    env.agents[0].rewards = &reward;
    env.agents[0].terminals = &terminal;
    env.agents[0].action_mask = action_mask;
    puf_init(&env, &kwargs);
    puf_reset(&env);
    assert(env.legal_masks.action_type[ACTION_SWAP_HAND_LEFT] == 0);
    assert(env.legal_masks.action_type[ACTION_SWAP_HAND_RIGHT] == 0);
    assert(env.legal_masks.action_type[ACTION_SORT_HAND_RANK] == 0);
    assert(env.legal_masks.action_type[ACTION_SORT_HAND_SUIT] == 0);
    assert(env.legal_masks.action_type[ACTION_SWAP_JOKERS_LEFT] == 0);
    assert(env.legal_masks.action_type[ACTION_SWAP_JOKERS_RIGHT] == 0);
    assert(action_mask[ACTION_SELECT_BLIND] == 1);
    assert(mask_u64(action_mask, POLICY_PRIMARY_OFFSET +
        ACTION_SELECT_BLIND * POLICY_PRIMARY_BYTES) == 1);

    /* A one-step limit must truncate a nonterminal initial blind-select step. */
    LegalMasks timeout_masks = {0};
    LegalView timeout_view = {0};
    assert(legal_masks(&env.state, &timeout_masks) == OK);
    assert(legal_expand(&timeout_masks, &timeout_view) == OK);
    Action timeout_action = {0};
    assert(legal_group_action(&timeout_view.groups[0], 0, &timeout_action) == OK);
    State expected_state = env.state;
    StepResult expected_result = {0};
    assert(step(&expected_state, &timeout_action, &expected_result) == OK);
    actions[0] = (float)timeout_action.type;
    actions[1] = (float)timeout_action.primary;
    actions[2] = (float)timeout_action.selection_count;
    for (int i = 0; i < MAX_SELECTION; ++i) actions[3 + i] = (float)timeout_action.selection[i];
    puf_step(&env);
    assert(env.log.truncations == 1.0f);
    assert(fabsf(env.log.score - env.log.reward) < 1e-6f);
    assert(env.log.perf == 0.0f);
    assert(terminal > 0.5f);
    /* The valid transition comes from Simulatro unchanged; truncation only
       adds the wrapper's timeout penalty. */
    assert(fabsf(reward - (expected_result.reward + TIMEOUT_REWARD)) < 1e-6f);

    /* Invalid structured combinations are discouraged, but are deliberately
       much cheaper than a terminal loss so early mask mistakes do not
       dominate the learning signal. */
    env.log.invalid_actions = 0.0f;
    env.log.truncations = 0.0f;
    env.log.n = 0.0f;
    env.log.score = 0.0f;
    env.log.perf = 0.0f;
    env.log.reward = 0.0f;
    env.max_episode_steps = 128;
    puf_reset(&env);
    actions[0] = (float)ACTION_REROLL_BOSS;
    actions[1] = 0.0f;
    actions[2] = 0.0f;
    for (int i = 0; i < MAX_SELECTION; ++i) actions[3 + i] = 0.0f;
    actions[8] = 0.0f;
    puf_step(&env);
    assert(env.log.invalid_actions == 1.0f);
    assert(env.log.action_counts[ACTION_REROLL_BOSS] == 1.0f);
    assert(terminal < 0.5f);
    assert(fabsf(reward - INVALID_ACTION_REWARD) < 1e-6f);
    env.max_episode_steps = 1;
    puf_reset(&env);
    puf_step(&env);
    assert(env.log.invalid_actions == 2.0f);
    assert(env.log.truncations == 1.0f);
    assert(terminal > 0.5f);
    assert(fabsf(reward - (INVALID_ACTION_REWARD + TIMEOUT_REWARD)) < 1e-6f);
    assert(fabsf(env.log.score - env.log.reward) < 1e-6f);
    assert(fabsf(env.log.reward - (INVALID_ACTION_REWARD + TIMEOUT_REWARD)) < 1e-6f);
    puf_reset(&env);
    env.log.invalid_actions = 0.0f;
    env.max_episode_steps = 128;

    int transitions = 0;
    int terminals = 0;
    for (; transitions < 1024; ++transitions) {
        LegalMasks masks = {0};
        LegalView view = {0};
        assert(legal_masks(&env.state, &masks) == OK);
        assert(legal_expand(&masks, &view) == OK);
        assert(view.group_count > 0);
        for (uint16_t group = 0; group < view.group_count; ++group) {
            if (view.groups[group].kind != LEGAL_SELECTION) continue;
            const SelectionFamily *family = &view.groups[group].selection;
            int entry = policy_selection_entry(
                family->type, family->primary);
            assert(entry >= 0);
            int base = POLICY_SELECTION_OFFSET +
                entry * POLICY_SELECTION_BYTES;
            assert(action_mask[base] == family->minimum);
            assert(action_mask[base + 1] == family->maximum);
            assert(mask_u64(action_mask, base + 2) == family->allowed_mask);
            assert(mask_u64(action_mask, base + 10) == family->required_mask);
        }
        Action action = {0};
        assert(legal_group_action(&view.groups[0], 0, &action) == OK);
        actions[0] = (float)action.type;
        actions[1] = (float)action.primary;
        actions[2] = (float)action.selection_count;
        for (int i = 0; i < MAX_SELECTION; ++i)
            actions[3 + i] = (float)action.selection[i];
        actions[8] = 0.0f;
        puf_step(&env);
        assert(isfinite(reward));
        assert(isfinite(terminal));
        if (terminal > 0.5f) ++terminals;
    }
    assert(env.log.invalid_actions == 0.0f);
    assert(env.log.action_counts[ACTION_SELECT_BLIND] > 0.0f);
    assert(terminals > 0);
    Dict log = {0};
    puf_log(&env.log, &log);
    assert(dict_find(&log, "play_actions"));
    assert(dict_find(&log, "discard_actions"));
    assert(dict_find(&log, "card_move_actions"));
    assert(dict_find(&log, "perf"));
    assert(dict_find(&log, "actions/play_hand"));
    printf("simulatro Puffer adapter: %d transitions, %d terminals, invalid=%.0f\n",
           transitions, terminals, env.log.invalid_actions);
    return 0;
}
