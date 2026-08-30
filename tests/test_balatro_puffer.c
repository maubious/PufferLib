#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "../ocean/balatro/balatro.h"

static uint64_t mask_u64(const unsigned char *mask, int offset) {
    uint64_t value = 0;
    memcpy(&value, mask + offset, sizeof(value));
    return value;
}

static const SelectionContract *selection_for(
        const LegalMasks *masks, uint8_t type, uint8_t primary) {
    if (type == ACTION_PLAY_HAND) return &masks->play;
    if (type == ACTION_DISCARD) return &masks->discard;
    if (type == ACTION_USE_CONSUMABLE && primary < OBS_MAX_CONSUMABLES)
        return &masks->consumable[primary];
    if (type == ACTION_BUY_AND_USE && primary < OBS_MAX_SHOP_MAIN)
        return &masks->shop[primary];
    if (type == ACTION_PICK_PACK_CARD && primary < OBS_MAX_PACK_CARDS)
        return &masks->pack[primary];
    return NULL;
}

static Action first_action(const LegalMasks *masks) {
    for (uint8_t type = 0; type < ACTION_TYPE_COUNT; ++type) {
        uint64_t primaries = masks->primary[type];
        if (!primaries) continue;
        uint8_t primary = (uint8_t)__builtin_ctzll(primaries);
        Action action = {.type = type, .primary = primary};
        const SelectionContract *selection = selection_for(masks, type, primary);
        if (!selection || !selection->valid) return action;

        uint64_t chosen = selection->required_hand;
        uint8_t target = selection->minimum;
        while ((uint8_t)__builtin_popcountll(chosen) < target) {
            uint64_t available = selection->allowed_hand & ~chosen;
            assert(available);
            chosen |= UINT64_C(1) << __builtin_ctzll(available);
        }
        for (uint8_t index = 0; index < MAX_HAND; ++index)
            if (chosen & (UINT64_C(1) << index))
                action.selection[action.selection_count++] = index;
        return action;
    }
    assert(!"legal mask has no action");
    return (Action){0};
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
    DictItem *potential = dict_item(&kwargs, "potential_scale");
    potential->value = 255.0;
    env.agents[0].observations = observation;
    env.agents[0].actions = actions;
    env.agents[0].rewards = &reward;
    env.agents[0].terminals = &terminal;
    env.agents[0].action_mask = action_mask;
    puf_init(&env, &kwargs);
    assert(env.config.potential_scale == 255);
    puf_reset(&env);
    assert(env.legal_masks.primary[ACTION_SWAP_HAND_LEFT] == 0);
    assert(env.legal_masks.primary[ACTION_SWAP_HAND_RIGHT] == 0);
    assert(env.legal_masks.primary[ACTION_SORT_HAND_RANK] == 0);
    assert(env.legal_masks.primary[ACTION_SORT_HAND_SUIT] == 0);
    assert(env.legal_masks.primary[ACTION_SWAP_JOKERS_LEFT] == 0);
    assert(env.legal_masks.primary[ACTION_SWAP_JOKERS_RIGHT] == 0);
    assert(action_mask[ACTION_SELECT_BLIND] == 1);
    assert(mask_u64(action_mask, POLICY_PRIMARY_OFFSET +
        ACTION_SELECT_BLIND * POLICY_PRIMARY_BYTES) == 1);

    /* A one-step limit must truncate a nonterminal initial blind-select step. */
    Action timeout_action = {.type = ACTION_SELECT_BLIND};
    State expected_state = env.state;
    StepResult expected_result = {0};
    assert(apply_step(&expected_state, &timeout_action, &env.legal_masks, &expected_result) == OK);
    actions[0] = (float)timeout_action.type;
    actions[1] = (float)timeout_action.primary;
    actions[2] = (float)timeout_action.selection_count;
    for (int i = 0; i < MAX_SELECTION; ++i) actions[3 + i] = (float)timeout_action.selection[i];
    puf_step(&env);
    assert(env.log.truncations == 1.0f);
    assert(fabsf(env.log.score - (expected_result.reward + TIMEOUT_REWARD)) < 1e-6f);
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
    env.max_episode_steps = 128;
    puf_reset(&env);
    actions[0] = (float)ACTION_REROLL_BOSS;
    actions[1] = 0.0f;
    actions[2] = 0.0f;
    for (int i = 0; i < MAX_SELECTION; ++i) actions[3 + i] = 0.0f;
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
    /* Score accumulates episode returns: the invalid reroll-boss press paid
       INVALID_ACTION_REWARD, then truncation added TIMEOUT_REWARD. */
    assert(fabsf(env.log.score - (INVALID_ACTION_REWARD + TIMEOUT_REWARD)) < 1e-6f);
    puf_reset(&env);
    env.log.invalid_actions = 0.0f;
    env.max_episode_steps = 128;

    int transitions = 0;
    int terminals = 0;
    for (; transitions < 1024; ++transitions) {
        LegalMasks masks = {0};
        Observation observation = {0};
        assert(observe(&env.state, &observation, &masks) == OK);
        assert(masks.primary[ACTION_SELECT_BLIND] ||
               masks.primary[ACTION_PLAY_HAND] ||
               masks.primary[ACTION_DISCARD] ||
               masks.primary[ACTION_CASH_OUT] ||
               masks.primary[ACTION_NEXT_ROUND] ||
               masks.primary[ACTION_SKIP_PACK]);
        for (uint8_t type = 0; type < ACTION_TYPE_COUNT; ++type)
            for (uint8_t primary = 0; primary < 64; ++primary) {
                if (!(masks.primary[type] & (UINT64_C(1) << primary))) continue;
                const SelectionContract *selection = selection_for(&masks, type, primary);
                if (!selection || !selection->valid) continue;
                int entry = policy_selection_entry(type, primary);
                assert(entry >= 0);
                int base = POLICY_SELECTION_OFFSET + entry * POLICY_SELECTION_BYTES;
                assert(action_mask[base] == selection->minimum);
                assert(action_mask[base + 1] == selection->maximum);
                assert(mask_u64(action_mask, base + 2) == selection->allowed_hand);
                assert(mask_u64(action_mask, base + 10) == selection->required_hand);
            }
        Action action = first_action(&masks);
        actions[0] = (float)action.type;
        actions[1] = (float)action.primary;
        actions[2] = (float)action.selection_count;
        for (int i = 0; i < MAX_SELECTION; ++i)
            actions[3 + i] = (float)action.selection[i];
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
    assert(dict_find(&log, "score"));
    assert(dict_find(&log, "perf"));
    assert(dict_find(&log, "wins"));
    printf("simulatro Puffer adapter: %d transitions, %d terminals, invalid=%.0f\n",
           transitions, terminals, env.log.invalid_actions);
    return 0;
}
