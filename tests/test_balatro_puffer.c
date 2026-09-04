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

static float completion_reward(uint8_t blind_on_deck, uint8_t skipped) {
    Config config;
    default_config(&config);
    config.shaped_reward = 1;
    config.progress_reward = 0.0f;
    config.blind_bonus = 0.2f;
    config.ante_bonus = 1.0f;
    config.win_bonus = 0.0f;
    config.loss_penalty = 0.0f;
    config.money_reward = 0.0f;
    State state;
    assert(init(&state, &config, 3 + blind_on_deck + skipped) == OK);
    state.phase = PHASE_SELECTING_HAND;
    state.blind_on_deck = blind_on_deck;
    state.blind_id = blind_on_deck == 0 ? BLIND_BL_SMALL
        : blind_on_deck == 1 ? BLIND_BL_BIG : BLIND_BL_CLUB;
    state.blind_skipped_mask = skipped;
    state.blind_disabled = 0;
    state.hands_left = 1;
    state.blind_chips = 1;
    state.deck_count = 0;
    state.discard_count = 0;
    state.hand_count = 1;
    state.hand[0] = (Card){
        .center_id = CENTER_C_BASE,
        .rank = 14,
        .suit = HEARTS,
        .sort_id = 1,
    };
    Action play = {
        .type = ACTION_PLAY_HAND,
        .selection_count = 1,
        .selection = {0},
    };
    StepResult result;
    assert(apply_step(&state, &play, NULL, &result) == OK);
    assert(state.phase == PHASE_ROUND_EVAL);
    return result.reward;
}

int main(void) {
    State credit_state = {0};
    credit_state.jokers[0] = (Card){.center_id = CENTER_J_CREDIT_CARD};
    credit_state.joker_count = 1;
    assert(can_afford(&credit_state, 20));
    assert(!can_afford(&credit_state, 21));
    credit_state.jokers[0].flags = CARD_DEBUFFED;
    assert(!can_afford(&credit_state, 1));

    float small_reward = completion_reward(0, 0);
    float big_reward = completion_reward(1, 0);
    assert(fabsf(small_reward - 0.2f) < 1e-6f);
    assert(fabsf(big_reward - 0.2f) < 1e-6f);
    assert(fabsf(completion_reward(2, 0) - 0.6f) < 1e-6f);
    assert(fabsf(completion_reward(2, 1) - 0.8f) < 1e-6f);
    assert(fabsf(completion_reward(2, 3) - 1.0f) < 1e-6f);
    assert(fabsf(small_reward + big_reward + completion_reward(2, 0) - 1.0f) < 1e-6f);
    assert(fabsf(big_reward + completion_reward(2, 1) - 1.0f) < 1e-6f);
    Config skip_config;
    default_config(&skip_config);
    skip_config.shaped_reward = 1;
    State skip_state;
    assert(init(&skip_state, &skip_config, 9) == OK);
    Action skip = {.type = ACTION_SKIP_BLIND};
    StepResult skip_result;
    assert(apply_step(&skip_state, &skip, NULL, &skip_result) == OK);
    assert(skip_result.reward == 0.0f);
    assert(skip_state.blind_on_deck == 1);
    assert(skip_state.blind_skipped_mask == 1);

    Config chaos_config;
    default_config(&chaos_config);
    State chaos_state;
    assert(init(&chaos_state, &chaos_config, 1) == OK);
    chaos_state.phase = PHASE_SHOP;
    chaos_state.dollars = 20;
    chaos_state.shop_main[0] = (Card){
        .center_id = CENTER_J_CHAOS,
        .cost = 4,
        .sell_cost = 2,
    };
    chaos_state.shop_main_count = 1;
    Observation chaos_observation;
    LegalMasks chaos_masks;
    assert(observe(&chaos_state, &chaos_observation, &chaos_masks) == OK);
    StepResult chaos_result;
    Action buy_chaos = {.type = ACTION_BUY_CARD};
    assert(apply_step(&chaos_state, &buy_chaos, &chaos_masks, &chaos_result) == OK);
    assert(chaos_state.joker_count == 1);
    assert(chaos_state.jokers[0].center_id == CENTER_J_CHAOS);
    assert(chaos_state.free_rerolls == 1);
    assert(chaos_state.reroll_cost == 0);
    int32_t dollars_after_chaos = chaos_state.dollars;
    assert(observe(&chaos_state, &chaos_observation, &chaos_masks) == OK);
    assert(chaos_observation.globals.free_rerolls == 1);
    assert(chaos_observation.globals.reroll_cost == 0);
    assert(chaos_masks.primary[ACTION_REROLL] == 1);
    Action reroll = {.type = ACTION_REROLL};
    assert(apply_step(&chaos_state, &reroll, &chaos_masks, &chaos_result) == OK);
    assert(chaos_state.dollars == dollars_after_chaos);
    assert(chaos_state.free_rerolls == 0);
    assert(chaos_state.reroll_cost == chaos_state.reroll_base);
    assert(observe(&chaos_state, &chaos_observation, &chaos_masks) == OK);
    assert(apply_step(&chaos_state, &reroll, &chaos_masks, &chaos_result) == OK);
    assert(chaos_state.dollars == dollars_after_chaos - chaos_state.reroll_base);
    assert(chaos_state.reroll_cost == chaos_state.reroll_base + 1);

    State mouth_state;
    assert(init(&mouth_state, &chaos_config, 2) == OK);
    mouth_state.phase = PHASE_SELECTING_HAND;
    mouth_state.blind_id = BLIND_BL_MOUTH;
    mouth_state.blind_disabled = 0;
    mouth_state.blind_only_hand = UINT8_MAX;
    mouth_state.hands_left = 2;
    mouth_state.hand_size = 3;
    mouth_state.blind_chips = 1000000;
    mouth_state.deck_count = 0;
    mouth_state.discard_count = 0;
    mouth_state.hand_count = 3;
    mouth_state.hand[0] = (Card){.center_id = CENTER_C_BASE, .rank = 2, .suit = HEARTS, .sort_id = 1};
    mouth_state.hand[1] = (Card){.center_id = CENTER_C_BASE, .rank = 2, .suit = SPADES, .sort_id = 2};
    mouth_state.hand[2] = (Card){.center_id = CENTER_C_BASE, .rank = 13, .suit = CLUBS, .sort_id = 3};
    Observation mouth_observation;
    LegalMasks mouth_masks;
    assert(observe(&mouth_state, &mouth_observation, &mouth_masks) == OK);
    Action play_pair = {
        .type = ACTION_PLAY_HAND,
        .selection_count = 2,
        .selection = {0, 1},
    };
    StepResult mouth_result;
    assert(apply_step(&mouth_state, &play_pair, &mouth_masks, &mouth_result) == OK);
    assert(mouth_state.blind_only_hand == PAIR);
    assert(mouth_state.last_hand_type == PAIR);
    assert(mouth_state.last_hand_score > 0);
    double pair_chips = mouth_state.chips;
    mouth_state.jokers[0] = (Card){.center_id = CENTER_J_MIDAS_MASK};
    mouth_state.jokers[1] = (Card){.center_id = CENTER_J_VAGABOND};
    mouth_state.joker_count = 2;
    assert(observe(&mouth_state, &mouth_observation, &mouth_masks) == OK);
    Action play_high_card = {
        .type = ACTION_PLAY_HAND,
        .selection_count = 1,
        .selection = {0},
    };
    assert(apply_step(&mouth_state, &play_high_card, &mouth_masks, &mouth_result) == OK);
    assert(mouth_state.last_hand_type == HIGH_CARD);
    assert(mouth_state.last_hand_score == 0);
    assert(mouth_state.chips == pair_chips);
    assert(mouth_state.discard[2].enhancement == ENHANCEMENT_NONE);
    assert(mouth_state.consumable_count == 0);

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
    DictItem *reorder = dict_item(&kwargs, "reorder_actions");
    reorder->value = 1.0;
    DictItem *money = dict_item(&kwargs, "money_reward");
    money->value = 0.1;
    DictItem *seed = dict_item(&kwargs, "seed");
    seed->value = 1234.0;
    env.agents[0].observations = observation;
    env.agents[0].actions = actions;
    env.agents[0].rewards = &reward;
    env.agents[0].terminals = &terminal;
    env.agents[0].action_mask = action_mask;
    puf_init(&env, &kwargs);
    assert(env.rng == 1234);
    Env adjacent = {.rng = 1};
    puf_init(&adjacent, &kwargs);
    assert(adjacent.rng == 1235);
    assert(env.config.potential_scale == 255);
    assert(env.reorder_actions == 1);
    assert(env.config.money_reward == 0.1f);
    unsigned int expected_rng = 1234;
    uint64_t expected_seed = ((uint64_t)rand_r(&expected_rng) << 32) |
        rand_r(&expected_rng);
    puf_reset(&env);
    assert(env.state.numeric_seed == expected_seed);
    assert(env.rng == expected_rng);
    uint64_t first_seed = env.state.numeric_seed;
    puf_reset(&env);
    assert(env.state.numeric_seed != first_seed);
    assert(action_mask[ACTION_SELECT_BLIND] == 1);
    assert(mask_u64(action_mask, POLICY_PRIMARY_OFFSET +
        ACTION_SELECT_BLIND * POLICY_PRIMARY_BYTES) == 1);

    Action select_action = {.type = ACTION_SELECT_BLIND};
    StepResult select_result = {0};
    assert(apply_step(&env.state, &select_action, &env.legal_masks, &select_result) == OK);
    assert(puffer_observe(&env) == OK);
    assert(action_mask[POLICY_ORDER_HAND_COUNT_OFFSET] == env.state.hand_count);
    assert(action_mask[POLICY_ORDER_JOKER_COUNT_OFFSET] == env.state.joker_count);
    assert(action_mask[POLICY_ORDER_ENABLED_OFFSET] == 1);
    assert(env.legal_masks.primary[ACTION_SWAP_HAND_LEFT]);
    assert(env.legal_masks.primary[ACTION_SWAP_HAND_RIGHT]);
    assert(env.legal_masks.primary[ACTION_SORT_HAND_RANK] == 1);
    assert(env.legal_masks.primary[ACTION_SORT_HAND_SUIT] == 1);
    /* Reordering is a policy suffix, so primitive ordering transitions are
       retained in the core mask but never exposed to the learned policy. */
    assert(action_mask[ACTION_SWAP_HAND_LEFT] == 0);
    assert(action_mask[ACTION_SWAP_HAND_RIGHT] == 0);
    assert(action_mask[ACTION_SORT_HAND_RANK] == 0);
    assert(action_mask[ACTION_SORT_HAND_SUIT] == 0);
    /* Force one inversion: the core still reports the primitive action. */
    Card swap_tmp = env.state.hand[0];
    env.state.hand[0] = env.state.hand[1];
    env.state.hand[1] = swap_tmp;
    assert(puffer_observe(&env) == OK);
    uint64_t swap_bits = mask_u64(action_mask, POLICY_PRIMARY_OFFSET +
        ACTION_SWAP_HAND_LEFT * POLICY_PRIMARY_BYTES);
    assert(swap_bits == 0);
    assert(action_mask[ACTION_SORT_HAND_RANK] == 0);
    /* Sorting clears the swap and sort options in the learned mask again. */
    Action sort_rank = {.type = ACTION_SORT_HAND_RANK};
    assert(apply_step(&env.state, &sort_rank, &env.legal_masks, &select_result) == OK);
    assert(puffer_observe(&env) == OK);
    assert(mask_u64(action_mask, POLICY_PRIMARY_OFFSET +
        ACTION_SWAP_HAND_LEFT * POLICY_PRIMARY_BYTES) == 0);
    assert(action_mask[ACTION_SORT_HAND_RANK] == 0);
    /* A joker permutation is applied before the consequential action, and
       the sell target follows its stable pre-permutation identity. */
    env.max_episode_steps = 0;
    env.state.jokers[0] = (Card){
        .center_id = CENTER_J_BANNER, .cost = 5, .sell_cost = 4};
    env.state.jokers[1] = (Card){
        .center_id = CENTER_J_MIDAS_MASK, .cost = 5, .sell_cost = 2};
    env.state.joker_count = 2;
    assert(puffer_observe(&env) == OK);
    reward = 0.0f;
    actions[0] = (float)ACTION_SELL_JOKER;
    actions[1] = 0.0f;
    actions[2] = 0.0f;
    for (int i = 0; i < MAX_SELECTION; ++i) actions[3 + i] = 0.0f;
    for (int i = 0; i < env.state.hand_count; ++i)
        actions[BALATRO_ORDER_HAND_OFFSET + i] = (float)i;
    for (int i = 0; i < env.state.joker_count; ++i)
        actions[BALATRO_ORDER_JOKER_OFFSET + i] = (float)(1 - i);
    Client render_client = {0};
    env.client = &render_client;
    render_client.paused = true;
    puf_step(&env);
    assert(env.state.joker_count == 2);
    assert(reward == 0.0f);
    assert(terminal == 0.0f);
    render_client.paused = false;
    puf_step(&env);
    env.client = NULL;
    assert(fabsf(reward - 0.1f * 4.0f) < 1e-6f);

    /* Death uses the final hand order directionally: after reversing the
       hand, the stable old slot 1 becomes the left target and old slot 0 the
       right source. */
    env.state.phase = PHASE_SELECTING_HAND;
    env.state.hands_left = 1;
    env.state.discards_left = 0;
    env.state.hand_count = 2;
    env.state.hand[0] = (Card){
        .center_id = CENTER_C_BASE, .rank = 2, .suit = HEARTS, .sort_id = 10};
    env.state.hand[1] = (Card){
        .center_id = CENTER_C_BASE, .rank = 14, .suit = SPADES, .sort_id = 11};
    env.state.consumables[0] = (Card){.center_id = CENTER_C_DEATH};
    env.state.consumable_count = 1;
    assert(puffer_observe(&env) == OK);
    reward = 0.0f;
    actions[0] = (float)ACTION_USE_CONSUMABLE;
    actions[1] = 0.0f;
    actions[2] = 2.0f;
    actions[3] = 0.0f;
    actions[4] = 1.0f;
    for (int i = 0; i < MAX_SELECTION - 2; ++i) actions[5 + i] = 0.0f;
    for (int i = 0; i < env.state.hand_count; ++i)
        actions[BALATRO_ORDER_HAND_OFFSET + i] = (float)(1 - i);
    for (int i = 0; i < env.state.joker_count; ++i)
        actions[BALATRO_ORDER_JOKER_OFFSET + i] = (float)i;
    puf_step(&env);
    assert(env.state.hand[0].rank == 2);
    assert(env.state.hand[1].rank == 2);
    assert(env.state.hand[0].suit == HEARTS);
    assert(env.state.hand[1].suit == HEARTS);
    assert(env.state.consumable_count == 0);

    assert(env.state.joker_count == 1);
    assert(env.state.jokers[0].center_id == CENTER_J_MIDAS_MASK);
    /* Money shaping: a sell yields money_reward x sell_cost (dollars 4 -> 8). */
    env.max_episode_steps = 0;
    env.state.jokers[0] = (Card){.center_id = CENTER_J_BANNER, .cost = 5, .sell_cost = 4};
    env.state.joker_count = 1;
    assert(puffer_observe(&env) == OK);
    reward = 0.0f;
    actions[0] = (float)ACTION_SELL_JOKER;
    actions[1] = 0.0f;
    actions[2] = 0.0f;
    for (int i = 0; i < MAX_SELECTION; ++i) actions[3 + i] = 0.0f;
    for (int i = 0; i < env.state.hand_count; ++i)
        actions[BALATRO_ORDER_HAND_OFFSET + i] = (float)i;
    for (int i = 0; i < env.state.joker_count; ++i)
        actions[BALATRO_ORDER_JOKER_OFFSET + i] = (float)i;
    puf_step(&env);
    assert(fabsf(reward - 0.1f * 4.0f) < 1e-6f);
    env.max_episode_steps = 1;
    env.reorder_actions = 0;
    assert(puffer_observe(&env) == OK);
    assert(env.legal_masks.primary[ACTION_SWAP_HAND_LEFT]);
    assert(env.legal_masks.primary[ACTION_SORT_HAND_RANK] == 1);
    assert(action_mask[ACTION_SWAP_HAND_LEFT] == 0);
    assert(action_mask[ACTION_SORT_HAND_RANK] == 0);
    assert(action_mask[POLICY_ORDER_ENABLED_OFFSET] == 0);
    env.reorder_actions = 1;
    puf_reset(&env);

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
        for (int i = 0; i < env.state.hand_count; ++i)
            actions[BALATRO_ORDER_HAND_OFFSET + i] = (float)i;
        for (int i = 0; i < env.state.joker_count; ++i)
            actions[BALATRO_ORDER_JOKER_OFFSET + i] = (float)i;
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

    /* The Serpent: initial draw is the normal hand size; every draw after a
       play or discard is exactly 3 cards and the hand grows past its size.
       (Mirrors the real game's draw_from_deck_to_hand serpent branch.) */
    puf_reset(&env);
    env.state.blind_on_deck = 2;
    env.state.next_boss_id = BLIND_BL_SERPENT;
    Action serpent_select = {.type = ACTION_SELECT_BLIND};
    assert(apply_step(&env.state, &serpent_select, &env.legal_masks, &select_result) == OK);
    assert(env.state.blind_id == BLIND_BL_SERPENT);
    assert(env.state.hand_count == 8);
    assert(env.state.deck_count == 44);
    Action serpent_play = {.type = ACTION_PLAY_HAND, .selection_count = 2, .selection = {0, 1}};
    assert(apply_step(&env.state, &serpent_play, &env.legal_masks, &select_result) == OK);
    assert(env.state.hand_count == 9);  /* 8 - 2 played + 3 drawn */
    assert(env.state.deck_count == 41);
    Action serpent_discard = {.type = ACTION_DISCARD, .selection_count = 1, .selection = {0}};
    assert(apply_step(&env.state, &serpent_discard, &env.legal_masks, &select_result) == OK);
    assert(env.state.hand_count == 11); /* 9 - 1 + 3, exceeds hand size 8 */
    assert(env.state.deck_count == 38);
    printf("serpent: initial 8, +3 after play, +3 after discard, hand grows to %d\n",
           env.state.hand_count);

    /* Drawing a face-down card must not reveal its identity by subtracting
       consecutive deck observations. It remains in the anonymous unseen
       multiset while its hand slot exposes only the facedown flag. Exercise
       both the plain-card histogram and the explicit modified-card stream. */
    for (int modified = 0; modified < 2; ++modified) {
        Config config;
        default_config(&config);
        State state;
        assert(init(&state, &config, (uint64_t)(100 + modified)) == OK);
        assert(state.hand_count == 0);
        assert(state.deck_count > 0);
        Card *source = &state.deck[state.deck_count - 1];
        if (modified) {
            source->enhancement = ENHANCEMENT_GLASS;
            source->edition = EDITION_FOIL;
            source->seal = SEAL_RED;
            source->perma_bonus = 17;
        }
        Observation before;
        Observation after;
        LegalMasks masks;
        assert(observe(&state, &before, &masks) == OK);
        Card hidden = *source;
        state.deck_count--;
        hidden.flags |= CARD_FACEDOWN;
        state.hand[state.hand_count++] = hidden;
        assert(observe(&state, &after, &masks) == OK);
        assert(memcmp(before.plain_deck, after.plain_deck,
            sizeof(before.plain_deck)) == 0);
        assert(before.counts.special_count == after.counts.special_count);
        assert(memcmp(before.specials, after.specials,
            sizeof(before.specials)) == 0);
        assert(after.hand[0].attributes == 0);
        assert(after.hand[0].perma_bonus == 0);
    }

    /* Test high-ante reward components */
    {
        /* 1. Ante Escalation Test */
        Config config;
        default_config(&config);
        config.shaped_reward = 1;
        config.progress_reward = 0.0f;
        config.blind_bonus = 0.2f;
        config.ante_bonus = 1.0f;
        config.win_bonus = 0.0f;
        config.loss_penalty = 0.0f;
        config.money_reward = 0.0f;
        config.ante_escalation = 0.25f;

        State state;
        assert(init(&state, &config, 42) == OK);
        state.phase = PHASE_SELECTING_HAND;
        state.blind_on_deck = 2; /* Boss blind */
        state.ante = 5;
        state.blind_skipped_mask = 0;
        state.blind_chips = 1;
        state.hands_left = 1;
        state.hand_count = 1;
        state.hand[0] = (Card){.center_id = CENTER_C_BASE, .rank = 14, .suit = HEARTS, .sort_id = 1};

        Action play = {.type = ACTION_PLAY_HAND, .selection_count = 1, .selection = {0}};
        StepResult result;
        assert(apply_step(&state, &play, NULL, &result) == OK);
        /* Base boss reward: 1.0 - 0.2 * 2 = 0.6.
           Completed ante was 5. Ante escalation factor: 1.0 + 0.25 * (5 - 1) = 2.0.
           Expected ante reward = 0.6 * 2.0 = 1.2. */
        assert(fabsf(result.reward - 1.2f) < 1e-5f);

        /* Ante 8 geometric escalation */
        State state8;
        assert(init(&state8, &config, 48) == OK);
        state8.phase = PHASE_SELECTING_HAND;
        state8.blind_on_deck = 2;
        state8.ante = 8;
        state8.blind_skipped_mask = 0;
        state8.blind_chips = 1;
        state8.hands_left = 1;
        state8.hand_count = 1;
        state8.hand[0] = (Card){.center_id = CENTER_C_BASE, .rank = 14, .suit = HEARTS, .sort_id = 1};
        assert(apply_step(&state8, &play, NULL, &result) == OK);
        /* Ante 8 factor: 1.0 + 0.25 * 4.0 * 1.0 = 2.0 -> 0.6 * 2.0 = 1.2 */
        assert(fabsf(result.reward - 1.2f) < 1e-5f);

        /* Ante 9 geometric escalation (doubles the ante 8 term) */
        State state9;
        assert(init(&state9, &config, 49) == OK);
        state9.phase = PHASE_SELECTING_HAND;
        state9.blind_on_deck = 2;
        state9.ante = 9;
        state9.blind_skipped_mask = 0;
        state9.blind_chips = 1;
        state9.hands_left = 1;
        state9.hand_count = 1;
        state9.hand[0] = (Card){.center_id = CENTER_C_BASE, .rank = 14, .suit = HEARTS, .sort_id = 1};
        assert(apply_step(&state9, &play, NULL, &result) == OK);
        /* Ante 9 factor: 1.0 + 0.25 * 4.0 * 2.0 = 3.0 -> 0.6 * 3.0 = 1.8 */
        assert(fabsf(result.reward - 1.8f) < 1e-5f);
    }
    {
        /* 2. High-Pass Log-Power Reward Test */
        Config config;
        default_config(&config);
        config.shaped_reward = 1;
        config.progress_reward = 0.0f;
        config.blind_bonus = 0.0f;
        config.ante_bonus = 0.0f;
        config.win_bonus = 0.0f;
        config.loss_penalty = 0.0f;
        config.power_reward = 0.5f;

        State state;
        assert(init(&state, &config, 43) == OK);
        state.phase = PHASE_SELECTING_HAND;
        state.blind_on_deck = 0;
        state.ante = 1;
        state.blind_chips = 10;
        state.hands_left = 1;
        state.hand_count = 1;
        /* Ace of Hearts: score = 16. 16 <= 10000 -> power reward must be 0! */
        state.hand[0] = (Card){.center_id = CENTER_C_BASE, .rank = 14, .suit = HEARTS, .sort_id = 1};

        Action play = {.type = ACTION_PLAY_HAND, .selection_count = 1, .selection = {0}};
        StepResult result;
        assert(apply_step(&state, &play, NULL, &result) == OK);
        assert(fabsf(result.reward - 0.0f) < 1e-5f);
        assert(state.peak_hand_log == 0.0);
    }
    {
        /* 3. Hand Efficiency Reward Test */
        Config config;
        default_config(&config);
        config.shaped_reward = 1;
        config.progress_reward = 0.0f;
        config.blind_bonus = 0.0f;
        config.ante_bonus = 0.0f;
        config.eff_reward = 0.4f;

        State state;
        assert(init(&state, &config, 44) == OK);
        state.phase = PHASE_SELECTING_HAND;
        state.blind_on_deck = 0;
        state.ante = 1;
        state.blind_chips = 1;
        state.hands_left = 4;
        state.hands_played = 0;
        state.hand_count = 1;
        state.hand[0] = (Card){.center_id = CENTER_C_BASE, .rank = 14, .suit = HEARTS, .sort_id = 1};

        Action play = {.type = ACTION_PLAY_HAND, .selection_count = 1, .selection = {0}};
        StepResult result;
        assert(apply_step(&state, &play, NULL, &result) == OK);
        /* During play, hands_left became 3, hands_played became 1.
           Total hands = 3 + 1 = 4. hands_left = 3.
           Efficiency = 3 / 4 = 0.75.
           Expected reward = 0.4 * 0.75 = 0.3. */
        assert(fabsf(result.reward - 0.3f) < 1e-5f);
    }
    {
        /* 4. Cashout Interest Reward Test */
        Config config;
        default_config(&config);
        config.shaped_reward = 1;
        config.interest_reward = 0.2f;

        State state;
        assert(init(&state, &config, 45) == OK);
        state.phase = PHASE_ROUND_EVAL;
        state.dollars = 20;
        state.round_earnings = 5; /* Total dollars after cashout = 25 */
        state.interest_cap = 25;

        Action cashout = {.type = ACTION_CASH_OUT};
        StepResult result;
        assert(apply_step(&state, &cashout, NULL, &result) == OK);
        /* After cashout, state.dollars = 25, interest_cap = 25.
           Reward = 0.2 * (25 / 25) = 0.2. */
        assert(fabsf(result.reward - 0.2f) < 1e-5f);
    }

    return 0;
}
