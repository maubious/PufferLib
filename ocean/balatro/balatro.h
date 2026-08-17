#ifndef PUFFER_ENV_H
#define PUFFER_ENV_H

#define PUFFER_BALATRO
#define PUFFER_PACKED_OBS
/* State curriculum support: by-value State state member + refresh hook
   (src/curriculum.cu is compiled in when this is defined). */
#define PUFFER_CURRICULUM

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "pufferenv.h"
#include <balatro_core.h>
#include "policy.h"

#define OBS_SIZE ((int)sizeof(Observation))
#define NUM_ATNS 8
#define ACT_SIZES {23, 64, 6, 64, 64, 64, 64, 64}
#define ACTION_MASK_SIZE POLICY_MASK_SIZE
#define INVALID_ACTION_REWARD (-0.002f)
#define TIMEOUT_REWARD (-1.0f)

typedef unsigned char obs_t;

typedef struct Log {
	float score;
	float perf;
	float wins;
	float ante;
	float invalid_actions;
	float truncations;
	float n;
	/* Episode-end state snapshots (autopsy), summed at terminal. */
	float end_dollars;
	float end_dollars_won;
	float end_jokers;
	float end_consumables;
	float end_hand_levels;
	float end_deck;
	float ep_length;
	float action_counts[ACTION_TYPE_COUNT];
} Log;

static const char *const action_log_names[ACTION_TYPE_COUNT] = {
    "actions/play_hand",
    "actions/discard",
    "actions/select_blind",
    "actions/skip_blind",
    "actions/cash_out",
    "actions/reroll",
    "actions/next_round",
    "actions/skip_pack",
    "actions/buy_card",
    "actions/sell_joker",
    "actions/sell_consumable",
    "actions/use_consumable",
    "actions/redeem_voucher",
    "actions/open_booster",
    "actions/pick_pack_card",
    "actions/swap_jokers_left",
    "actions/swap_jokers_right",
    "actions/swap_hand_left",
    "actions/swap_hand_right",
    "actions/sort_hand_rank",
    "actions/sort_hand_suit",
    "actions/buy_and_use",
    "actions/reroll_boss",
};

typedef struct Env {
    Log log;
    int num_agents;
    int tag;
    int boundary_reached;
    float episode_reward;
    unsigned int rng;
    uint32_t episode_steps;
    uint32_t max_episode_steps;
    Agent agents[1];
    Config config;
    State state;
    LegalMasks legal_masks;
    float invalid_action_reward;
} Env;

static inline float hand_level_sum(const State *state) {
    float levels = 0.0f;
    for (int i = 0; i < HAND_COUNT; ++i) levels += (float)state->hand_levels[i];
    return levels;
}

static inline void log_episode_end(Env *env, int won) {
    const State *state = &env->state;
    env->log.end_dollars += (float)state->dollars;
    if (won) env->log.end_dollars_won += (float)state->dollars;
    env->log.end_jokers += (float)state->joker_count;
    env->log.end_consumables += (float)state->consumable_count;
    env->log.end_hand_levels += hand_level_sum(state);
    env->log.end_deck += (float)state->deck_count;
    env->log.ep_length += (float)env->episode_steps;
}

static inline void disable_move_actions(LegalMasks *legal) {
    static const int move_actions[] = {
        ACTION_SWAP_HAND_LEFT, ACTION_SWAP_HAND_RIGHT,
        ACTION_SORT_HAND_RANK, ACTION_SORT_HAND_SUIT,
        ACTION_SWAP_JOKERS_LEFT, ACTION_SWAP_JOKERS_RIGHT,
    };
    for (unsigned int i = 0; i < sizeof(move_actions) / sizeof(move_actions[0]); ++i) {
        legal->action_type[move_actions[i]] = 0;
        legal->primary[move_actions[i]] = 0;
    }
}

static inline void store_u64(unsigned char *out, uint64_t value) {
    memcpy(out, &value, sizeof(value));
}

static inline void store_selection(
        unsigned char *mask, int entry,
        const ObservedSelection *selection) {
    if (!selection || !selection->valid || entry < 0) return;
    unsigned char *out = mask + POLICY_SELECTION_OFFSET +
        entry * POLICY_SELECTION_BYTES;
    out[0] = selection->minimum;
    out[1] = selection->maximum;
    store_u64(out + 2, selection->allowed_hand);
    store_u64(out + 10, selection->required_hand);
}

static void puffer_mask(const LegalMasks *legal, unsigned char *mask,
        const State *state) {
    memset(mask, 0, ACTION_MASK_SIZE);

    for (int type = 0; type < ACTION_TYPE_COUNT; ++type) {
        if (!legal->action_type[type]) continue;
        mask[type] = 1;
        store_u64(mask + POLICY_PRIMARY_OFFSET +
            type * POLICY_PRIMARY_BYTES, legal->primary[type]);
    }

    /* Per-option card attrs for the AR selection heads: (suit << 4) | rank per
       hand slot. Zero past hand_count; options are masked illegal there. */
    int attr_n = state->hand_count < POLICY_CARD_ATTR_BYTES
        ? state->hand_count : POLICY_CARD_ATTR_BYTES;
    for (int i = 0; i < attr_n; ++i) {
        mask[POLICY_CARD_ATTR_OFFSET + i] =
            (unsigned char)((state->hand[i].suit << POLICY_CARD_ATTR_SUIT_SHIFT)
                | (state->hand[i].rank & POLICY_CARD_ATTR_RANK_MASK));
    }

    store_selection(mask,
        policy_selection_entry(ACTION_PLAY_HAND, 0),
        &legal->play);
    store_selection(mask,
        policy_selection_entry(ACTION_DISCARD, 0),
        &legal->discard);
    for (int i = 0; i < OBS_MAX_CONSUMABLES; ++i)
        store_selection(mask,
            policy_selection_entry(ACTION_USE_CONSUMABLE, i),
            &legal->consumable[i]);
    for (int i = 0; i < OBS_MAX_SHOP_MAIN; ++i)
        store_selection(mask,
            policy_selection_entry(ACTION_BUY_AND_USE, i),
            &legal->shop[i]);
    for (int i = 0; i < OBS_MAX_PACK_CARDS; ++i)
        store_selection(mask,
            policy_selection_entry(ACTION_PICK_PACK_CARD, i),
            &legal->pack[i]);
}

static int puffer_observe(Env *env) {
    obs_t *out = (obs_t *)env->agents[0].observations;
    int error = observe(&env->state, (Observation *)out, &env->legal_masks);
    if (error == OK) {
        disable_move_actions(&env->legal_masks);
        if (env->agents[0].action_mask)
            puffer_mask(&env->legal_masks, env->agents[0].action_mask,
                &env->state);
    } else {
        memset(out, 0, OBS_SIZE * sizeof(*out));
    }
    return error;
}

/* State-curriculum restore hook (src/curriculum.cu): rebuild the observation
   and legal/action masks from env->state after a state restore, and reset the
   episode bookkeeping so the restored episode starts a fresh accounting. */
static inline void puffer_state_refresh(Env *env) {
    puffer_observe(env);
    env->episode_steps = 0;
    env->episode_reward = 0.0f;
    env->boundary_reached = 0;
}

static inline const ObservedSelection *cached_selection(
        const LegalMasks *legal, uint8_t type, uint8_t primary) {
    if (type == ACTION_PLAY_HAND) return &legal->play;
    if (type == ACTION_DISCARD) return &legal->discard;
    if (type == ACTION_USE_CONSUMABLE && primary < OBS_MAX_CONSUMABLES)
        return &legal->consumable[primary];
    if (type == ACTION_BUY_AND_USE && primary < OBS_MAX_SHOP_MAIN)
        return &legal->shop[primary];
    if (type == ACTION_PICK_PACK_CARD && primary < OBS_MAX_PACK_CARDS)
        return &legal->pack[primary];
    return NULL;
}

static inline int cached_action_is_legal(const Env *env,
        const Action *policy) {
    if (!policy || policy->type >= ACTION_TYPE_COUNT ||
        policy->selection_count > MAX_SELECTION)
        return 0;
    uint8_t type = policy->type;
    uint8_t primary = (uint8_t)policy->primary;
    int has_primary = type >= ACTION_BUY_CARD &&
                      type <= ACTION_SWAP_HAND_RIGHT;
    if (!has_primary && primary != 0) return 0;
    uint8_t observed_primary = has_primary ? primary : 0;
    const LegalMasks *legal = &env->legal_masks;
    if (!legal->action_type[type] || observed_primary >= 64 ||
        !(legal->primary[type] & (UINT64_C(1) << observed_primary)))
        return 0;

    const ObservedSelection *selection =
        cached_selection(legal, type, observed_primary);
    if (!selection || !selection->valid)
        return policy->selection_count == 0;
    if (policy->selection_count < selection->minimum ||
        policy->selection_count > selection->maximum)
        return 0;

    uint64_t selected = 0;
    for (uint8_t i = 0; i < policy->selection_count; ++i) {
        uint8_t index = policy->selection[i];
        if (index >= MAX_HAND || index >= env->state.hand_count ||
            (i && policy->selection[i - 1] >= index))
            return 0;
        selected |= UINT64_C(1) << index;
    }
    return selected && !(selected & ~selection->allowed_hand) &&
           (selected & selection->required_hand) == selection->required_hand;
}

static inline void canonicalize_policy_action(
        const Env *env, Action *policy) {
    if (!policy || policy->type >= ACTION_TYPE_COUNT) return;
    int has_primary = policy->type >= ACTION_BUY_CARD &&
                      policy->type <= ACTION_SWAP_HAND_RIGHT;
    if (!has_primary) {
        policy->primary = 0;
    } else {
        uint64_t primary = env->legal_masks.primary[policy->type];
        if (!primary) return;
        if (policy->primary >= 64 || !(primary & (UINT64_C(1) << policy->primary)))
            policy->primary = (uint8_t)__builtin_ctzll(primary);
    }
    const ObservedSelection *selection = cached_selection(
        &env->legal_masks, policy->type, policy->primary);
    if (!selection || !selection->valid) {
        policy->selection_count = 0;
        return;
    }

    uint8_t target = policy->selection_count;
    if (target < selection->minimum) target = selection->minimum;
    if (target > selection->maximum) target = selection->maximum;
    uint8_t required_count = (uint8_t)__builtin_popcountll(selection->required_hand);
    if (target < required_count) target = required_count;
    uint64_t chosen = selection->required_hand;
    for (uint8_t i = 0; i < policy->selection_count &&
                        __builtin_popcountll(chosen) < target; ++i) {
        uint8_t index = policy->selection[i];
        if (index >= env->state.hand_count || index >= MAX_HAND) continue;
        uint64_t bit = UINT64_C(1) << index;
        if (selection->allowed_hand & bit) chosen |= bit;
    }
    uint64_t allowed = selection->allowed_hand & ~chosen;
    while (__builtin_popcountll(chosen) < target && allowed) {
        int index = __builtin_ctzll(allowed);
        chosen |= UINT64_C(1) << index;
        allowed &= allowed - 1;
    }
    policy->selection_count = 0;
    for (int index = 0; index < MAX_HAND &&
                        policy->selection_count < target; ++index) {
        if (chosen & (UINT64_C(1) << index))
            policy->selection[policy->selection_count++] = (uint8_t)index;
    }
}


void puf_init(Env *env, Dict *kwargs) {
    env->num_agents = 1;
    env->tag = 0;
    env->boundary_reached = 0;
    env->episode_reward = 0.0f;
    env->agents[0].policy = 0;
    default_config(&env->config);
    DictItem *shaped = dict_find(kwargs, "shaped_reward");
    DictItem *win_ante = dict_find(kwargs, "win_ante");
    DictItem *max_episode_steps = dict_find(kwargs, "max_episode_steps");
    DictItem *invalid_action_reward = dict_find(kwargs, "invalid_action_reward");
    DictItem *fast_rng = dict_find(kwargs, "fast_rng");
    env->config.shaped_reward = shaped ? (shaped->value != 0.0) : 1;
    /* Fast splitmix RNG by default (training). Set fast_rng=0 for the
       bit-compatible reference-game RNG used by the differential tools. */
    env->config.fast_rng = fast_rng ? (fast_rng->value != 0.0) : 1;
    env->invalid_action_reward = invalid_action_reward
        ? (float)invalid_action_reward->value : INVALID_ACTION_REWARD;
    env->max_episode_steps = 0;
    if (max_episode_steps && max_episode_steps->value > 0.0) {
        double value = max_episode_steps->value;
        if (value > UINT32_MAX) value = UINT32_MAX;
        env->max_episode_steps = (uint32_t)value;
    }
    if (win_ante) {
        int value = (int)win_ante->value;
        if (value < 1) value = 1;
        if (value > UINT8_MAX) value = UINT8_MAX;
        env->config.win_ante = (uint8_t)value;
    }
}

void puf_reset(Env *env) {
    uint64_t seed = ((uint64_t)rand_r(&env->rng) << 32) | rand_r(&env->rng);
    init(&env->state, &env->config, seed);
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    env->boundary_reached = 0;
    env->episode_reward = 0.0f;
    env->episode_steps = 0;
    puffer_observe(env);
}

static int episode_timed_out(const Env *env) {
    return env->max_episode_steps > 0 && env->episode_steps >= env->max_episode_steps;
}

static float episode_perf(const Env *env) {
    const State *state = &env->state;
    const double target_antes = state->config.win_ante ? state->config.win_ante : 1.0;
    const double completed_antes = state->ante > 0 ? (double)state->ante - 1.0 : 0.0;
    double perf = completed_antes / target_antes;

    double blind_fraction = 0.0;
    if ((state->phase == PHASE_SELECTING_HAND || state->phase == PHASE_GAME_OVER) &&
        state->blind_chips > 0.0) {
        blind_fraction = state->chips / state->blind_chips;
        if (!(blind_fraction >= 0.0)) blind_fraction = 0.0;
        if (blind_fraction > 1.0) blind_fraction = 1.0;
    }
    double blind_index = state->blind_on_deck < 3 ? (double)state->blind_on_deck : 3.0;
    perf += (blind_index + blind_fraction) / (3.0 * target_antes);
    if (!(perf >= 0.0)) perf = 0.0;
    if (perf > 1.0) perf = 1.0;
    return (float)perf;
}

static void truncate_episode(Env *env) {
    float transition_reward = env->agents[0].rewards[0] + TIMEOUT_REWARD;
    env->boundary_reached = 1;
    env->agents[0].terminals[0] = 1.0f;
    env->log.truncations += 1.0f;
    env->log.n += 1.0f;
    env->agents[0].rewards[0] = transition_reward;
    env->episode_reward += TIMEOUT_REWARD;
    env->log.score += env->episode_reward;
    env->log.perf += episode_perf(env);
    log_episode_end(env, 0);
    puf_reset(env);
    /* puf_reset clears the reward slot; restore the boundary transition's
       reward after resetting the next episode's state. */
    env->agents[0].rewards[0] = transition_reward;
    env->agents[0].terminals[0] = 1.0f;
}

void puf_step(Env *env) {
    Action policy = {0};
    policy.type = (uint8_t)env->agents[0].actions[0];
    policy.primary = (uint8_t)env->agents[0].actions[1];
    policy.selection_count = (uint8_t)env->agents[0].actions[2];
    if (policy.selection_count > MAX_SELECTION)
        policy.selection_count = MAX_SELECTION;
    for (int i = 0; i < MAX_SELECTION; ++i)
        policy.selection[i] = (uint8_t)env->agents[0].actions[3 + i];
    canonicalize_policy_action(env, &policy);
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    if (env->episode_steps < UINT32_MAX) env->episode_steps++;
    if (policy.type < ACTION_TYPE_COUNT)
        env->log.action_counts[policy.type] += 1.0f;
    StepResult result = {0};
    int error;
    if (!cached_action_is_legal(env, &policy)) {
        env->log.invalid_actions += 1.0f;
        env->agents[0].rewards[0] = env->invalid_action_reward;
        env->episode_reward += env->invalid_action_reward;
        if (episode_timed_out(env)) truncate_episode(env);
        return;
    }
    error = apply_step(&env->state, &policy, &env->legal_masks, &result);
    if (error != OK) {
        env->log.invalid_actions += 1.0f;
        env->agents[0].rewards[0] = env->invalid_action_reward;
        env->episode_reward += env->invalid_action_reward;
        if (episode_timed_out(env)) truncate_episode(env);
        else if (error != ERR_ACTION) {
            /* step validates before mutating the state. An invalid action
               therefore leaves the cached observation and legal mask current. */
            puffer_observe(env);
        }
        return;
    }
    /* Do not transform or replace result.reward here: it is Simulatro's
       shaped transition reward. */
    env->agents[0].rewards[0] = result.reward;
    env->episode_reward += result.reward;
    if (result.terminal) {
        env->boundary_reached = 1;
        env->agents[0].terminals[0] = 1.0f;
        /* Puffer's score is the episodic return used for sweep ranking.
           Keep wins separate as a sparse evaluation metric. */
        env->log.score += env->episode_reward;
        env->log.wins += result.won;
        env->log.perf += episode_perf(env);
        env->log.ante += result.ante;
        env->log.n += 1.0f;
        log_episode_end(env, result.won);
        puf_reset(env);
        env->agents[0].rewards[0] = result.reward;
        env->agents[0].terminals[0] = 1.0f;
    } else if (episode_timed_out(env)) {
        truncate_episode(env);
    } else {
        puffer_observe(env);
    }
}

void puf_render(Env *env) { (void)env; }
void puf_close(Env *env) { (void)env; }

void puf_log(Log *log, Dict *out) {
    dict_set(out, "score", log->score);
    dict_set(out, "perf", log->perf);
    dict_set(out, "wins", log->wins);
    dict_set(out, "ante", log->ante);
	dict_set(out, "invalid_actions", log->invalid_actions);
	dict_set(out, "truncations", log->truncations);
    dict_set(out, "end_dollars", log->end_dollars);
    dict_set(out, "end_dollars_won", log->end_dollars_won);
    dict_set(out, "end_jokers", log->end_jokers);
    dict_set(out, "end_consumables", log->end_consumables);
    dict_set(out, "end_hand_levels", log->end_hand_levels);
    dict_set(out, "end_deck", log->end_deck);
    dict_set(out, "ep_length", log->ep_length);

    for (int i = 0; i < ACTION_TYPE_COUNT; ++i)
        dict_set(out, action_log_names[i], log->action_counts[i]);
}

#endif
