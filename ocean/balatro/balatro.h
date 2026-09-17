#ifndef PUFFER_ENV_H
#define PUFFER_ENV_H

#define PUFFER_PACKED_OBS

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"
typedef unsigned char obs_t;
#include "pufferenv.h"
#include "balatro_core.h"

#define EXTRA_LOGS 0

#define POLICY_PRIMARY_COUNT 64
#define POLICY_PRIMARY_BYTES 8
#define POLICY_SELECTION_ENTRIES \
    (2 + MAX_CONSUMABLES + MAX_PACK_CARDS + MAX_SHOP_MAIN)
#define POLICY_SELECTION_BYTES 18
#define POLICY_PRIMARY_OFFSET ACTION_TYPE_COUNT
#define POLICY_SELECTION_OFFSET \
    (POLICY_PRIMARY_OFFSET + POLICY_PRIMARY_BYTES * ACTION_TYPE_COUNT)
#define POLICY_SELECTION_SIZE \
    (POLICY_SELECTION_OFFSET + POLICY_SELECTION_ENTRIES * POLICY_SELECTION_BYTES)
#define POLICY_ORDER_META_OFFSET POLICY_SELECTION_SIZE
#define POLICY_ORDER_META_BYTES 4
#define POLICY_ORDER_HAND_COUNT_OFFSET POLICY_ORDER_META_OFFSET
#define POLICY_ORDER_JOKER_COUNT_OFFSET (POLICY_ORDER_META_OFFSET + 1)
#define POLICY_ORDER_ENABLED_OFFSET (POLICY_ORDER_META_OFFSET + 2)
#define POLICY_MASK_SIZE \
    (POLICY_ORDER_META_OFFSET + POLICY_ORDER_META_BYTES)

#define AR_EMBED_DIM 32
#define DECODER_STATE 64
#define DECODER_CATEGORIES (ACTION_TYPE_COUNT + 6)
#define ROLE_OFFSET (DECODER_CATEGORIES * AR_EMBED_DIM)
#define AR_CONDITION_SIZE (ROLE_OFFSET + ACTION_STORAGE_SIZE * AR_EMBED_DIM)

#ifndef PUF_WARP_MASK
#define PUF_WARP_MASK 0xffffffff
#endif

#if defined(__CUDACC__)
#define POLICY_INLINE __host__ __device__ __forceinline__
#else
#define POLICY_INLINE static inline
#endif

POLICY_INLINE int policy_selection_entry(int type, int primary) {
    if (type == ACTION_PLAY_HAND) return primary == 0 ? 0 : -1;
    if (type == ACTION_DISCARD) return primary == 0 ? 1 : -1;
    if (type == ACTION_USE_CONSUMABLE)
        return primary >= 0 && primary < MAX_CONSUMABLES ? 2 + primary : -1;
    if (type == ACTION_PICK_PACK_CARD)
        return primary >= 0 && primary < MAX_PACK_CARDS
            ? 2 + MAX_CONSUMABLES + primary : -1;
    if (type == ACTION_BUY_AND_USE)
        return primary >= 0 && primary < MAX_SHOP_MAIN
            ? 2 + MAX_CONSUMABLES + MAX_PACK_CARDS + primary : -1;
    return -1;
}

#undef POLICY_INLINE

#define OBS_SIZE ((int)sizeof(Observation))
#define BASE_ACTION_HEADS 8
#define ORDER_HAND_OFFSET BASE_ACTION_HEADS
#define ORDER_JOKER_OFFSET \
    (ORDER_HAND_OFFSET + MAX_HAND)
#define ACTION_STORAGE_SIZE \
    (ORDER_JOKER_OFFSET + MAX_JOKERS)
#define POLICY_NUM_ATNS 10
#define NUM_ATNS ACTION_STORAGE_SIZE
#define ACT_SIZES {23, POLICY_PRIMARY_COUNT, 6, 64, 64, 64, 64, 64, 64, 32}
#define ACTION_MASK_SIZE POLICY_MASK_SIZE
#define INVALID_ACTION_REWARD (-0.002f)
#define TIMEOUT_REWARD (0.0f)

typedef struct ScoreAnim {
    bool pending;
    bool active;
    float t;
    float duration;
    Card played[MAX_SELECTION];
    int played_count;
    HandType hand_type;
    int level;
    int chips;
    int mult;
    double total;
    double chips_before;
    double blind_target;
    uint8_t scoring_mask;
    HandType allowed_hand_type;
    bool hand_not_allowed;
    bool blind_beaten;
    bool bust;
} ScoreAnim;

typedef struct ConsumableAnim {
    bool pending;
    bool active;
    bool hand_changed;
    bool jokers_changed;
    float t;
    float duration;
    Card used;
    Card before_hand[MAX_HAND];
    Card after_hand[MAX_HAND];
    Card before_jokers[MAX_JOKERS];
    Card after_jokers[MAX_JOKERS];
    uint16_t before_deck_ids[MAX_DECK];
    uint16_t selected_ids[MAX_SELECTION];
    uint8_t selected_count;
    uint8_t before_hand_count;
    uint8_t after_hand_count;
    uint8_t before_joker_count;
    uint8_t after_joker_count;
    uint16_t before_deck_count;
    bool pack_pick;
    uint8_t before_levels[HAND_COUNT];
    uint8_t after_levels[HAND_COUNT];
    int32_t before_dollars;
    int32_t after_dollars;
} ConsumableAnim;

typedef struct Client {
    int width;
    int height;
    bool auto_play;
    int auto_play_timer;
    int selected_count;
    uint8_t selected_indices[MAX_SELECTION];
    int forced_index;
    uint8_t last_phase;

    bool show_tooltip;
    char tooltip_title[64];
    char tooltip_type[64];
    char tooltip_desc[160];
    char tooltip_stats[64];
    Vector2 tooltip_pos;
    Texture2D puffer;

    bool paused;
    bool show_run_menu;
    bool show_controls_menu;
    int step_delay_frames;
    int step_cooldown;
    ScoreAnim score_anim;
    ConsumableAnim consumable_anim;
} Client;

typedef struct Log {
	float score;
	float perf;
	float wins;
	float ante;
	float max_hand_log10;
	float invalid_actions;
	float truncations;
	float n;
	float end_dollars;
	float end_hand_levels;
	float end_deck;
	float ep_length;
	float reorder_hand;
	float reorder_joker;
	float planets_bought;
	float planets_used;
	float celestial_packs_opened;
	float cleared_ante_8;
	float deaths_ante[16];
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
    float episode_peak_hand_log10;
    unsigned int rng;
    uint32_t episode_steps;
    uint32_t max_episode_steps;
    Agent agents[1];
    Config config;
    State state;
    LegalMasks legal_masks;
    float invalid_action_reward;
    uint8_t deck_config;
    uint8_t stake_config;
    uint8_t reorder_actions;
    Client *client;
} Env;

static inline void log_episode_end(Env *env, int won) {
    const State *state = &env->state;
    env->log.end_dollars += (float)state->dollars;
    float hand_levels = 0.0f;
    for (int i = 0; i < HAND_COUNT; ++i)
        hand_levels += (float)state->hand_levels[i];
    env->log.end_hand_levels += hand_levels;
    env->log.end_deck += (float)(state->deck_count + state->hand_count + state->discard_count);
    env->log.ep_length += (float)env->episode_steps;

    if (state->cleared_ante_8) {
        env->log.cleared_ante_8 += 1.0f;
    }
    int ante_idx = (int)state->ante - 1;
    if (ante_idx < 0) ante_idx = 0;
    if (ante_idx > 15) ante_idx = 15;
    env->log.deaths_ante[ante_idx] += 1.0f;
}

static inline void store_u64(unsigned char *out, uint64_t value) {
    memcpy(out, &value, sizeof(value));
}

static inline void store_selection(
        unsigned char *mask, int entry,
        const SelectionContract *selection) {
    assert(selection && entry >= 0);
    if (!selection->valid) return;
    unsigned char *out = mask + POLICY_SELECTION_OFFSET +
        entry * POLICY_SELECTION_BYTES;
    out[0] = selection->minimum;
    out[1] = selection->maximum;
    store_u64(out + 2, selection->allowed_hand);
    store_u64(out + 10, selection->required_hand);
}

static int puffer_observe(Env *env) {
    obs_t *out = (obs_t *)env->agents[0].observations;
    int error = observe(&env->state, (Observation *)out, &env->legal_masks);
    if (error != OK) {
        memset(out, 0, OBS_SIZE * sizeof(*out));
        return error;
    }
    if (env->agents[0].action_mask) {
        memset(env->agents[0].action_mask, 0, ACTION_MASK_SIZE);
        for (int type = 0; type < ACTION_TYPE_COUNT; ++type) {
            if (type >= ACTION_SWAP_JOKERS_LEFT && type <= ACTION_SORT_HAND_SUIT)
                continue;
            if (!env->legal_masks.primary[type]) continue;
            uint64_t primaries = env->legal_masks.primary[type];
            env->agents[0].action_mask[type] = 1;
            store_u64(env->agents[0].action_mask + POLICY_PRIMARY_OFFSET +
                type * POLICY_PRIMARY_BYTES, primaries);
        }
        int choices = 0;
        for (int type = 0; type < ACTION_TYPE_COUNT; ++type) choices += env->agents[0].action_mask[type];
        if (!choices) {
            fprintf(stderr, "Empty engine mask: phase=%u hand=%u deck=%u discard=%u jokers=%u consumables=%u terminal=%u\n",
                env->state.phase, env->state.hand_count, env->state.deck_count, env->state.discard_count,
                env->state.joker_count, env->state.consumable_count, env->state.terminal);
        }
        assert(choices);
        store_selection(env->agents[0].action_mask,
            policy_selection_entry(ACTION_PLAY_HAND, 0),
            &env->legal_masks.play);
        store_selection(env->agents[0].action_mask,
            policy_selection_entry(ACTION_DISCARD, 0),
            &env->legal_masks.discard);
        for (int i = 0; i < MAX_CONSUMABLES; ++i)
            store_selection(env->agents[0].action_mask,
                policy_selection_entry(ACTION_USE_CONSUMABLE, i),
                &env->legal_masks.consumable[i]);
        for (int i = 0; i < MAX_SHOP_MAIN; ++i)
            store_selection(env->agents[0].action_mask,
                policy_selection_entry(ACTION_BUY_AND_USE, i),
                &env->legal_masks.shop[i]);
        for (int i = 0; i < MAX_PACK_CARDS; ++i)
            store_selection(env->agents[0].action_mask,
                policy_selection_entry(ACTION_PICK_PACK_CARD, i),
                &env->legal_masks.pack[i]);
        env->agents[0].action_mask[POLICY_ORDER_HAND_COUNT_OFFSET] =
            env->state.hand_count;
        env->agents[0].action_mask[POLICY_ORDER_JOKER_COUNT_OFFSET] =
            env->state.joker_count;
        env->agents[0].action_mask[POLICY_ORDER_ENABLED_OFFSET] =
            env->reorder_actions != 0 && env->state.phase != PHASE_ROUND_EVAL &&
            env->state.phase != PHASE_GAME_OVER;
    }
    return OK;
}

void puf_init(Env *env, Dict *kwargs) {
    env->num_agents = 1;
    env->tag = 0;
    env->boundary_reached = 0;
    env->episode_reward = 0.0f;
    env->episode_peak_hand_log10 = 0.0f;
    env->agents[0].policy = 0;
    default_config(&env->config);
    env->deck_config = 0;
    env->stake_config = 1;
    DictItem *seed = dict_find(kwargs, "seed");
    if (seed) {
        assert(seed->value >= 0.0 && seed->value <= UINT_MAX);
        assert(seed->value == (double)(unsigned int)seed->value);
        env->rng += (unsigned int)seed->value;
    }
    DictItem *deck_item = dict_find(kwargs, "deck");
    if (deck_item) {
        if (deck_item->str && (strcmp(deck_item->str, "all") == 0 || strcmp(deck_item->str, "random") == 0 || strcmp(deck_item->str, "any") == 0)) {
            env->deck_config = 255;
        } else if (deck_item->str && strcmp(deck_item->str, "red") == 0) {
            env->deck_config = CENTER_B_RED;
        } else if (deck_item->str && strcmp(deck_item->str, "blue") == 0) {
            env->deck_config = CENTER_B_BLUE;
        } else if (deck_item->str && strcmp(deck_item->str, "yellow") == 0) {
            env->deck_config = CENTER_B_YELLOW;
        } else if (deck_item->str && strcmp(deck_item->str, "green") == 0) {
            env->deck_config = CENTER_B_GREEN;
        } else if (deck_item->str && strcmp(deck_item->str, "black") == 0) {
            env->deck_config = CENTER_B_BLACK;
        } else if (deck_item->str && strcmp(deck_item->str, "magic") == 0) {
            env->deck_config = CENTER_B_MAGIC;
        } else if (deck_item->str && strcmp(deck_item->str, "nebula") == 0) {
            env->deck_config = CENTER_B_NEBULA;
        } else if (deck_item->str && strcmp(deck_item->str, "ghost") == 0) {
            env->deck_config = CENTER_B_GHOST;
        } else if (deck_item->str && strcmp(deck_item->str, "abandoned") == 0) {
            env->deck_config = CENTER_B_ABANDONED;
        } else if (deck_item->str && strcmp(deck_item->str, "checkered") == 0) {
            env->deck_config = CENTER_B_CHECKERED;
        } else if (deck_item->str && strcmp(deck_item->str, "zodiac") == 0) {
            env->deck_config = CENTER_B_ZODIAC;
        } else if (deck_item->str && strcmp(deck_item->str, "painted") == 0) {
            env->deck_config = CENTER_B_PAINTED;
        } else if (deck_item->str && strcmp(deck_item->str, "anaglyph") == 0) {
            env->deck_config = CENTER_B_ANAGLYPH;
        } else if (deck_item->str && strcmp(deck_item->str, "plasma") == 0) {
            env->deck_config = CENTER_B_PLASMA;
        } else if (deck_item->str && strcmp(deck_item->str, "erratic") == 0) {
            env->deck_config = CENTER_B_ERRATIC;
        } else {
            int val = (int)deck_item->value;
            if (val == -1 || val == 255) env->deck_config = 255;
            else if (val >= 0 && val <= 16) env->deck_config = (uint8_t)val;
        }
    }
    DictItem *stake_item = dict_find(kwargs, "stake");
    if (stake_item) {
        if (stake_item->str && (strcmp(stake_item->str, "all") == 0 || strcmp(stake_item->str, "random") == 0 || strcmp(stake_item->str, "any") == 0)) {
            env->stake_config = 255;
        } else if (stake_item->str && strcmp(stake_item->str, "white") == 0) {
            env->stake_config = 1;
        } else if (stake_item->str && strcmp(stake_item->str, "red") == 0) {
            env->stake_config = 2;
        } else if (stake_item->str && strcmp(stake_item->str, "green") == 0) {
            env->stake_config = 3;
        } else if (stake_item->str && strcmp(stake_item->str, "black") == 0) {
            env->stake_config = 4;
        } else if (stake_item->str && strcmp(stake_item->str, "blue") == 0) {
            env->stake_config = 5;
        } else if (stake_item->str && strcmp(stake_item->str, "purple") == 0) {
            env->stake_config = 6;
        } else if (stake_item->str && strcmp(stake_item->str, "orange") == 0) {
            env->stake_config = 7;
        } else if (stake_item->str && strcmp(stake_item->str, "gold") == 0) {
            env->stake_config = 8;
        } else {
            int val = (int)stake_item->value;
            if (val == -1 || val == 255) env->stake_config = 255;
            else if (val >= 1 && val <= 8) env->stake_config = (uint8_t)val;
        }
    }
    DictItem *shaped = dict_find(kwargs, "shaped_reward");
    DictItem *win_ante = dict_find(kwargs, "win_ante");
    DictItem *max_episode_steps = dict_find(kwargs, "max_episode_steps");
    DictItem *invalid_action_reward = dict_find(kwargs, "invalid_action_reward");
    DictItem *fast_rng = dict_find(kwargs, "fast_rng");
    DictItem *reorder_actions = dict_find(kwargs, "reorder_actions");
#define FLOAT_REWARD_OPT(name, field, cmp) do { \
    DictItem *item = dict_find(kwargs, name); \
    if (item) { assert(item->value cmp 0.0); env->config.field = (float)item->value; } \
} while (0)
    FLOAT_REWARD_OPT("progress_reward", progress_reward, >=);
	FLOAT_REWARD_OPT("contact_reward", contact_reward, >=);
    FLOAT_REWARD_OPT("blind_bonus", blind_bonus, >=);
    FLOAT_REWARD_OPT("ante_bonus", ante_bonus, >=);
    FLOAT_REWARD_OPT("ante_escalation", ante_escalation, >=);
	FLOAT_REWARD_OPT("efficiency_hand", efficiency_hand, >=);
	FLOAT_REWARD_OPT("efficiency_discard", efficiency_discard, >=);
	FLOAT_REWARD_OPT("wealth_weight", wealth_weight, >=);
#undef FLOAT_REWARD_OPT
    assert(env->config.blind_bonus <= 0.5f);
    assert(env->config.ante_bonus <= 0.8f);
    assert(env->config.ante_escalation >= 1.0f);
	assert(env->config.ante_escalation <= 2.0f);
    assert(env->config.progress_reward <= 0.2f);
	assert(env->config.contact_reward <= 0.2f);
	assert(env->config.efficiency_hand <= 0.5f);
	assert(env->config.efficiency_discard <= 0.5f);
	assert(env->config.wealth_weight <= 1.0f);
    env->config.shaped_reward = shaped ? (shaped->value != 0.0) : 1;
    env->reorder_actions = reorder_actions && reorder_actions->value != 0.0;
    env->config.fast_rng = fast_rng ? (fast_rng->value != 0.0) : 1;
	assert(env->invalid_action_reward >= -1.0f && env->invalid_action_reward <= 1.0f);
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
    static const uint8_t ALL_DECKS[15] = {
        CENTER_B_RED, CENTER_B_BLUE, CENTER_B_YELLOW, CENTER_B_GREEN,
        CENTER_B_BLACK, CENTER_B_MAGIC, CENTER_B_NEBULA, CENTER_B_GHOST,
        CENTER_B_ABANDONED, CENTER_B_CHECKERED, CENTER_B_ZODIAC, CENTER_B_PAINTED,
        CENTER_B_ANAGLYPH, CENTER_B_PLASMA, CENTER_B_ERRATIC
    };
    if (env->deck_config == 255) {
        env->config.deck = ALL_DECKS[rand_r(&env->rng) % 15];
    } else {
        env->config.deck = env->deck_config;
    }
    if (env->stake_config == 255) {
        env->config.stake = 1 + (rand_r(&env->rng) % 8);
    } else {
        env->config.stake = env->stake_config;
    }
    uint64_t seed = ((uint64_t)rand_r(&env->rng) << 32) | rand_r(&env->rng);
    init(&env->state, &env->config, seed);
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    env->boundary_reached = 0;
    env->episode_reward = 0.0f;
    env->episode_peak_hand_log10 = 0.0f;
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
    float transition_reward = TIMEOUT_REWARD;
    env->boundary_reached = 1;
    env->agents[0].terminals[0] = 1.0f;
    env->log.truncations += 1.0f;
    env->log.n += 1.0f;
    env->agents[0].rewards[0] = transition_reward;
    env->episode_reward += TIMEOUT_REWARD;
    env->log.score += env->episode_reward;
    env->log.perf += episode_perf(env);
	env->log.ante += env->state.ante;
    log_episode_end(env, 0);
    env->log.max_hand_log10 += env->episode_peak_hand_log10;
    puf_reset(env);
    /* puf_reset clears the reward slot; restore the boundary transition's
       reward after resetting the next episode's state. */
    env->agents[0].rewards[0] = transition_reward;
    env->agents[0].terminals[0] = 1.0f;
}

static void score_anim_capture(Client *client, const State *state, const Action *action);
static void score_anim_start(Client *client, const State *state);
static void capture_consumable(Client *client, const State *state, const Action *action);
static void start_consumable(Client *client, const State *state);

void puf_step(Env *env) {
    if (env->client && env->client->paused) {
        env->agents[0].rewards[0] = 0.0f;
        env->agents[0].terminals[0] = 0.0f;
        return;
    }
    if (env->client && (env->client->score_anim.active || env->client->consumable_anim.active)) {
        env->agents[0].rewards[0] = 0.0f;
        env->agents[0].terminals[0] = 0.0f;
        return;
    }
    if (env->client && env->client->step_cooldown > 0) {
        env->client->step_cooldown--;
        env->agents[0].rewards[0] = 0.0f;
        env->agents[0].terminals[0] = 0.0f;
        return;
    }

    Action policy = {0};
    policy.type = (uint8_t)env->agents[0].actions[0];
    policy.primary = (uint8_t)env->agents[0].actions[1];
    policy.selection_count = (uint8_t)env->agents[0].actions[2];
    assert(policy.selection_count <= MAX_SELECTION);
    for (int i = 0; i < MAX_SELECTION; ++i)
        policy.selection[i] = (uint8_t)env->agents[0].actions[3 + i];
    const LegalMasks *legal = &env->legal_masks;
    int action_is_legal = 1;
    if (policy.type >= ACTION_TYPE_COUNT || policy.selection_count > MAX_SELECTION ||
        !legal->primary[policy.type]) {
        action_is_legal = 0;
    } else {
        int has_primary = (policy.type >= ACTION_BUY_CARD &&
                           policy.type <= ACTION_SWAP_HAND_RIGHT) ||
                          policy.type == ACTION_BUY_AND_USE;
        if (!has_primary) {
            policy.primary = 0;
        } else {
            uint64_t primary = legal->primary[policy.type];
            if (!primary)
                action_is_legal = 0;
            else if (policy.primary >= 64 || !(primary & (UINT64_C(1) << policy.primary)))
                policy.primary = (uint8_t)__builtin_ctzll(primary);
        }
        if (action_is_legal) {
            const SelectionContract *selection =
                cached_selection(legal, policy.type, policy.primary);
            if (!selection || !selection->valid) {
                policy.selection_count = 0;
            } else {
                uint8_t target = policy.selection_count;
                if (target < selection->minimum) target = selection->minimum;
                if (target > selection->maximum) target = selection->maximum;
                uint8_t required_count = (uint8_t)__builtin_popcountll(selection->required_hand);
                if (target < required_count) target = required_count;
                uint64_t chosen = selection->required_hand;
                for (uint8_t i = 0; i < policy.selection_count &&
                                    __builtin_popcountll(chosen) < target; ++i) {
                    uint8_t index = policy.selection[i];
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
                policy.selection_count = 0;
                for (int index = 0; index < MAX_HAND &&
                                    policy.selection_count < target; ++index) {
                    if (chosen & (UINT64_C(1) << index))
                        policy.selection[policy.selection_count++] = (uint8_t)index;
                }
                if (policy.selection_count < selection->minimum ||
                    policy.selection_count > selection->maximum ||
                    (chosen & ~selection->allowed_hand) != 0)
                    action_is_legal = 0;
            }
        }
    }
    if (action_is_legal && env->reorder_actions &&
        env->state.phase != PHASE_ROUND_EVAL &&
        env->state.phase != PHASE_GAME_OVER &&
        (policy.type < ACTION_SWAP_JOKERS_LEFT ||
         policy.type > ACTION_SORT_HAND_SUIT)) {
        int hand_count = env->state.hand_count;
        int joker_count = env->state.joker_count;
        int hand_order[MAX_HAND];
        int joker_order[MAX_JOKERS];
        int hand_map[MAX_HAND];
        int joker_map[MAX_JOKERS];
        bool hand_seen[MAX_HAND] = {0};
        bool joker_seen[MAX_JOKERS] = {0};
        bool hand_identity = 1;
        bool joker_identity = 1;
        for (int i = 0; i < hand_count; ++i) {
            int old = (int)env->agents[0].actions[
                ORDER_HAND_OFFSET + i];
            assert(old >= 0 && old < hand_count);
            assert(!hand_seen[old]);
            hand_seen[old] = 1;
            hand_order[i] = old;
            hand_map[old] = i;
            if (old != i) hand_identity = 0;
        }
        for (int i = 0; i < joker_count; ++i) {
            int old = (int)env->agents[0].actions[
                ORDER_JOKER_OFFSET + i];
            assert(old >= 0 && old < joker_count);
            assert(!joker_seen[old]);
            joker_seen[old] = 1;
            joker_order[i] = old;
            joker_map[old] = i;
            if (old != i) joker_identity = 0;
        }
        if (!hand_identity) {
            env->log.reorder_hand += 1.0f;
            Card hand_copy[MAX_HAND];
            for (int i = 0; i < hand_count; ++i)
                hand_copy[i] = env->state.hand[hand_order[i]];
            memcpy(env->state.hand, hand_copy, sizeof(Card) * hand_count);
            for (int i = 0; i < policy.selection_count; ++i)
                policy.selection[i] = (uint8_t)hand_map[policy.selection[i]];
            for (int i = 1; i < policy.selection_count; ++i) {
                uint8_t value = policy.selection[i];
                int j = i;
                while (j > 0 && policy.selection[j - 1] > value) {
                    policy.selection[j] = policy.selection[j - 1];
                    --j;
                }
                policy.selection[j] = value;
            }
        }
        if (!joker_identity) {
            env->log.reorder_joker += 1.0f;
            Card joker_copy[MAX_JOKERS];
            for (int i = 0; i < joker_count; ++i)
                joker_copy[i] = env->state.jokers[joker_order[i]];
            memcpy(env->state.jokers, joker_copy, sizeof(Card) * joker_count);
        }
        if (policy.type == ACTION_SELL_JOKER && !joker_identity)
            policy.primary = (uint8_t)joker_map[policy.primary];
    }
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    if (env->episode_steps < UINT32_MAX) env->episode_steps++;
    if (policy.type < ACTION_TYPE_COUNT)
        env->log.action_counts[policy.type] += 1.0f;
    if (action_is_legal &&
        (policy.type == ACTION_BUY_CARD || policy.type == ACTION_BUY_AND_USE)) {
        uint16_t center = env->state.shop_main[policy.primary].center_id;
        if (planet_hand(center) < HAND_COUNT)
            env->log.planets_bought += 1.0f;
    }
    if (action_is_legal && policy.type == ACTION_USE_CONSUMABLE) {
        uint16_t center = env->state.consumables[policy.primary].center_id;
        if (planet_hand(center) < HAND_COUNT)
            env->log.planets_used += 1.0f;
    }
    if (action_is_legal && policy.type == ACTION_BUY_AND_USE) {
        uint16_t center = env->state.shop_main[policy.primary].center_id;
        if (planet_hand(center) < HAND_COUNT)
            env->log.planets_used += 1.0f;
    }
    if (action_is_legal && policy.type == ACTION_PICK_PACK_CARD) {
        uint16_t center = env->state.pack_cards[policy.primary].center_id;
        if (planet_hand(center) < HAND_COUNT)
            env->log.planets_used += 1.0f;
    }
    if (action_is_legal && policy.type == ACTION_OPEN_BOOSTER) {
        uint16_t center = env->state.shop_boosters[policy.primary].center_id;
        if (center >= CENTER_P_CELESTIAL_JUMBO_1 && center <= CENTER_P_CELESTIAL_NORMAL_4)
            env->log.celestial_packs_opened += 1.0f;
    }
    if (action_is_legal && policy.type == ACTION_PLAY_HAND && env->client) {
        score_anim_capture(env->client, &env->state, &policy);
    }
    if (action_is_legal && env->client &&
        (policy.type == ACTION_USE_CONSUMABLE || policy.type == ACTION_BUY_AND_USE ||
         policy.type == ACTION_PICK_PACK_CARD))
        capture_consumable(env->client, &env->state, &policy);
    StepResult result = {0};
    int error = action_is_legal
        ? apply_step(&env->state, &policy, &env->legal_masks, &result)
        : ERR_ACTION;
    if (error == OK && policy.type == ACTION_PLAY_HAND) {
        double hand_score = env->state.last_hand_score;
        if (isfinite(hand_score) && hand_score > 0.0) {
            double peak = log10(hand_score);
            if (peak > env->episode_peak_hand_log10)
                env->episode_peak_hand_log10 = (float)peak;
        }
    }
    if (error == OK && policy.type == ACTION_PLAY_HAND && env->client) {
        score_anim_start(env->client, &env->state);
    }
    if (error == OK && env->client && env->client->consumable_anim.pending)
        start_consumable(env->client, &env->state);
    if (error == OK && env->client) {
        env->client->step_cooldown = env->client->step_delay_frames;
    }
    if (error != OK) {
        env->log.invalid_actions += 1.0f;
        env->agents[0].rewards[0] = env->invalid_action_reward;
        env->episode_reward += env->invalid_action_reward;
        if (episode_timed_out(env))
            truncate_episode(env);
        else if (error != ERR_ACTION) {
            puffer_observe(env);
        }
        return;
    }
    env->agents[0].rewards[0] = result.reward;
    env->episode_reward += result.reward;
    if (result.terminal) {
        env->boundary_reached = 1;
        env->agents[0].terminals[0] = 1.0f;
        env->log.score += env->episode_reward;
        env->log.wins += result.won;
        env->log.perf += episode_perf(env);
        env->log.ante += result.ante;
        env->log.n += 1.0f;
        log_episode_end(env, result.won);
        env->log.max_hand_log10 += env->episode_peak_hand_log10;
        puf_reset(env);
        env->agents[0].rewards[0] = result.reward;
        env->agents[0].terminals[0] = 1.0f;
    } else if (episode_timed_out(env)) {
        truncate_episode(env);
    } else {
        puffer_observe(env);
    }
}

#include "balatro_render.h"

void puf_render(Env *env) {
    balatro_render(env);
}

void puf_close(Env *env) {
    if (env && env->client) {
        close_client(env->client);
        env->client = NULL;
    }
}

void puf_log(Log *log, Dict *out) {
    dict_set(out, "score", log->score);
    dict_set(out, "perf", log->perf);
    dict_set(out, "cleared_ante_8", log->cleared_ante_8);
    dict_set(out, "ante", log->ante);

#if EXTRA_LOGS
    dict_set(out, "max_hand_log10", log->max_hand_log10);
	dict_set(out, "invalid_actions", log->invalid_actions);
	dict_set(out, "truncations", log->truncations);
    dict_set(out, "end_dollars", log->end_dollars);
    dict_set(out, "end_hand_levels", log->end_hand_levels);
    dict_set(out, "end_deck", log->end_deck);
    dict_set(out, "ep_length", log->ep_length);

    static const char *const deaths_ante_names[16] = {
        "deaths/ante_1",
        "deaths/ante_2",
        "deaths/ante_3",
        "deaths/ante_4",
        "deaths/ante_5",
        "deaths/ante_6",
        "deaths/ante_7",
        "deaths/ante_8",
        "deaths/ante_9",
        "deaths/ante_10",
        "deaths/ante_11",
        "deaths/ante_12",
        "deaths/ante_13",
        "deaths/ante_14",
        "deaths/ante_15",
        "deaths/ante_16_plus",
    };
    for (int i = 0; i < 16; ++i)
        dict_set(out, deaths_ante_names[i], log->deaths_ante[i]);

    dict_set(out, "reorders/hand", log->reorder_hand);
    dict_set(out, "reorders/joker", log->reorder_joker);
    dict_set(out, "planets/bought", log->planets_bought);
    dict_set(out, "planets/used", log->planets_used);
    dict_set(out, "planets/celestial_packs_opened", log->celestial_packs_opened);

    for (int i = 0; i < ACTION_TYPE_COUNT; ++i)
        dict_set(out, action_log_names[i], log->action_counts[i]);
#endif
}

#endif
