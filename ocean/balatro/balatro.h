#ifndef PUFFER_BALATRO_H
#define PUFFER_BALATRO_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "pufferenv.h"
#include <balatro_core.h>

/* PufferLib's tensor boundary is flat, while Simulatro's public observation
   is a fixed-layout structure. Encode each public byte as a finite float so
   the complete current observation remains available without duplicating its
   schema here. */
#define OBS_SIZE ((int)sizeof(BalatroObservation))
#define NUM_ATNS 9
#define ACT_SIZES {23, 64, 6, 64, 64, 64, 64, 64, 64}
#define ACTION_MASK_SIZE (23 + 64 + 6 + 64 + 64 + 64 + 64 + 64 + 64)

typedef float obs_t;

typedef struct Log {
    float score;
	float reward;
    float wins;
    float ante;
    float invalid_actions;
    float truncations;
    float n;
} Log;

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
    BalatroConfig config;
    BalatroState state;
    BalatroAction legal[BALATRO_MAX_LEGAL_ACTIONS];
    int legal_count;
} Env;

static void balatro_encode_observation(const BalatroObservation *observation, float *out) {
    const unsigned char *bytes = (const unsigned char *)observation;
    for (int i = 0; i < OBS_SIZE; ++i)
        out[i] = (float)bytes[i] * (1.0f / 255.0f);
}

static void balatro_mask_range(unsigned char *mask, int offset, int count) {
    for (int i = 0; i < count; ++i) mask[offset + i] = 1;
}

static void balatro_puffer_mask(const BalatroObservation *observation, unsigned char *mask) {
    memset(mask, 0, ACTION_MASK_SIZE);
    int type_offset = 0;
    int primary_offset = type_offset + 23;
    int count_offset = primary_offset + 64;
    int selection_offset = count_offset + 6;
    int reorder_offset = selection_offset + 5 * 64;
    int any_type = 0;
    int any_selection = 0;
    int any_reorder = 0;

    for (int type = 0; type < BALATRO_ACTION_TYPE_COUNT; ++type) {
        if (!observation->legal.action_type[type]) continue;
        mask[type_offset + type] = 1;
        any_type = 1;
        uint64_t primary = observation->legal.primary[type];
        for (int slot = 0; slot < 64; ++slot)
            if (primary & (UINT64_C(1) << slot)) mask[primary_offset + slot] = 1;
    }

    const BalatroObservedSelection *families[BALATRO_OBS_MAX_CONSUMABLES +
        BALATRO_OBS_MAX_SHOP_MAIN + BALATRO_OBS_MAX_PACK_CARDS + 2];
    int family_count = 0;
    families[family_count++] = &observation->legal.play;
    families[family_count++] = &observation->legal.discard;
    for (int i = 0; i < BALATRO_OBS_MAX_CONSUMABLES; ++i)
        families[family_count++] = &observation->legal.consumable[i];
    for (int i = 0; i < BALATRO_OBS_MAX_SHOP_MAIN; ++i)
        families[family_count++] = &observation->legal.shop[i];
    for (int i = 0; i < BALATRO_OBS_MAX_PACK_CARDS; ++i)
        families[family_count++] = &observation->legal.pack[i];
    for (int i = 0; i < family_count; ++i) {
        const BalatroObservedSelection *selection = families[i];
        if (!selection->valid) continue;
        any_selection = 1;
        for (int card = 0; card < 64; ++card)
            if (selection->allowed_hand & (UINT64_C(1) << card))
                for (int position = 0; position < 5; ++position)
                    mask[selection_offset + position * 64 + card] = 1;
    }
    for (int i = 0; i < BALATRO_OBS_MAX_HAND; ++i)
        if (observation->legal.hand_reorder_destination[i]) {
            any_reorder = 1;
            for (int destination = 0; destination < 64; ++destination)
                if (observation->legal.hand_reorder_destination[i] &
                    (UINT64_C(1) << destination))
                    mask[reorder_offset + destination] = 1;
        }
    for (int i = 0; i < BALATRO_OBS_MAX_JOKERS; ++i)
        if (observation->legal.joker_reorder_destination[i]) {
            any_reorder = 1;
            for (int destination = 0; destination < 64; ++destination)
                if (observation->legal.joker_reorder_destination[i] &
                    (UINT64_C(1) << destination))
                    mask[reorder_offset + destination] = 1;
        }

    /* Every categorical head must have at least one finite option. The
       structured validator remains authoritative for cross-head combinations. */
    if (!any_type) mask[type_offset] = 1;
    int any_primary = 0;
    for (int i = 0; i < 64; ++i) any_primary |= mask[primary_offset + i] != 0;
    if (!any_primary) mask[primary_offset] = 1;
    if (any_selection) balatro_mask_range(mask, count_offset, 6);
    else mask[count_offset] = 1;
    if (!any_selection)
        for (int position = 0; position < 5; ++position) mask[selection_offset + position * 64] = 1;
    if (!any_reorder) mask[reorder_offset] = 1;
}

static int balatro_puffer_observe(Env *env) {
    BalatroObservation observation;
    int error = balatro_observe(&env->state, &observation);
    float *out = (float *)env->agents[0].observations;
    memset(out, 0, OBS_SIZE * sizeof(float));
    if (error == BALATRO_OK) {
        balatro_encode_observation(&observation, out);
        if (env->agents[0].action_mask)
            balatro_puffer_mask(&observation, env->agents[0].action_mask);
    }
    env->legal_count = 0;
    return error;
}

void puf_init(Env *env, Dict *kwargs) {
    env->num_agents = 1;
    env->tag = 0;
    env->boundary_reached = 0;
    env->episode_reward = 0.0f;
    env->agents[0].policy = 0;
    balatro_default_config(&env->config);
    /* Training defaults to the progress potential, while the same wrapper
       can run the ABI's sparse win-only objective for an apples-to-apples
       baseline.  Optional env.* overrides are useful for curriculum runs. */
    DictItem *shaped = dict_find(kwargs, "shaped_reward");
    DictItem *win_ante = dict_find(kwargs, "win_ante");
    DictItem *max_episode_steps = dict_find(kwargs, "max_episode_steps");
    env->config.shaped_reward = shaped ? (shaped->value != 0.0) : 1;
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
    balatro_init(&env->state, &env->config, seed);
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    env->boundary_reached = 0;
    env->episode_reward = 0.0f;
    env->episode_steps = 0;
    balatro_puffer_observe(env);
}

static int balatro_episode_timed_out(const Env *env) {
    return env->max_episode_steps > 0 && env->episode_steps >= env->max_episode_steps;
}

static void balatro_truncate_episode(Env *env) {
    env->boundary_reached = 1;
    env->agents[0].terminals[0] = 1.0f;
    env->log.truncations += 1.0f;
    env->log.n += 1.0f;
    env->log.reward += env->episode_reward;
    puf_reset(env);
    env->agents[0].terminals[0] = 1.0f;
}

void puf_step(Env *env) {
    BalatroPolicyAction policy = {0};
    policy.type = (uint8_t)env->agents[0].actions[0];
    policy.primary = (uint16_t)env->agents[0].actions[1];
    policy.selection_count = (uint8_t)env->agents[0].actions[2];
    if (policy.selection_count > BALATRO_MAX_SELECTION)
        policy.selection_count = BALATRO_MAX_SELECTION;
    for (int i = 0; i < BALATRO_MAX_SELECTION; ++i)
        policy.selection[i] = (uint16_t)env->agents[0].actions[3 + i];
    policy.reorder_destination = (uint16_t)env->agents[0].actions[8];
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    if (env->episode_steps < UINT32_MAX) env->episode_steps++;
    BalatroStepResult result;
    BalatroObservation observation;
    int error = balatro_step_observe(&env->state, &policy, &result, &observation);
    if (error != BALATRO_OK) {
        env->log.invalid_actions += 1.0f;
        env->agents[0].rewards[0] = -1.0f;
        if (balatro_episode_timed_out(env)) balatro_truncate_episode(env);
        else balatro_puffer_observe(env);
        return;
    }
    env->agents[0].rewards[0] = result.reward;
    env->episode_reward += result.reward;
    if (result.terminal) {
        env->boundary_reached = 1;
        env->agents[0].terminals[0] = 1.0f;
        env->log.score += result.won;
        env->log.wins += result.won;
        env->log.ante += result.ante;
        env->log.n += 1.0f;
		env->log.reward += env->episode_reward;
        puf_reset(env);
        env->agents[0].terminals[0] = 1.0f;
    } else if (balatro_episode_timed_out(env)) {
        balatro_truncate_episode(env);
    } else {
        balatro_puffer_observe(env);
    }
}

void puf_render(Env *env) { (void)env; }
void puf_close(Env *env) { (void)env; }

void puf_log(Log *log, Dict *out) {
    dict_set(out, "score", log->score);
    dict_set(out, "wins", log->wins);
    dict_set(out, "ante", log->ante);
    dict_set(out, "invalid_actions", log->invalid_actions);
	dict_set(out, "truncations", log->truncations);
	dict_set(out, "reward", log->reward);
}

#endif
