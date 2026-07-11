#ifndef PUFFER_BALATRO_H
#define PUFFER_BALATRO_H

#include <stdlib.h>
#include <string.h>

#include "pufferenv.h"
#include <balatro_core.h>

#define OBS_SIZE BALATRO_OBSERVATION_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {BALATRO_MAX_LEGAL_ACTIONS}

typedef float obs_t;

typedef struct Log {
    float score;
    float wins;
    float ante;
    float invalid_actions;
    float n;
} Log;

typedef struct Env {
    Log log;
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;
    Agent agents[1];
    BalatroConfig config;
    BalatroState state;
    BalatroAction legal[BALATRO_MAX_LEGAL_ACTIONS];
    int legal_count;
} Env;

static void balatro_puffer_observe(Env *env) {
    BalatroObservation observation;
    balatro_observe(&env->state, &observation);
    float *out = (float *)env->agents[0].observations;
    memset(out, 0, OBS_SIZE * sizeof(float));
    memcpy(out, observation.values, observation.length * sizeof(float));

    env->legal_count = balatro_legal_actions(
        &env->state,
        env->legal,
        env->agents[0].action_mask,
        BALATRO_MAX_LEGAL_ACTIONS
    );
    if (env->legal_count < 0) env->legal_count = 0;
}

void puf_init(Env *env, Dict *kwargs) {
    (void)kwargs;
    env->num_agents = 1;
    env->tag = 0;
    env->boundary_reached = 0;
    env->agents[0].policy = 0;
    balatro_default_config(&env->config);
}

void puf_reset(Env *env) {
    uint64_t seed = ((uint64_t)rand_r(&env->rng) << 32) | rand_r(&env->rng);
    balatro_init(&env->state, &env->config, seed);
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    env->boundary_reached = 0;
    balatro_puffer_observe(env);
}

void puf_step(Env *env) {
    int slot = (int)env->agents[0].actions[0];
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    if (slot < 0 || slot >= env->legal_count) {
        env->log.invalid_actions += 1.0f;
        env->agents[0].rewards[0] = -1.0f;
        balatro_puffer_observe(env);
        return;
    }

    BalatroStepResult result;
    int error = balatro_step(&env->state, &env->legal[slot], &result);
    if (error != BALATRO_OK) {
        env->log.invalid_actions += 1.0f;
        env->agents[0].rewards[0] = -1.0f;
        balatro_puffer_observe(env);
        return;
    }
    env->agents[0].rewards[0] = result.reward;
    if (result.terminal) {
        env->boundary_reached = 1;
        env->agents[0].terminals[0] = 1.0f;
        env->log.score += result.won;
        env->log.wins += result.won;
        env->log.ante += result.ante;
        env->log.n += 1.0f;
        puf_reset(env);
        env->agents[0].terminals[0] = 1.0f;
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
}

#endif
