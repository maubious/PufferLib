// Best-trajectory state curriculum.
//
// Ported from upstream 5.0-simplecl src/curriculum.cu, state-set portion only:
// the priority-replay buffers (prio_weights / CDF / multinomial sampling) are
// deliberately excluded. This keeps the learner minibatch selection untouched
// (contiguous slices) and limits the port to episode-state capture/replay.
//
// Mechanism (identical to upstream):
//   - A host-side StateBuffer stores a pool of "best" mid-episode states. Each
//     pool slot is a trajectory (up to state_trajectory_max_len checkpoints,
//     recorded every state_checkpoint_interval steps) that produced the
//     highest episodic return seen so far.
//   - FRESH envs roll full episodes from a seeded start; their checkpoints
//     compete to replace the best trajectory for their slot.
//   - CL envs resume from a random checkpoint inside a best trajectory
//     (prefix return/step carried), continue to terminal, and splice their
//     tail back in. cl_frac decays toward 0 when anneal_cl is set.
//   - VANILLA envs (the remainder) train exactly as before and are the only
//     envs that report environment logs (log_env_limit).
//
// Gated by PUFFER_CURRICULUM: envs define it when they expose a by-value
// `State state` member and a `puffer_state_refresh(Env*)` that rebuilds the
// observation / action mask / legal masks from the state. Included from
// pufferl.cu after Env, VecEnv, PuffeRL, cpu_upload, and rollout_time_view are
// defined, so this file can reuse OBS_SIZE / obs_t / EnvBuf directly.

#ifdef PUFFER_CURRICULUM

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef State PufferState;

enum CurriculumRole {
    CURRICULUM_ROLE_VANILLA = 0,
    CURRICULUM_ROLE_FRESH = 1,
    CURRICULUM_ROLE_CL = 2,
};

struct StateBuffer {
    PufferState* best_states;
    PufferState* active_states;
    float* best_state_return;
    float* active_state_return;
    int* best_state_step;
    int* active_state_step;
    int* best_valid;
    int* best_len;
    int* best_episode_len;
    float* best_return;
    int* env_role;
    int* env_active_slot;
    int* env_best_slot;
    int* env_sample_offset;
    int* env_history_count;
    float* env_episode_return;
    float* env_prefix_return;
    int* env_episode_len;
    int* env_prefix_step;
    long* env_last_observed_step;
    long long* best_generation;
    long long* env_best_generation;
    int num_envs;
    int agents_per_env;
    int max_active_envs;
    int num_start_states;
    int trajectory_max_len;
    int checkpoint_interval;
    int num_vanilla_envs;
    int num_fresh_envs;
    int num_cl_envs;
    int seeded;
    int host_lock;
    long capture_ticket;
    // Snapshot of pufferl->global_step taken at rollout_begin. Workers read
    // this instead of global_step (the main thread increments global_step
    // mid-horizon in the async path, which would otherwise tear serials).
    long rollout_step_base;
};

static inline int curriculum_clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

void close_state_buffer(StateBuffer* buf);

void register_state_buffer(StateBuffer* buf,
        int num_envs, int agents_per_env, int max_active_envs,
        int num_start_states, int trajectory_max_len, int checkpoint_interval) {
    if (num_start_states < 1) {
        num_start_states = 1;
    }
    if (trajectory_max_len < 1) {
        trajectory_max_len = 1;
    }
    max_active_envs = curriculum_clamp_int(max_active_envs, 1, num_envs);

    buf->num_envs = num_envs;
    buf->agents_per_env = agents_per_env;
    buf->max_active_envs = max_active_envs;
    buf->num_start_states = num_start_states;
    buf->trajectory_max_len = trajectory_max_len;
    buf->checkpoint_interval = checkpoint_interval;
    buf->num_vanilla_envs = num_envs;
    buf->num_fresh_envs = 0;
    buf->num_cl_envs = 0;
    buf->seeded = 0;
    buf->host_lock = 0;
    buf->capture_ticket = 0;
}

int init_state_buffer(StateBuffer* buf) {
    size_t best_rows = (size_t)buf->num_start_states;
    size_t traj_len = (size_t)buf->trajectory_max_len;
    size_t active_rows = (size_t)buf->max_active_envs;
    size_t best_entries = best_rows * traj_len;
    size_t active_entries = active_rows * traj_len;
    size_t envs = (size_t)buf->num_envs;

    buf->best_states = (PufferState*)calloc(best_entries, sizeof(PufferState));
    buf->active_states = (PufferState*)calloc(active_entries, sizeof(PufferState));
    buf->best_state_return = (float*)calloc(best_entries, sizeof(float));
    buf->active_state_return = (float*)calloc(active_entries, sizeof(float));
    buf->best_state_step = (int*)calloc(best_entries, sizeof(int));
    buf->active_state_step = (int*)calloc(active_entries, sizeof(int));
    buf->best_valid = (int*)calloc(best_rows, sizeof(int));
    buf->best_len = (int*)calloc(best_rows, sizeof(int));
    buf->best_episode_len = (int*)calloc(best_rows, sizeof(int));
    buf->best_return = (float*)calloc(best_rows, sizeof(float));
    buf->env_role = (int*)calloc(envs, sizeof(int));
    buf->env_active_slot = (int*)calloc(envs, sizeof(int));
    buf->env_best_slot = (int*)calloc(envs, sizeof(int));
    buf->env_sample_offset = (int*)calloc(envs, sizeof(int));
    buf->env_history_count = (int*)calloc(envs, sizeof(int));
    buf->env_episode_return = (float*)calloc(envs, sizeof(float));
    buf->env_prefix_return = (float*)calloc(envs, sizeof(float));
    buf->env_episode_len = (int*)calloc(envs, sizeof(int));
    buf->env_prefix_step = (int*)calloc(envs, sizeof(int));
    buf->env_last_observed_step = (long*)calloc(envs, sizeof(long));
    buf->best_generation = (long long*)calloc(best_rows, sizeof(long long));
    buf->env_best_generation = (long long*)calloc(envs, sizeof(long long));

    if (buf->best_states == NULL || buf->active_states == NULL
            || buf->best_state_return == NULL || buf->active_state_return == NULL
            || buf->best_state_step == NULL || buf->active_state_step == NULL
            || buf->best_valid == NULL || buf->best_len == NULL
            || buf->best_episode_len == NULL || buf->best_return == NULL
            || buf->env_role == NULL || buf->env_active_slot == NULL
            || buf->env_best_slot == NULL || buf->env_sample_offset == NULL
            || buf->env_history_count == NULL || buf->env_episode_return == NULL
            || buf->env_prefix_return == NULL || buf->env_episode_len == NULL
            || buf->env_prefix_step == NULL || buf->env_last_observed_step == NULL
            || buf->best_generation == NULL || buf->env_best_generation == NULL) {
        fprintf(stderr,
            "Failed to allocate curriculum state buffer: starts=%d len=%d "
            "active=%d state_size=%d\n",
            buf->num_start_states, buf->trajectory_max_len,
            buf->max_active_envs, (int)sizeof(PufferState));
        close_state_buffer(buf);
        return 0;
    }

    for (int i = 0; i < buf->num_start_states; i++) {
        buf->best_return[i] = -3.402823466e+38F;
    }
    for (int i = 0; i < buf->num_envs; i++) {
        buf->env_role[i] = CURRICULUM_ROLE_VANILLA;
        buf->env_active_slot[i] = -1;
        buf->env_best_slot[i] = -1;
        buf->env_sample_offset[i] = 0;
        buf->env_history_count[i] = 0;
        buf->env_episode_return[i] = 0.0f;
        buf->env_prefix_return[i] = 0.0f;
        buf->env_episode_len[i] = 0;
        buf->env_prefix_step[i] = 0;
        buf->env_last_observed_step[i] = -1;
        buf->env_best_generation[i] = -1;
    }

    return 1;
}

void close_state_buffer(StateBuffer* buf) {
    free(buf->best_states);
    free(buf->active_states);
    free(buf->best_state_return);
    free(buf->active_state_return);
    free(buf->best_state_step);
    free(buf->active_state_step);
    free(buf->best_valid);
    free(buf->best_len);
    free(buf->best_episode_len);
    free(buf->best_return);
    free(buf->env_role);
    free(buf->env_active_slot);
    free(buf->env_best_slot);
    free(buf->env_sample_offset);
    free(buf->env_history_count);
    free(buf->env_episode_return);
    free(buf->env_prefix_return);
    free(buf->env_episode_len);
    free(buf->env_prefix_step);
    free(buf->env_last_observed_step);
    free(buf->best_generation);
    free(buf->env_best_generation);
    memset(buf, 0, sizeof(*buf));
}

static inline int curriculum_host_lock(StateBuffer* buf) {
    while (__sync_lock_test_and_set(&buf->host_lock, 1)) {}
    return 1;
}

static inline void curriculum_host_unlock(StateBuffer* buf) {
    __sync_lock_release(&buf->host_lock);
}

static int curriculum_fixed_agents_per_env(VecEnv* vec) {
    assert(vec->size > 0 && "state curriculum requires at least one env");
    int agents_per_env = vec->envs[0].num_agents;
    assert(agents_per_env > 0 && "env num_agents must be positive");
    for (int i = 0; i < vec->size; i++) {
        assert(vec->envs[i].num_agents == agents_per_env
            && "state curriculum requires fixed num_agents per env");
    }
    assert(vec->num_policies == 1
        && "state curriculum requires num_policies == 1 (identity env->agent mapping)");
    assert(vec->size * agents_per_env == vec->total_agents
        && "env agent counts must sum to total_agents");
    assert(vec->agents_per_buf % agents_per_env == 0
        && "state curriculum requires agents_per_buf divisible by num_agents");
    return agents_per_env;
}

static inline unsigned int curriculum_mix32(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static inline int curriculum_best_base(StateBuffer* buf, int slot) {
    return slot * buf->trajectory_max_len;
}

static inline int curriculum_count_valid(StateBuffer* buf) {
    int count = 0;
    for (int i = 0; i < buf->num_start_states; i++) {
        count += buf->best_valid[i] ? 1 : 0;
    }
    return count;
}

static inline void curriculum_seed_best(PuffeRL* pufferl) {
    StateBuffer* buf = pufferl->state_buf;
    VecEnv* vec = pufferl->vec;
    if (buf->seeded || vec == NULL || vec->size <= 0) {
        return;
    }

    for (int slot = 0; slot < buf->num_start_states; slot++) {
        Env* env = &vec->envs[slot % vec->size];
        PufferState* dst = buf->best_states + curriculum_best_base(buf, slot);
        dst[0] = env->state;
        buf->best_valid[slot] = 1;
        buf->best_len[slot] = 1;
        buf->best_episode_len[slot] = 0;
        buf->best_return[slot] = 0.0f;
        buf->best_state_return[curriculum_best_base(buf, slot)] = 0.0f;
        buf->best_state_step[curriculum_best_base(buf, slot)] = 0;
        buf->best_generation[slot] = 1;
    }
    buf->seeded = 1;
}

// Sample a pool slot whose best_return is at least min_return (admission
// floor). Returns -1 when none qualify. min_return <= 0 admits any valid slot.
static inline int curriculum_sample_best_slot(StateBuffer* buf, unsigned int salt,
        float min_return) {
    int valid = 0;
    for (int slot = 0; slot < buf->num_start_states; slot++) {
        if (buf->best_valid[slot] && buf->best_return[slot] >= min_return) {
            valid++;
        }
    }
    if (valid <= 0) {
        return -1;
    }
    int pick = (int)(curriculum_mix32(salt) % (unsigned int)valid);
    for (int slot = 0; slot < buf->num_start_states; slot++) {
        if (!buf->best_valid[slot] || buf->best_return[slot] < min_return) {
            continue;
        }
        if (pick == 0) {
            return slot;
        }
        pick--;
    }
    return -1;
}

static inline int curriculum_count_qualified(StateBuffer* buf, float min_return) {
    int count = 0;
    for (int i = 0; i < buf->num_start_states; i++) {
        if (buf->best_valid[i] && buf->best_return[i] >= min_return) {
            count++;
        }
    }
    return count;
}

// Checkpoint offset for a resumed episode. Uniform when tail_bias is off;
// otherwise weighted toward the trajectory tail (sqrt of a uniform draw),
// where accumulated setup (jokers / money / hand levels) lives.
static inline int curriculum_sample_offset(StateBuffer* buf, int slot,
        unsigned int salt, int tail_bias) {
    int len = slot >= 0 && slot < buf->num_start_states ? buf->best_len[slot] : 0;
    if (len <= 1) {
        return 0;
    }
    if (!tail_bias) {
        return (int)(curriculum_mix32(salt) % (unsigned int)len);
    }
    float u = (float)curriculum_mix32(salt ^ 0x9e3779b9U) / 4294967296.0f;
    int offset = (int)(sqrtf(u) * (float)len);
    if (offset >= len) {
        offset = len - 1;
    }
    if (offset < 0) {
        offset = 0;
    }
    return offset;
}

// Push the restored env's obs/rewards/terminals/mask to the device. t >= 0 is
// the worker path (rewrite the rollout slice the upcoming forward reads, since
// cpu_upload already wrote the pre-restore obs into it); t < 0 is the
// rollout_begin path (the horizon-boundary handoff in env.obs, which the next
// t=0 forward folds into slice 0). Stream-ordered with the caller's forward.
static inline void curriculum_push_env(PuffeRL* p, int env_idx, int t,
        cudaStream_t stream) {
    VecEnv* vec = p->vec;
    StateBuffer* buf = p->state_buf;
    int agent_start = env_idx * buf->agents_per_env;
    int n = buf->agents_per_env;
    size_t obs_bytes = (size_t)n * OBS_SIZE * sizeof(obs_t);
    if (t >= 0) {
        RolloutBuf rollouts = p->rollouts;
        if (p->hypers.async) {
            rollouts = rollout_time_view(&p->rollouts,
                p->write_slot * p->hypers.horizon, p->hypers.horizon);
        }
        long obs_stride = rollouts.observations.shape[1]
            * rollouts.observations.shape[2];
        cudaMemcpyAsync(rollouts.observations.data
                + (size_t)t * obs_stride + (size_t)agent_start * OBS_SIZE,
            vec->observations + (size_t)agent_start * OBS_SIZE,
            obs_bytes, cudaMemcpyHostToDevice, stream);
    } else {
        cudaMemcpyAsync(p->env.obs.data + (size_t)agent_start * OBS_SIZE,
            vec->observations + (size_t)agent_start * OBS_SIZE,
            obs_bytes, cudaMemcpyHostToDevice, stream);
    }
    cudaMemcpyAsync(p->env.rewards.data + agent_start,
        vec->rewards + agent_start, (size_t)n * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(p->env.terminals.data + agent_start,
        vec->terminals + agent_start, (size_t)n * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    if (vec->mask_size > 0) {
        cudaMemcpyAsync(p->env.action_mask.data
                + (size_t)agent_start * vec->mask_size,
            vec->action_mask + (size_t)agent_start * vec->mask_size,
            (size_t)n * vec->mask_size * sizeof(unsigned char),
            cudaMemcpyHostToDevice, stream);
    }
}

static inline void curriculum_set_env_state(Env* env, const PufferState* state,
        int clear_outputs) {
    env->state = *state;
    puffer_state_refresh(env);
    if (clear_outputs) {
        for (int a = 0; a < env->num_agents; a++) {
            env->agents[a].rewards[0] = 0.0f;
            env->agents[a].terminals[0] = 0.0f;
        }
    }
}

static inline int curriculum_active_row(StateBuffer* buf, int env_idx) {
    int row = env_idx - buf->num_vanilla_envs;
    if (row < 0 || row >= buf->max_active_envs) {
        return -1;
    }
    return row;
}

static inline void curriculum_record_state(StateBuffer* buf, int env_idx,
        const PufferState* state) {
    int row = buf->env_active_slot[env_idx];
    if (row < 0 || row >= buf->max_active_envs) {
        return;
    }
    int count = buf->env_history_count[env_idx];
    if (count < 0 || count >= buf->trajectory_max_len) {
        return;
    }
    int base = row * buf->trajectory_max_len + count;
    buf->active_states[base] = *state;
    buf->active_state_return[base] = buf->env_episode_return[env_idx];
    buf->active_state_step[base] = buf->env_episode_len[env_idx];
    buf->env_history_count[env_idx] = count + 1;
}

static inline int curriculum_should_record_checkpoint(StateBuffer* buf, int env_idx) {
    int interval = buf->checkpoint_interval;
    if (interval <= 1) {
        return 1;
    }
    int len = buf->env_episode_len[env_idx];
    return len > 0 && (len % interval) == 0;
}

static inline void curriculum_observe_step(StateBuffer* buf, int env_idx,
        Env* env, long step_serial) {
    int role = buf->env_role[env_idx];
    if (role != CURRICULUM_ROLE_FRESH && role != CURRICULUM_ROLE_CL) {
        return;
    }
    if (buf->env_last_observed_step[env_idx] == step_serial) {
        return;
    }
    buf->env_last_observed_step[env_idx] = step_serial;
    float reward = 0.0f;
    for (int a = 0; a < env->num_agents; a++) {
        float value = env->agents[a].rewards[0];
        if (!isnan(value) && !isinf(value)) {
            reward += value;
        }
    }
    buf->env_episode_return[env_idx] += reward;
    buf->env_episode_len[env_idx] += 1;
}

// Restore a best-state episode into env_idx. t selects the obs target for the
// device push (see curriculum_push_env); stream is where that push runs.
static inline int curriculum_start_env(PuffeRL* pufferl, int env_idx,
        int role, unsigned int salt, int clear_outputs, int t,
        cudaStream_t stream) {
    StateBuffer* buf = pufferl->state_buf;
    VecEnv* vec = pufferl->vec;
    int active_row = curriculum_active_row(buf, env_idx);
    if (active_row < 0) {
        return 0;
    }

    int slot = -1;
    int offset = 0;
    float prefix_return = 0.0f;
    int prefix_step = 0;
    long long generation = -1;
    PufferState start_state;

    float min_admit = pufferl->hypers.curriculum_min_admit;
    int tail_bias = pufferl->hypers.curriculum_cl_tail_bias;

    curriculum_host_lock(buf);
    // Fresh envs are the pool's searchers: they may start from any valid slot.
    // CL envs only resume from trajectories that cleared the admission floor.
    slot = curriculum_sample_best_slot(buf, salt,
        role == CURRICULUM_ROLE_CL ? min_admit : 0.0f);
    if (slot < 0 || !buf->best_valid[slot]) {
        curriculum_host_unlock(buf);
        return 0;
    }
    offset = role == CURRICULUM_ROLE_CL
        ? curriculum_sample_offset(buf, slot, salt ^ 0xa511e9b3U, tail_bias)
        : 0;
    int best_base = curriculum_best_base(buf, slot);
    start_state = buf->best_states[best_base + offset];
    if (role == CURRICULUM_ROLE_CL) {
        prefix_return = buf->best_state_return[best_base + offset];
        prefix_step = buf->best_state_step[best_base + offset];
    }
    generation = buf->best_generation[slot];
    curriculum_host_unlock(buf);

    Env* env = &vec->envs[env_idx];
    curriculum_set_env_state(env, &start_state, clear_outputs);
    buf->env_role[env_idx] = role;
    buf->env_active_slot[env_idx] = active_row;
    buf->env_best_slot[env_idx] = slot;
    buf->env_sample_offset[env_idx] = offset;
    buf->env_history_count[env_idx] = 0;
    buf->env_episode_return[env_idx] = 0.0f;
    buf->env_prefix_return[env_idx] = prefix_return;
    buf->env_episode_len[env_idx] = 0;
    buf->env_prefix_step[env_idx] = prefix_step;
    buf->env_best_generation[env_idx] = generation;
    if (clear_outputs) {
        buf->env_last_observed_step[env_idx] = -1;
    }
    curriculum_record_state(buf, env_idx, &env->state);
    curriculum_push_env(pufferl, env_idx, t, stream);
    return 1;
}

static inline float curriculum_total_return(StateBuffer* buf, int env_idx) {
    float ret = buf->env_episode_return[env_idx];
    if (buf->env_role[env_idx] == CURRICULUM_ROLE_CL) {
        ret += buf->env_prefix_return[env_idx];
    }
    if (isnan(ret) || isinf(ret)) {
        ret = -3.402823466e+38F;
    }
    return ret;
}

static inline int curriculum_total_episode_len(StateBuffer* buf, int env_idx) {
    int len = buf->env_episode_len[env_idx];
    if (buf->env_role[env_idx] == CURRICULUM_ROLE_CL) {
        len += buf->env_prefix_step[env_idx];
    }
    return len;
}

static inline int curriculum_replace_full(PuffeRL* pufferl, int slot, int env_idx,
        float terminal_return, int episode_len) {
    StateBuffer* buf = pufferl->state_buf;
    int row = buf->env_active_slot[env_idx];
    int count = buf->env_history_count[env_idx];
    if (slot < 0 || row < 0 || count <= 0) {
        return 0;
    }
    if (count > buf->trajectory_max_len) {
        count = buf->trajectory_max_len;
    }

    int active_base = row * buf->trajectory_max_len;
    float min_admit = pufferl->hypers.curriculum_min_admit;
    curriculum_host_lock(buf);
    int replace = 0;
    if (terminal_return < min_admit) {
        replace = 0;  // below admission floor: keep the current best
    } else if (!buf->best_valid[slot]) {
        replace = 1;
    } else if (terminal_return > buf->best_return[slot] + 1e-6f) {
        replace = 1;
    } else if (fabsf(terminal_return - buf->best_return[slot]) <= 1e-6f
            && episode_len < buf->best_episode_len[slot]) {
        replace = 1;
    }
    if (!replace) {
        curriculum_host_unlock(buf);
        return 0;
    }

    memcpy(buf->best_states + curriculum_best_base(buf, slot),
        buf->active_states + active_base,
        (size_t)count * sizeof(PufferState));
    memcpy(buf->best_state_return + curriculum_best_base(buf, slot),
        buf->active_state_return + active_base,
        (size_t)count * sizeof(float));
    memcpy(buf->best_state_step + curriculum_best_base(buf, slot),
        buf->active_state_step + active_base,
        (size_t)count * sizeof(int));
    buf->best_valid[slot] = 1;
    buf->best_len[slot] = count;
    buf->best_episode_len[slot] = episode_len;
    buf->best_return[slot] = terminal_return;
    buf->best_generation[slot]++;
    if (buf->best_generation[slot] <= 0) {
        buf->best_generation[slot] = 1;
    }
    curriculum_host_unlock(buf);
    return 1;
}

static inline int curriculum_replace_tail(PuffeRL* pufferl, int slot, int env_idx,
        float terminal_return, int episode_len) {
    StateBuffer* buf = pufferl->state_buf;
    int row = buf->env_active_slot[env_idx];
    int count = buf->env_history_count[env_idx];
    int offset = buf->env_sample_offset[env_idx];
    if (slot < 0 || row < 0 || count <= 0 || !buf->best_valid[slot]) {
        return 0;
    }
    if (offset < 0) {
        offset = 0;
    }
    if (offset >= buf->trajectory_max_len) {
        offset = buf->trajectory_max_len - 1;
    }
    int copy_count = count;
    if (offset + copy_count > buf->trajectory_max_len) {
        copy_count = buf->trajectory_max_len - offset;
    }
    if (copy_count <= 0) {
        return 0;
    }

    float prefix_return = buf->env_prefix_return[env_idx];
    int prefix_step = buf->env_prefix_step[env_idx];
    int active_base = row * buf->trajectory_max_len;

    curriculum_host_lock(buf);
    if (terminal_return <= buf->best_return[slot] + 1e-6f
            || buf->env_best_generation[env_idx] != buf->best_generation[slot]
            || terminal_return < pufferl->hypers.curriculum_min_admit) {
        curriculum_host_unlock(buf);
        return 0;
    }
    int best_base = curriculum_best_base(buf, slot);
    memcpy(buf->best_states + best_base + offset,
        buf->active_states + active_base,
        (size_t)copy_count * sizeof(PufferState));
    for (int i = 0; i < copy_count; i++) {
        buf->best_state_return[best_base + offset + i] =
            prefix_return + buf->active_state_return[active_base + i];
        buf->best_state_step[best_base + offset + i] =
            prefix_step + buf->active_state_step[active_base + i];
    }
    buf->best_len[slot] = offset + copy_count;
    buf->best_episode_len[slot] = episode_len;
    buf->best_return[slot] = terminal_return;
    buf->best_generation[slot]++;
    if (buf->best_generation[slot] <= 0) {
        buf->best_generation[slot] = 1;
    }
    curriculum_host_unlock(buf);
    return 1;
}

// Terminal processing runs in the worker thread after the env's puf_step
// auto-reset (env->terminals[0] still flags the finished transition). The
// episode's recorded checkpoints compete into best, then the env restarts
// from a sampled best state. t is the current horizon step: the restored
// obs must land in rollout slice t before the forward at t reads it.
static inline void curriculum_process_terminal(PuffeRL* pufferl, int env_idx,
        int buffer_idx, int t, cudaStream_t stream) {
    StateBuffer* buf = pufferl->state_buf;
    int role = buf->env_role[env_idx];
    if (role != CURRICULUM_ROLE_FRESH && role != CURRICULUM_ROLE_CL) {
        return;
    }
    int slot = buf->env_best_slot[env_idx];
    float terminal_return = curriculum_total_return(buf, env_idx);
    int episode_len = curriculum_total_episode_len(buf, env_idx);
    unsigned int salt = (unsigned int)(pufferl->seed
        + (uint64_t)pufferl->epoch * 1000003ULL
        + (uint64_t)buffer_idx * 9176ULL
        + (uint64_t)t * 101ULL
        + (uint64_t)env_idx * 17ULL);

    if (role == CURRICULUM_ROLE_FRESH) {
        curriculum_replace_full(pufferl, slot, env_idx, terminal_return,
            episode_len);
        curriculum_start_env(pufferl, env_idx, CURRICULUM_ROLE_FRESH,
            salt ^ 0x2f1bbcdcU, 0, t, stream);
    } else {
        curriculum_replace_tail(pufferl, slot, env_idx, terminal_return,
            episode_len);
        curriculum_start_env(pufferl, env_idx, CURRICULUM_ROLE_CL,
            salt ^ 0x9e3779b9U, 0, t, stream);
    }
}

// Called by each worker thread at the top of its per-step iteration, before
// the forward at horizon step t. Processes the transition t-1 -> t (observe
// reward; on terminal, splice into best and restart from a best state).
static void capture_curriculum_checkpoint(PuffeRL* pufferl, int buffer_idx, int t) {
    StateBuffer* buf = pufferl->state_buf;
    VecEnv* vec = pufferl->vec;
    long ticket = (long)t * (long)vec->buffers + (long)buffer_idx;
    while (__atomic_load_n(&buf->capture_ticket, __ATOMIC_SEQ_CST) != ticket) {}
    long rollout_step = buf->rollout_step_base;
    int env_start = vec->env_starts[buffer_idx];
    int env_count = vec->env_counts[buffer_idx];

    for (int i = 0; i < env_count; i++) {
        int env_idx = env_start + i;
        int role = buf->env_role[env_idx];
        if (role != CURRICULUM_ROLE_FRESH && role != CURRICULUM_ROLE_CL) {
            continue;
        }
        Env* env = &vec->envs[env_idx];
        if (t > 0) {
            long step_serial = rollout_step
                + (long)(t - 1) * (long)vec->total_agents;
            curriculum_observe_step(buf, env_idx, env, step_serial);
        }
        if (t > 0 && env->agents[0].terminals[0] > 0.5f) {
            curriculum_process_terminal(pufferl, env_idx, buffer_idx, t,
                pufferl->streams[buffer_idx]);
            continue;
        }
        if (t > 0 && curriculum_should_record_checkpoint(buf, env_idx)) {
            curriculum_record_state(buf, env_idx, &env->state);
        }
    }
    __sync_fetch_and_add(&buf->capture_ticket, 1);
}

// Main-thread hook, called once per rollout before the workers start. Seeds
// the best pool, absorbs the horizon-boundary transition, rebalances the
// vanilla/fresh/CL split (annealing cl_frac), and (re)starts fresh/CL envs.
// Any restored obs land in env.obs on p->default_stream; the caller orders
// worker streams after that upload (see rollout_start).
void curriculum_rollout_begin(PuffeRL* pufferl) {
    StateBuffer* buf = pufferl->state_buf;
    VecEnv* vec = pufferl->vec;
    Hypers* h = &pufferl->hypers;
    int total_envs = vec->size;
    long pending_step = pufferl->global_step - (long)vec->total_agents;
    if (pending_step < 0) {
        pending_step = 0;
    }
    __atomic_store_n(&buf->capture_ticket, 0, __ATOMIC_SEQ_CST);
    buf->rollout_step_base = pufferl->global_step;

    curriculum_seed_best(pufferl);
    if (curriculum_count_valid(buf) <= 0) {
        vec->log_env_limit = 0;
        return;
    }

    for (int env_idx = 0; env_idx < total_envs; env_idx++) {
        int role = buf->env_role[env_idx];
        if ((role == CURRICULUM_ROLE_FRESH || role == CURRICULUM_ROLE_CL)
                && vec->envs[env_idx].agents[0].rewards != NULL) {
            Env* env = &vec->envs[env_idx];
            curriculum_observe_step(buf, env_idx, env, pending_step);
            if (env->agents[0].terminals[0] > 0.5f) {
                curriculum_process_terminal(pufferl, env_idx, -1, -1,
                    pufferl->default_stream);
            } else if (curriculum_should_record_checkpoint(buf, env_idx)) {
                curriculum_record_state(buf, env_idx, &env->state);
            }
        }
    }

    float current_cl_frac = h->cl_frac;
    if (h->anneal_cl && h->total_timesteps > 0) {
        float progress = (float)pufferl->global_step / (float)h->total_timesteps;
        progress = fminf(fmaxf(progress, 0.0f), 1.0f);
        current_cl_frac *= 1.0f - progress;
    }
    int num_cl_envs = curriculum_clamp_int(
        (int)(current_cl_frac * (float)total_envs), 0, total_envs);
    int num_fresh_envs = curriculum_clamp_int(
        (int)(h->fresh_frac * (float)total_envs), 0, total_envs);
    // CL envs only resume from trajectories that cleared the admission floor;
    // until one exists they stay vanilla (still logged, still exploring).
    if (h->curriculum_min_admit > 0.0f
            && curriculum_count_qualified(buf, h->curriculum_min_admit) <= 0) {
        num_cl_envs = 0;
    }
    if (num_cl_envs + num_fresh_envs > total_envs) {
        num_cl_envs = total_envs - num_fresh_envs;
    }
    int active_envs = num_cl_envs + num_fresh_envs;
    if (active_envs > buf->max_active_envs) {
        int overflow = active_envs - buf->max_active_envs;
        num_cl_envs = curriculum_clamp_int(num_cl_envs - overflow, 0, total_envs);
        active_envs = num_cl_envs + num_fresh_envs;
    }
    if (active_envs > buf->max_active_envs) {
        num_fresh_envs = curriculum_clamp_int(
            buf->max_active_envs - num_cl_envs, 0, total_envs);
        active_envs = num_cl_envs + num_fresh_envs;
    }

    buf->num_cl_envs = num_cl_envs;
    buf->num_fresh_envs = num_fresh_envs;
    buf->num_vanilla_envs = total_envs - active_envs;
    vec->log_env_limit = active_envs > 0 ? buf->num_vanilla_envs : 0;

    for (int i = 0; i < buf->num_vanilla_envs; i++) {
        buf->env_role[i] = CURRICULUM_ROLE_VANILLA;
        buf->env_active_slot[i] = -1;
        buf->env_best_slot[i] = -1;
        buf->env_sample_offset[i] = 0;
        buf->env_history_count[i] = 0;
        buf->env_episode_return[i] = 0.0f;
        buf->env_prefix_return[i] = 0.0f;
        buf->env_episode_len[i] = 0;
        buf->env_prefix_step[i] = 0;
        buf->env_last_observed_step[i] = -1;
        buf->env_best_generation[i] = -1;
    }

    int started = 0;
    int fresh_start = buf->num_vanilla_envs;
    int cl_start = fresh_start + num_fresh_envs;
    for (int i = 0; i < num_fresh_envs; i++) {
        int env_idx = fresh_start + i;
        int active_row = curriculum_active_row(buf, env_idx);
        if (buf->env_role[env_idx] == CURRICULUM_ROLE_FRESH
                && buf->env_active_slot[env_idx] == active_row
                && buf->env_best_slot[env_idx] >= 0) {
            started++;
            continue;
        }
        unsigned int salt = (unsigned int)(pufferl->seed
            + pufferl->epoch * 4099 + env_idx * 31);
        started += curriculum_start_env(pufferl, env_idx,
            CURRICULUM_ROLE_FRESH, salt ^ 0xb5297a4dU, 1, -1,
            pufferl->default_stream);
    }
    for (int i = 0; i < num_cl_envs; i++) {
        int env_idx = cl_start + i;
        int active_row = curriculum_active_row(buf, env_idx);
        if (buf->env_role[env_idx] == CURRICULUM_ROLE_CL
                && buf->env_active_slot[env_idx] == active_row
                && buf->env_best_slot[env_idx] >= 0) {
            started++;
            continue;
        }
        unsigned int salt = (unsigned int)(pufferl->seed
            + pufferl->epoch * 65537 + env_idx * 131);
        started += curriculum_start_env(pufferl, env_idx,
            CURRICULUM_ROLE_CL, salt ^ 0x68e31da4U, 1, -1,
            pufferl->default_stream);
    }
    if (started <= 0) {
        buf->num_cl_envs = 0;
        buf->num_fresh_envs = 0;
        buf->num_vanilla_envs = total_envs;
        vec->log_env_limit = 0;
    }
}

// Per-epoch training-log status: role split and best-pool quality.
void curriculum_log(PuffeRL* pufferl, Dict* out) {
    StateBuffer* buf = pufferl->state_buf;
    if (buf == NULL) {
        return;
    }
    dict_set(out, "curriculum/fresh_envs", (double)buf->num_fresh_envs);
    dict_set(out, "curriculum/cl_envs", (double)buf->num_cl_envs);
    dict_set(out, "curriculum/best_count",
        (double)curriculum_count_valid(buf));
    float best_return = 0.0f;
    int best_len = 0;
    for (int i = 0; i < buf->num_start_states; i++) {
        if (buf->best_valid[i] && buf->best_return[i] > best_return) {
            best_return = buf->best_return[i];
        }
        if (buf->best_valid[i] && buf->best_len[i] > best_len) {
            best_len = buf->best_len[i];
        }
    }
    dict_set(out, "curriculum/best_return_max", (double)best_return);
    dict_set(out, "curriculum/best_len_max", (double)best_len);
}

#endif  // PUFFER_CURRICULUM
