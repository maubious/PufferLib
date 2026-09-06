// One wave evaluates a decision. During teacher forcing, decisions from the
// same row are independent: prefix entries are observed action tokens, not
// recurrent decoder states.
static constexpr float score_scale = 0.1767766952966369f;
struct Decision {
    int options, start, step, last, prefix_count;
    bool active;
    uint64_t legal;
    float gate[32], query[32], preactivation[32];
    float scores[64];
    float logsum, entropy;
};

__device__ __forceinline__ float wave_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        x += __shfl_down_sync(PUF_WARP_MASK, x, offset, 32);
    return __shfl_sync(PUF_WARP_MASK, x, 0, 32);
}

__device__ __forceinline__ float wave_max(float x) {
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        x = fmaxf(x, __shfl_down_sync(PUF_WARP_MASK, x, offset, 32));
    return __shfl_sync(PUF_WARP_MASK, x, 0, 32);
}

__device__ __forceinline__ void decision_forward(Decision* s, const precision_t* entities,
        const precision_t* actions, const precision_t* mask, const int* counts, int step, const float* evaluation) {
    int lane = threadIdx.x & 31;
    if (lane == 0) {
        do {
            s->active = false;
            if (step == 0) {
                s->active = true;
                s->options = ACTION_TYPE_COUNT;
                s->start = token_start(counts, BA_POOL_SECTIONS);
                s->legal = 0;
                for (int option = 0; option < ACTION_TYPE_COUNT; ++option)
                    if (mask_byte(mask, 0, option)) s->legal |= UINT64_C(1) << option;
                assert(s->legal);
                break;
            }

            if (step >= 8) {
                int position = step < 72 ? step - 8 : step - 72;
                int count = mask_byte(mask, 0, step < 72 ? POLICY_ORDER_HAND_COUNT_OFFSET : POLICY_ORDER_JOKER_COUNT_OFFSET);
                if (!mask_byte(mask, 0, POLICY_ORDER_ENABLED_OFFSET) || count <= 1 || position >= count) break;
                s->active = true;
                s->options = count;
                s->start = token_start(counts, step < 72 ? ZONE_HAND : ZONE_JOKER);
                s->legal = UINT64_MAX >> (64 - count);
                int offset = step < 72 ? 8 : 72;
                for (int j = 0; j < position; ++j)
                    s->legal &= ~(UINT64_C(1) << (int)to_float(actions[offset + j]));
                assert(s->legal);
                break;
            }
            ARContext context = {};
            context.type = ar_action_value(actions, 0, 0, ACTION_TYPE_COUNT);
            context.has_primary = (context.type >= ACTION_BUY_CARD && context.type <= ACTION_SWAP_HAND_RIGHT)
                || context.type == ACTION_BUY_AND_USE;
            context.primary = context.has_primary ? ar_action_value(actions, 0, 1, 64) : 0;
            context.primary_bits = context.has_primary ? mask_u64(mask, 0,
                POLICY_PRIMARY_OFFSET + context.type * POLICY_PRIMARY_BYTES) : 0;
            int entry = policy_selection_entry(context.type, context.primary);
            int offset = POLICY_SELECTION_OFFSET + entry * POLICY_SELECTION_BYTES;
            context.sb = entry >= 0 && mask_byte(mask, 0, offset + 1) ? offset : -1;
            if (context.sb >= 0) {
                context.min_count = mask_byte(mask, 0, context.sb);
                context.max_count = mask_byte(mask, 0, context.sb + 1);
                context.allowed = mask_u64(mask, 0, context.sb + 2);
                context.required = mask_u64(mask, 0, context.sb + 10);
            }
            context.count = ar_action_value(actions, 0, 2, 6);
            context.prev = -1;
            context.required_left = context.required;
            s->active = (step == 1 && context.has_primary)
                || (step == 2 && context.sb >= 0)
                || (step >= 3 && context.sb >= 0 && step - 3 < context.count);
            if (!s->active) break;
            int zone = step == 1 ? primary_zone(context.type) : ZONE_HAND;
            s->options = step == 2 ? 6 : counts[zone];
            s->start = step == 2 ? token_start(counts, BA_POOL_SECTIONS) + ACTION_TYPE_COUNT
                : token_start(counts, zone);
            if (step >= 3) {
                for (int j = 3; j < step; ++j) {
                    int option = (int)to_float(actions[j]);
                    context.selected |= UINT64_C(1) << option;
                    context.prev = option;
                }
                context.required_left = context.required & ~context.selected;
            }
            s->legal = step == 1 ? context.primary_bits : 0;
            if (step != 1) {
                for (int option = 0; option < s->options; ++option)
                    if (ar_option_legal(&context, mask, 0, step, option))
                        s->legal |= UINT64_C(1) << option;
            }
            assert(s->legal);
        } while (false);
    }
    __syncwarp();
    if (!s->active) return;
    if (__popcll(s->legal) == 1) {
        for (int option = lane; option < s->options; option += 32) s->scores[option] = 0.0f;
        if (lane == 0) { s->logsum = 0.0f; s->entropy = 0.0f; }
        __syncwarp();
        return;
    }
    float query = evaluation[lane];
    s->preactivation[lane] = evaluation[32 + lane];
    s->query[lane] = query;
    float maximum = -INFINITY;
    int channel = lane & 7;
    float q0 = __shfl_sync(PUF_WARP_MASK, query, channel, 32);
    float q1 = __shfl_sync(PUF_WARP_MASK, query, channel + 8, 32);
    float q2 = __shfl_sync(PUF_WARP_MASK, query, channel + 16, 32);
    float q3 = __shfl_sync(PUF_WARP_MASK, query, channel + 24, 32);
    for (int group = 0; group < s->options; group += 4) {
        int option = group + lane / 8;
        float score = -INFINITY;
        bool legal = option < s->options && (s->legal & (UINT64_C(1) << option));
        float product = 0.0f;
        if (legal) {
            const precision_t* key = entities + (s->start + option) * ENTITY_WIDTH;
            product = q0 * to_float(key[channel]) + q1 * to_float(key[channel + 8])
                + q2 * to_float(key[channel + 16]) + q3 * to_float(key[channel + 24]);
        }
        #pragma unroll
        for (int offset = 4; offset; offset >>= 1)
            product += __shfl_down_sync(PUF_WARP_MASK, product, offset, 8);
        product = __shfl_sync(PUF_WARP_MASK, product, 0, 8);
        if (legal) score = product * score_scale;
        if (channel == 0 && option < s->options) s->scores[option] = score;
        maximum = fmaxf(maximum, score);
    }
    __syncwarp();
    maximum = wave_max(maximum);
    float sum = 0.0f;
    float mean = 0.0f;
    for (int option = lane; option < s->options; option += 32) {
        if (!(s->legal & (UINT64_C(1) << option))) continue;
        float weight = __expf(s->scores[option] - maximum);
        sum += weight;
        mean += weight * (s->scores[option] - maximum);
    }
    sum = wave_sum(sum);
    mean = wave_sum(mean);
    if (lane == 0) {
        s->logsum = maximum + __logf(sum);
        s->entropy = __logf(sum) - mean / sum;
    }
    __syncwarp();
}

__device__ __forceinline__ void decision_backward(Decision* s, const precision_t* entities,
        const precision_t* parameters, int selected, float dlogp, float dentropy,
        float* state_gradient, float* entity_gradient, float* parameter_gradient, float* prefix_gradient) {
    if (!s->active || __popcll(s->legal) == 1) return;
    int lane = threadIdx.x & 31;
    float gradient_query = 0.0f;
    float query = s->query[lane];
    for (int option = 0; option < s->options; ++option) {
        if (!(s->legal & (UINT64_C(1) << option))) continue;
        float logp = s->scores[option] - s->logsum;
        float probability = __expf(logp);
        float gradient = ((option == selected ? 1.0f : 0.0f) - probability) * dlogp
            - dentropy * probability * (logp + s->entropy);
        int entity = s->start + option;
        float k = to_float(entities[entity * ENTITY_WIDTH + lane]);
        gradient_query += gradient * score_scale * k;
        atomicAdd(entity_gradient + entity * ENTITY_WIDTH + lane,
            gradient * score_scale * query);
    }
    float u = s->preactivation[lane];
    float sigmoid = 1.0f / (1.0f + __expf(-u));
    float gate = s->gate[lane];
    float du = gradient_query * gate * sigmoid * (1.0f + u * (1.0f - sigmoid));
    atomicAdd(state_gradient + lane, du);
    atomicAdd(state_gradient + 32 + lane, gradient_query * u * sigmoid * (gate * (2.0f - gate)));
    atomicAdd(parameter_gradient + ROLE_OFFSET + s->step * 32 + lane, du);
    if (s->last >= 0) atomicAdd(entity_gradient + s->last * ENTITY_WIDTH + lane, du);
    float normalization = s->prefix_count ? rsqrtf((float)s->prefix_count) : 0.0f;
    prefix_gradient[lane] = du * normalization;

}

// Build all teacher-forced queries with one prefix scan per row.
__global__ void cache_queries(const precision_t* state, const precision_t* entities,
        const precision_t* parameters, const precision_t* actions, const precision_t* mask,
        const int* counts, float* evaluations, int batch) {
    int row = blockIdx.x * 4 + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (row >= batch) return;
    actions += row * ACTION_STORAGE_SIZE;
    mask += row * POLICY_MASK_SIZE;
    counts += row * BA_POOL_SECTIONS;
    entities += (int64_t)row * ENTITY_COUNT * ENTITY_WIDTH;
    state += row * DECODER_COLUMNS;
    int type = (int)to_float(actions[0]);
    bool primary = (type >= ACTION_BUY_CARD && type <= ACTION_SWAP_HAND_RIGHT) || type == ACTION_BUY_AND_USE;
    int choice = primary ? (int)to_float(actions[1]) : 0;
    int entry = policy_selection_entry(type, choice);
    bool selection = entry >= 0 && mask_byte(mask, 0, POLICY_SELECTION_OFFSET + entry * POLICY_SELECTION_BYTES + 1);
    int selected = selection ? (int)to_float(actions[2]) : 0;
    bool ordering = mask_byte(mask, 0, POLICY_ORDER_ENABLED_OFFSET);
    int hand = ordering ? mask_byte(mask, 0, POLICY_ORDER_HAND_COUNT_OFFSET) : 0;
    int jokers = ordering ? mask_byte(mask, 0, POLICY_ORDER_JOKER_COUNT_OFFSET) : 0;
    float sum = 0.0f, last = 0.0f;
    int length = 0, last_entity = -1;
    float global = to_float(state[lane]);
    float gate = 2.0f / (1.0f + __expf(-2.0f * to_float(state[32 + lane])));
    for (int index = 0; index < 8 + hand + jokers; ++index) {
        int step = index < 8 + hand ? index : 72 + index - 8 - hand;
        float role = to_float(parameters[ROLE_OFFSET + step * 32 + lane]);
        float u = global + role + last + (length ? sum * rsqrtf((float)length) : 0.0f);
        float* evaluation = evaluations + ((int64_t)row * DECODER_STEPS + step) * EVALUATION_WIDTH;
        evaluation[lane] = u / (1.0f + __expf(-u)) * gate;
        evaluation[32 + lane] = u;
        if (lane == 0) { evaluation[130] = (float)length; evaluation[131] = (float)last_entity; }
        int entity = -1;
        if (step == 0) entity = token_start(counts, BA_POOL_SECTIONS) + type;
        else if (step == 1 && primary) entity = token_start(counts, primary_zone(type)) + choice;
        else if (step == 2 && selection) entity = token_start(counts, BA_POOL_SECTIONS) + ACTION_TYPE_COUNT + selected;
        else if (step >= 3 && step < 8 && step - 3 < selected)
            entity = token_start(counts, ZONE_HAND) + (int)to_float(actions[step]);
        else if (step >= 8)
            entity = token_start(counts, step < 72 ? ZONE_HAND : ZONE_JOKER) + (int)to_float(actions[step]);
        if (entity >= 0) {
            last_entity = entity;
            last = to_float(entities[entity * ENTITY_WIDTH + lane]);
            sum += role * last;
            ++length;
        }
    }
}

__global__ void cache_decisions(const precision_t* entities, const precision_t* actions,
        const precision_t* mask, const int* counts, float* statistics, float* evaluations, int batch) {
    constexpr int waves = 2;
    __shared__ Decision decisions[waves];
    int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    int row = blockIdx.x;
    int hand = mask_byte(mask, row * POLICY_MASK_SIZE, POLICY_ORDER_HAND_COUNT_OFFSET);
    int jokers = mask_byte(mask, row * POLICY_MASK_SIZE, POLICY_ORDER_JOKER_COUNT_OFFSET);
    if (!mask_byte(mask, row * POLICY_MASK_SIZE, POLICY_ORDER_ENABLED_OFFSET)) hand = jokers = 0;
    for (int i = threadIdx.x; i < DECODER_STEPS * 2; i += blockDim.x)
        statistics[row * DECODER_STEPS * 2 + i] = 0.0f;
    __syncthreads();
    for (int index = warp; index < 8 + hand + jokers; index += waves) {

        int step = index < 8 + hand ? index : 72 + index - 8 - hand;
        int item = row * DECODER_STEPS + step;
        Decision* s = decisions + warp;
        decision_forward(s, entities + (int64_t)row * ENTITY_COUNT * ENTITY_WIDTH,
            actions + row * ACTION_STORAGE_SIZE, mask + row * POLICY_MASK_SIZE,
            counts + row * BA_POOL_SECTIONS, step, evaluations + (int64_t)item * EVALUATION_WIDTH);
        if (lane == 0) {
            float* evaluation = evaluations + (int64_t)item * EVALUATION_WIDTH;
            evaluation[132] = s->active ? (float)s->options : 0.0f;
            evaluation[133] = s->active ? (float)s->start : 0.0f;
            reinterpret_cast<uint64_t*>(evaluation + 134)[0] = s->active ? s->legal : 0;
        }
        if (s->active && __popcll(s->legal) > 1) {
            float* evaluation = evaluations + (int64_t)item * EVALUATION_WIDTH;
            evaluation[lane] = s->query[lane];
            evaluation[32 + lane] = s->preactivation[lane];
            for (int option = lane; option < s->options; option += 32)
                evaluation[64 + option] = s->scores[option];
            if (lane == 0) { evaluation[128] = s->logsum; evaluation[129] = s->entropy; }
        }
        if (lane == 0) {
            int selected = (int)to_float(actions[row * ACTION_STORAGE_SIZE + step]);
            assert(!s->active || (selected >= 0 && selected < s->options
                && (s->legal & (UINT64_C(1) << selected))));
            statistics[item * 2] = s->active ? s->scores[selected] - s->logsum : 0.0f;
            statistics[item * 2 + 1] = s->active ? s->entropy : 0.0f;
        }
    }
}

__global__ void finish_likelihood(const precision_t* state, const float* statistics,
        const float* old_logprobs, precision_t* importance, precision_t* values,
        float* logprobs, int batch) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    float logp = 0.0f;
    for (int step = 0; step < DECODER_STEPS; ++step) logp += statistics[(row * DECODER_STEPS + step) * 2];
    logprobs[row] = logp;
    importance[row] = from_float(__expf(logp - old_logprobs[row]));
    values[row] = state[row * DECODER_COLUMNS + DECODER_STATE];
}

__global__ void backward_decisions(const precision_t* state, const precision_t* entities,
        const precision_t* parameters, const precision_t* actions, const precision_t* mask,
        const int* counts, const float* coefficients, float* state_gradient,
        float* entity_gradient, float* parts, float* evaluations, int batch) {
    constexpr int waves = 2;
    __shared__ Decision decisions[waves];
    int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    int row = blockIdx.x;
    int hand = mask_byte(mask, row * POLICY_MASK_SIZE, POLICY_ORDER_HAND_COUNT_OFFSET);
    int jokers = mask_byte(mask, row * POLICY_MASK_SIZE, POLICY_ORDER_JOKER_COUNT_OFFSET);
    if (!mask_byte(mask, row * POLICY_MASK_SIZE, POLICY_ORDER_ENABLED_OFFSET)) hand = jokers = 0;
    for (int index = warp; index < 8 + hand + jokers; index += waves) {
        int step = index < 8 + hand ? index : 72 + index - 8 - hand;
        int item = row * DECODER_STEPS + step;
        Decision* s = decisions + warp;
        float* evaluation = evaluations + (int64_t)item * EVALUATION_WIDTH;
        if (lane == 0) {
            s->step = step;
            s->options = (int)evaluation[132];
            s->active = s->options > 0;
            s->start = (int)evaluation[133];
            s->legal = reinterpret_cast<const uint64_t*>(evaluation + 134)[0];
            s->prefix_count = (int)evaluation[130];
            s->last = (int)evaluation[131];
        }
        __syncwarp();
        if (!s->active || __popcll(s->legal) == 1) {
            evaluation[lane] = 0.0f;
            continue;
        }
        s->query[lane] = evaluation[lane];
        s->preactivation[lane] = evaluation[32 + lane];
        s->gate[lane] = 2.0f / (1.0f + __expf(-2.0f * to_float(state[row * DECODER_COLUMNS + 32 + lane])));
        for (int option = lane; option < s->options; option += 32)
            s->scores[option] = evaluation[64 + option];
        if (lane == 0) { s->logsum = evaluation[128]; s->entropy = evaluation[129]; }
        __syncwarp();
        int selected = (int)to_float(actions[row * ACTION_STORAGE_SIZE + step]);
        decision_backward(s, entities + (int64_t)row * ENTITY_COUNT * ENTITY_WIDTH,
            parameters, selected, coefficients[row * 2], coefficients[row * 2 + 1],
            state_gradient + row * DECODER_STATE,
            entity_gradient + (int64_t)row * ENTITY_COUNT * ENTITY_WIDTH,
            parts + (row % COND_STRIPE_COUNT) * AR_CONDITION_SIZE, evaluation);
    }

}

// Each earlier token receives the sum of normalized query gradients from
// later decisions. A reverse scan replaces the quadratic prefix atomics.
__global__ void accumulate_prefix(const precision_t* entities, const precision_t* parameters,
        const precision_t* actions, const precision_t* mask, const int* counts,
        const float* evaluations, float* entity_gradient, float* parts, float* raw_gradient, int batch) {
    int row = blockIdx.x * 4 + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (row >= batch) return;
    actions += row * ACTION_STORAGE_SIZE;
    mask += row * POLICY_MASK_SIZE;
    counts += row * BA_POOL_SECTIONS;
    entities += (int64_t)row * ENTITY_COUNT * ENTITY_WIDTH;
    entity_gradient += (int64_t)row * ENTITY_COUNT * ENTITY_WIDTH;
    parts += (row % COND_STRIPE_COUNT) * AR_CONDITION_SIZE;
    int type = (int)to_float(actions[0]);
    bool primary = (type >= ACTION_BUY_CARD && type <= ACTION_SWAP_HAND_RIGHT) || type == ACTION_BUY_AND_USE;
    int choice = primary ? (int)to_float(actions[1]) : 0;
    int entry = policy_selection_entry(type, choice);
    bool selection = entry >= 0 && mask_byte(mask, 0, POLICY_SELECTION_OFFSET + entry * POLICY_SELECTION_BYTES + 1);
    int selected = selection ? (int)to_float(actions[2]) : 0;
    bool ordering = mask_byte(mask, 0, POLICY_ORDER_ENABLED_OFFSET);
    int hand = ordering ? mask_byte(mask, 0, POLICY_ORDER_HAND_COUNT_OFFSET) : 0;
    int jokers = ordering ? mask_byte(mask, 0, POLICY_ORDER_JOKER_COUNT_OFFSET) : 0;
    float sum = 0.0f;
    for (int index = 8 + hand + jokers - 1; index >= 0; --index) {
        int step = index < 8 + hand ? index : 72 + index - 8 - hand;
        int entity = -1;
        if (step == 0) entity = token_start(counts, BA_POOL_SECTIONS) + type;
        else if (step == 1 && primary) entity = token_start(counts, primary_zone(type)) + choice;
        else if (step == 2 && selection) entity = token_start(counts, BA_POOL_SECTIONS) + ACTION_TYPE_COUNT + selected;
        else if (step >= 3 && step < 8 && step - 3 < selected)
            entity = token_start(counts, ZONE_HAND) + (int)to_float(actions[step]);
        else if (step >= 8)
            entity = token_start(counts, step < 72 ? ZONE_HAND : ZONE_JOKER) + (int)to_float(actions[step]);
        if (entity >= 0) {
            entity_gradient[entity * ENTITY_WIDTH + lane] +=
                sum * to_float(parameters[ROLE_OFFSET + step * 32 + lane]);
            atomicAdd(parts + ROLE_OFFSET + step * 32 + lane,
                sum * to_float(entities[entity * ENTITY_WIDTH + lane]));
        }
        sum += evaluations[((int64_t)row * DECODER_STEPS + step) * EVALUATION_WIDTH + lane];
    }
    int live = 0;
    for (int zone = 0; zone < BA_POOL_SECTIONS; ++zone) live += counts[zone];
    for (int slot = 0; slot < live + DECODER_CATEGORIES; ++slot) {
        float value = entity_gradient[slot * ENTITY_WIDTH + lane];
        if (slot < live)
            raw_gradient[((int64_t)row * BA_KEY_CAP + slot) * ENTITY_WIDTH + lane] = value;
        else atomicAdd(parts + (slot - live) * ENTITY_WIDTH + lane, value);
    }
}

__global__ void ppo_loss_balatro(float* partials, PPOKernelArgs a, PPOGraphArgs g) {
    int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    int row = blockIdx.x * (PPO_THREADS / 32) + warp;
    int batch = a.N * a.T_seq;
    __shared__ float losses[LOSS_N][PPO_THREADS / 32];
    if (lane == 0) {
        float metrics[LOSS_N] = {};
        if (row < batch) {
            float inv = 1.0f / batch;
            float logratio = a.grad_values_pred[row] - g.old_logprobs[row];
            float ratio = __expf(logratio);
            float advantage = to_float(g.advantages[row]);
            float clipped_ratio = fminf(fmaxf(ratio, 1.0f - a.clip_coef), 1.0f + a.clip_coef);
            float loss = -advantage * ratio;
            float clipped_loss = -advantage * clipped_ratio;
            float dlogp = clipped_loss > loss ? 0.0f : -advantage * ratio * inv;
            float predicted = to_float(a.logits[row * DECODER_COLUMNS + DECODER_STATE]);
            float value = to_float(g.values[row]);
            float target = to_float(g.returns[row]);
            float clipped = value + fminf(fmaxf(predicted - value, -a.vf_clip_coef), a.vf_clip_coef);
            float square = (predicted - target) * (predicted - target);
            float clipped_square = (clipped - target) * (clipped - target);
            a.grad_values_pred[row] = clipped_square > square ? 0.0f : a.vf_coef * inv * (predicted - target);
            float entropy = 0.0f;
            for (int step = 0; step < DECODER_STEPS; ++step)
                entropy += a.statistics[(row * DECODER_STEPS + step) * 2 + 1];
            float coefficient = *a.ent_coef;
            a.coefficients[row * 2] = dlogp;
            a.coefficients[row * 2 + 1] = -coefficient * inv;
            metrics[LOSS_PG] = fmaxf(loss, clipped_loss) * inv;
            metrics[LOSS_VF] = 0.5f * fmaxf(square, clipped_square) * inv;
            metrics[LOSS_ENT] = entropy * inv;
            metrics[LOSS_TOTAL] = metrics[LOSS_PG] + a.vf_coef * metrics[LOSS_VF] - coefficient * metrics[LOSS_ENT];
            metrics[LOSS_OLD_APPROX_KL] = -logratio * inv;
            metrics[LOSS_APPROX_KL] = (ratio - 1.0f - logratio) * inv;
            metrics[LOSS_CLIPFRAC] = (fabsf(ratio - 1.0f) > a.clip_coef) * inv;
            metrics[LOSS_IMP] = ratio * inv;
        }
        for (int metric = 0; metric < LOSS_N; ++metric) losses[metric][warp] = metrics[metric];
    }
    __syncthreads();
    if (threadIdx.x < LOSS_N) {
        float sum = 0.0f;
        for (int i = 0; i < PPO_THREADS / 32; ++i) sum += losses[threadIdx.x][i];
        partials[blockIdx.x * LOSS_N + threadIdx.x] = sum;
    }
}
