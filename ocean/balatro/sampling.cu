__global__ void sample_logits_balatro(
        Prec state, float* actions, precision_t* logprobs,
        precision_t* values, curandStatePhilox4_32_10_t* rng_states,
        const precision_t* mask, const precision_t* parameters,
        const precision_t* entities, const int* counts) {
    constexpr int waves = BLOCK_SIZE / 32;
    __shared__ Decision decisions[waves];
    int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    int row = blockIdx.x * waves + warp;
    if (row >= state.shape[0]) return;
    float* action = actions + row * ACTION_STORAGE_SIZE;
    for (int step = lane; step < ACTION_STORAGE_SIZE; step += 32) action[step] = 0.0f;
    __syncwarp();
    curandStatePhilox4_32_10_t rng;
    if (lane == 0) rng = rng_states[row];
    float logp = 0.0f;
    Decision* s = decisions + warp;
    __shared__ float s_vec[waves][32];
    int hand_count = mask_byte(mask, row * POLICY_MASK_SIZE, POLICY_ORDER_HAND_COUNT_OFFSET);
    int joker_count = mask_byte(mask, row * POLICY_MASK_SIZE, POLICY_ORDER_JOKER_COUNT_OFFSET);
    bool ordering = mask_byte(mask, row * POLICY_MASK_SIZE, POLICY_ORDER_ENABLED_OFFSET);
    float last = 0.0f, card_sum = 0.0f, a_val = 0.0f, c_val = 0.0f;
    int selected_count = 0, sampled_type = 0;
    bool has_last = false;
    float global = to_float(state.data[row * DECODER_COLUMNS + lane]);

    float w_trans[32], w_synergy[32];
    #pragma unroll
    for (int a = 0; a < 32; ++a) {
        w_trans[a] = to_float(parameters[TRANS_OFFSET + a * 32 + lane]);
        w_synergy[a] = to_float(parameters[SYNERGY_OFFSET + a * 32 + lane]);
    }

    for (int step = 0; step < ACTION_STORAGE_SIZE; ++step) {
        if (step == BASE_ACTION_HEADS && !ordering) break;
        if (step >= BASE_ACTION_HEADS && step < ORDER_JOKER_OFFSET && step >= BASE_ACTION_HEADS + hand_count) step = ORDER_JOKER_OFFSET;
        if (step >= ORDER_JOKER_OFFSET + joker_count) break;
        if (step == BASE_ACTION_HEADS || step == ORDER_JOKER_OFFSET) {
            has_last = false;
        }
        float role = to_float(parameters[ROLE_OFFSET + step * 32 + lane]);
        float act_val = (step == 1) ? a_val : (step >= 2 && step < BASE_ACTION_HEADS) ? (a_val + c_val) : 0.0f;
        float t_val = 0.0f;
        if (has_last) {
            s_vec[warp][lane] = last;
            __syncwarp();
            #pragma unroll
            for (int a = 0; a < 32; ++a) {
                t_val += w_trans[a] * s_vec[warp][a];
            }
        }
        float s_val = 0.0f;
        if (step >= 4 && step < BASE_ACTION_HEADS && step - 3 < selected_count) {
            int pos = step - 3;
            float card_mean = card_sum / (float)pos;
            s_vec[warp][lane] = card_mean;
            __syncwarp();
            #pragma unroll
            for (int a = 0; a < 32; ++a) {
                s_val += w_synergy[a] * s_vec[warp][a];
            }
        }
        float u = global + role + act_val + t_val + s_val;
        float gate_role = to_float(parameters[GATE_ROLE_OFFSET + step * 32 + lane]);
        float g_val = to_float(state.data[row * DECODER_COLUMNS + 32 + lane]) + gate_role;
        float gate = 2.0f / (1.0f + __expf(-2.0f * g_val));
        s->query[lane] = u / (1.0f + __expf(-u)) * gate;
        s->preactivation[lane] = u;
        __syncwarp();
        decision_forward(s, entities + (int64_t)row * ENTITY_COUNT * ENTITY_WIDTH,
            action, mask + row * POLICY_MASK_SIZE, counts + row * POOL_SECTIONS, step, s->query);
        int selected = 0;
        if (s->active) {
            selected = __ffsll((unsigned long long)s->legal) - 1;
            if (__popcll(s->legal) > 1) {
                float threshold = lane == 0 ? curand_uniform(&rng) : 0.0f;
                threshold = __shfl_sync(PUF_WARP_MASK, threshold, 0, 32);
                bool first_legal = lane < s->options && (s->legal & (UINT64_C(1) << lane));
                bool second_legal = lane + 32 < s->options && (s->legal & (UINT64_C(1) << (lane + 32)));
                float first = first_legal ? __expf(s->scores[lane] - s->logsum) : 0.0f;
                float second = second_legal ? __expf(s->scores[lane + 32] - s->logsum) : 0.0f;
                #pragma unroll
                for (int offset = 1; offset < 32; offset <<= 1) {
                    float a = __shfl_up_sync(PUF_WARP_MASK, first, offset, 32);
                    float b = __shfl_up_sync(PUF_WARP_MASK, second, offset, 32);
                    if (lane >= offset) { first += a; second += b; }
                }
                second += __shfl_sync(PUF_WARP_MASK, first, 31, 32);
                uint32_t first_hits = (uint32_t)__ballot_sync(PUF_WARP_MASK, first_legal && threshold <= first);
                uint32_t second_hits = (uint32_t)__ballot_sync(PUF_WARP_MASK, second_legal && threshold <= second);
                selected = first_hits ? __ffs(first_hits) - 1 : second_hits ? 31 + __ffs(second_hits)
                    : 63 - __clzll((unsigned long long)s->legal);
                if (lane == 0) logp += s->scores[selected] - s->logsum;
            }
            if (step == 0) {
                sampled_type = selected;
                int type_entity = token_start(counts + row * POOL_SECTIONS, POOL_SECTIONS) + selected;
                float type_val = to_float(entities[((int64_t)row * ENTITY_COUNT + type_entity) * ENTITY_WIDTH + lane]);
                s_vec[warp][lane] = type_val;
                __syncwarp();
                #pragma unroll
                for (int a = 0; a < 32; ++a) {
                    a_val += to_float(parameters[ACTION_OFFSET + a * 32 + lane]) * s_vec[warp][a];
                }
            } else if (step == 1) {
                bool primary = (sampled_type >= ACTION_BUY_CARD && sampled_type <= ACTION_SWAP_HAND_RIGHT) || sampled_type == ACTION_BUY_AND_USE;
                int choice = selected;
                int zone = primary ? primary_zone(sampled_type) : -1;
                const int* row_counts = counts + row * POOL_SECTIONS;
                if (primary && zone >= 0 && choice >= 0 && choice < row_counts[zone]) {
                    int choice_entity = token_start(row_counts, zone) + choice;
                    float choice_val = to_float(entities[((int64_t)row * ENTITY_COUNT + choice_entity) * ENTITY_WIDTH + lane]);
                    s_vec[warp][lane] = choice_val;
                    __syncwarp();
                    #pragma unroll
                    for (int a = 0; a < 32; ++a) {
                        c_val += to_float(parameters[ACTION_OFFSET + a * 32 + lane]) * s_vec[warp][a];
                    }
                }
            } else if (step == 2) {
                selected_count = selected;
            }
            if (lane == 0) action[step] = (float)selected;
        }
        __syncwarp();
        if (step >= 3 && step < BASE_ACTION_HEADS && step - 3 < selected_count) {
            int card_idx = selected;
            int entity = token_start(counts + row * POOL_SECTIONS, ZONE_HAND) + card_idx;
            last = to_float(entities[((int64_t)row * ENTITY_COUNT + entity) * ENTITY_WIDTH + lane]);
            has_last = true;
            card_sum += last;
        } else if (step >= BASE_ACTION_HEADS && s->active) {
            int start = token_start(counts + row * POOL_SECTIONS, step < ORDER_JOKER_OFFSET ? ZONE_HAND : ZONE_JOKER);
            int entity = start + selected;
            last = to_float(entities[((int64_t)row * ENTITY_COUNT + entity) * ENTITY_WIDTH + lane]);
            has_last = true;
        }
        __syncwarp();
    }
    if (lane == 0) {
        logprobs[row] = from_float(logp);
        values[row] = state.data[row * DECODER_COLUMNS + DECODER_STATE];
        rng_states[row] = rng;
    }
}

static void sample_actions(void* weights, void* activations,
        const Sampling& input, cudaStream_t stream) {
    int mask_stride = (int)input.mask.shape[1];
    assert(mask_stride == POLICY_MASK_SIZE && "Balatro requires mask stride (act_n) to equal POLICY_MASK_SIZE (985)");
    auto* w = (BalatroDecoderWeights*)weights;
    auto* a = (BalatroDecoderActivations*)activations;
    int batch = input.output.shape[0];
    constexpr int waves = BLOCK_SIZE / 32;
    sample_logits_balatro<<<(batch + waves - 1) / waves, BLOCK_SIZE, 0, stream>>>(
        input.output, input.actions.data, input.logprobs.data, input.values.data,
        input.rng, input.mask.data, w->condition.data, a->entities.data, a->enc->counts.data);
}
