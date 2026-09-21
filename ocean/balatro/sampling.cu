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
    int hand_count = mask_byte(mask, row * POLICY_MASK_SIZE, POLICY_ORDER_HAND_COUNT_OFFSET);
    int joker_count = mask_byte(mask, row * POLICY_MASK_SIZE, POLICY_ORDER_JOKER_COUNT_OFFSET);
    bool ordering = mask_byte(mask, row * POLICY_MASK_SIZE, POLICY_ORDER_ENABLED_OFFSET);
    float sum = 0.0f, last = 0.0f, card_sum = 0.0f;
    int length = 0, selected_count = 0;
    float global = to_float(state.data[row * DECODER_COLUMNS + lane]);
    float gate = 2.0f / (1.0f + __expf(-2.0f * to_float(state.data[row * DECODER_COLUMNS + 32 + lane])));
    for (int step = 0; step < ACTION_STORAGE_SIZE; ++step) {
        if (step == BASE_ACTION_HEADS && !ordering) break;
        if (step >= BASE_ACTION_HEADS && step < ORDER_JOKER_OFFSET && step >= BASE_ACTION_HEADS + hand_count) step = ORDER_JOKER_OFFSET;
        if (step >= ORDER_JOKER_OFFSET + joker_count) break;
        float role = to_float(parameters[ROLE_OFFSET + step * 32 + lane]);
        float u;
        if (step >= 4 && step < BASE_ACTION_HEADS && step - 3 < selected_count) {
            int pos = step - 3;
            float card_mean = card_sum / (float)pos;
            u = 0.5f * global + role + 1.5f * card_mean + 0.5f * last;
        } else {
            u = global + role + last + (length ? sum * rsqrtf((float)length) : 0.0f);
        }
        s->query[lane] = u / (1.0f + __expf(-u)) * gate;
        s->preactivation[lane] = u;
        __syncwarp();
        decision_forward(s, entities + (int64_t)row * ENTITY_COUNT * ENTITY_WIDTH,
            action, mask + row * POLICY_MASK_SIZE, counts + row * POOL_SECTIONS, step, s->query);
        if (s->active) {
            int selected = __ffsll((unsigned long long)s->legal) - 1;
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
            if (step == 2) {
                selected_count = __shfl_sync(PUF_WARP_MASK, selected, 0, 32);
            }
            if (lane == 0) action[step] = (float)selected;
        }
        __syncwarp();
        if (s->active || step >= BASE_ACTION_HEADS) {
            int start = step >= BASE_ACTION_HEADS ? token_start(counts + row * POOL_SECTIONS,
                step < ORDER_JOKER_OFFSET ? ZONE_HAND : ZONE_JOKER) : s->start;
            int entity = start + (int)action[step];
            last = to_float(entities[((int64_t)row * ENTITY_COUNT + entity) * ENTITY_WIDTH + lane]);
            sum += role * last;
            ++length;
            if (step >= 3 && step < BASE_ACTION_HEADS && step - 3 < selected_count) {
                card_sum += last;
            }
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
