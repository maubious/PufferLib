// Query[32], preactivation[32], scores[64], logsum, entropy,
// prefix count, last entity, option count, entity start, and a uint64 legality mask.
static constexpr int EVALUATION_WIDTH = 136;

__global__ void accumulate_prefix(const precision_t*, const precision_t*, const precision_t*,
    const precision_t*, const int*, const float*, float*, float*, float*, int);

// Cache entity features once and clear only live gradient entries.
__global__ void cache_entities(precision_t* entities, const precision_t* raw,
        const precision_t* parameters, const int* counts, float* gradient) {
    int row = blockIdx.x;
    int live = 0;
    for (int zone = 0; zone < BA_POOL_SECTIONS; ++zone) live += counts[row * BA_POOL_SECTIONS + zone];
    for (int index = threadIdx.x; index < (live + DECODER_CATEGORIES) * 32; index += blockDim.x) {
        int entity = index / 32, d = index % 32;
        entities[((int64_t)row * ENTITY_COUNT + entity) * 32 + d] = entity < live
            ? raw[((int64_t)row * BA_KEY_CAP + entity) * 32 + d]
            : parameters[(entity - live) * 32 + d];
        if (gradient) gradient[((int64_t)row * ENTITY_COUNT + entity) * 32 + d] = 0.0f;
    }
}

__global__ void finish_decoder_gradient(
        precision_t* output, const float* state, const float* value, int batch) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * DECODER_COLUMNS) return;
    int row = index / DECODER_COLUMNS, column = index % DECODER_COLUMNS;
    output[index] = from_float(column == DECODER_STATE ? value[row] : state[row * DECODER_STATE + column]);
}

__global__ void finish_parameter_gradient(precision_t* output, const float* parts, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    float sum = 0.0f;
    for (int stripe = 0; stripe < COND_STRIPE_COUNT; ++stripe)
        sum += parts[stripe * AR_CONDITION_SIZE + index];
    output[index] = from_float(sum);
}

struct BalatroDecoderWeights {
    Prec weight, logstd, condition;
    int hidden_dim, output_dim;
    bool continuous, ar;
};

struct BalatroDecoderActivations {
    // Framework-visible DecoderActivations prefix.
    Prec out, grad_out, saved_input, grad_input, wgrad_scratch, logstd_scratch,
        condition_scratch;
    Float condition_accum, cond_accum_parts;
    BalatroEncoderActivations* enc;
    const precision_t* actions;
    const precision_t* mask;
    Prec entities, dall, weight_grad;
    Float entity_gradient, decisions, coefficients, evaluations;
};

static Prec decode(void* weights, void* activations, Prec input, cudaStream_t stream) {
    auto* w = (BalatroDecoderWeights*)weights;
    auto* a = (BalatroDecoderActivations*)activations;
    int batch = input.shape[0];
    if (a->saved_input.data) puf_copy(&a->saved_input, &input, stream);
    puf_mm(&input, &w->weight, &a->out, stream);
    cache_entities<<<batch, 256, 0, stream>>>(
        a->entities.data, a->enc->keys.data, w->condition.data, a->enc->counts.data, a->entity_gradient.data);
    return a->out;
}

static Prec differentiate_decoder(void* weights, void* activations,
        Float state_gradient, Float logstd_gradient, Float value_gradient, cudaStream_t stream) {
    auto* w = (BalatroDecoderWeights*)weights;
    auto* a = (BalatroDecoderActivations*)activations;
    int batch = a->out.shape[0];
    accumulate_prefix<<<(batch + 3) / 4, 128, 0, stream>>>(a->entities.data, w->condition.data,
        a->actions, a->mask, a->enc->counts.data, a->evaluations.data, a->entity_gradient.data,
        a->cond_accum_parts.data, a->enc->key_grad.data, batch);
    finish_parameter_gradient<<<grid_size(AR_CONDITION_SIZE), BLOCK_SIZE, 0, stream>>>(
        a->condition_scratch.data, a->cond_accum_parts.data, AR_CONDITION_SIZE);
    finish_decoder_gradient<<<grid_size(batch * DECODER_COLUMNS), BLOCK_SIZE, 0, stream>>>(
        a->dall.data, state_gradient.data, value_gradient.data, batch);
    puf_mm_tn(&a->dall, &a->saved_input, &a->weight_grad, stream);
    puf_mm_nn(&a->dall, &w->weight, &a->grad_input, stream);
    return a->grad_input;
}

static void initialize_decoder(void* weights, uint64_t* seed, cudaStream_t stream) {
    auto* w = (BalatroDecoderWeights*)weights;
    puf_kaiming_init(&w->weight, 1.0f, (*seed)++, stream);
    puf_normal_init(&w->condition, 0.1f, (*seed)++, stream);

}

static void register_decoder_parameters(void* weights, Allocator* allocator) {
    auto* w = (BalatroDecoderWeights*)weights;
    w->weight = {.shape = {DECODER_COLUMNS, w->hidden_dim}};
    w->condition = {.shape = {AR_CONDITION_SIZE}};
    alloc_register(allocator, &w->weight);
    alloc_register(allocator, &w->condition);
}

static void register_decoder_training(void* weights, void* activations,
        Allocator* acts, Allocator* grads, int batch) {
    auto* w = (BalatroDecoderWeights*)weights;
    auto* a = (BalatroDecoderActivations*)activations;
    *a = {};
    a->enc = ba_enc_last;
    a->out = {.shape = {batch, DECODER_COLUMNS}};
    a->saved_input = {.shape = {batch, w->hidden_dim}};
    a->grad_input = {.shape = {batch, w->hidden_dim}};
    a->condition_scratch = {.shape = {AR_CONDITION_SIZE}};
    a->cond_accum_parts = {.shape = {COND_STRIPE_COUNT, AR_CONDITION_SIZE}};
    a->entities = {.shape = {batch, ENTITY_COUNT, ENTITY_WIDTH}};
    a->entity_gradient = {.shape = {batch, ENTITY_COUNT, ENTITY_WIDTH}};
    a->evaluations = {.shape = {batch, DECODER_STEPS, EVALUATION_WIDTH}};
    a->decisions = {.shape = {batch, DECODER_STEPS, 2}};
    a->coefficients = {.shape = {batch, 2}};
    a->dall = {.shape = {batch, DECODER_COLUMNS}};
    a->weight_grad = {.shape = {DECODER_COLUMNS, w->hidden_dim}};
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->saved_input);
    alloc_register(acts, &a->grad_input);
    alloc_register(acts, &a->cond_accum_parts);
    alloc_register(acts, &a->entities);
    alloc_register(acts, &a->entity_gradient);
    alloc_register(acts, &a->evaluations);
    alloc_register(acts, &a->decisions);
    alloc_register(acts, &a->coefficients);
    alloc_register(acts, &a->dall);
    alloc_register(grads, &a->weight_grad);
    alloc_register(grads, &a->condition_scratch);
}

static void register_decoder_rollout(void* weights, void* activations, Allocator* allocator, int batch) {
    auto* a = (BalatroDecoderActivations*)activations;
    *a = {};
    a->enc = ba_enc_last;
    a->out = {.shape = {batch, DECODER_COLUMNS}};
    a->entities = {.shape = {batch, ENTITY_COUNT, ENTITY_WIDTH}};
    alloc_register(allocator, &a->out);
    alloc_register(allocator, &a->entities);
}

static void* create_decoder_weights(void* self) {
    auto* decoder = (Decoder*)self;
    auto* w = (BalatroDecoderWeights*)calloc(1, sizeof(BalatroDecoderWeights));
    w->hidden_dim = decoder->hidden_dim;
    w->output_dim = decoder->output_dim;
    w->ar = true;
    return w;
}

static void create_balatro_decoder(Decoder* decoder) {
    *decoder = Decoder{
        .forward = decode,
        .backward = differentiate_decoder,
        .init_weights = initialize_decoder,
        .reg_params = register_decoder_parameters,
        .reg_train = register_decoder_training,
        .reg_rollout = register_decoder_rollout,
        .create_weights = create_decoder_weights,
        .hidden_dim = decoder->hidden_dim,
        .output_dim = decoder->output_dim,
        .continuous = false,
        .ar = true,
        .activation_size = sizeof(BalatroDecoderActivations),
    };
}
