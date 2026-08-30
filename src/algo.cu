// PufferNet model API + architecture
#ifndef AR_CONDITION_SIZE
#define AR_CONDITION_SIZE 0
#endif
// Row-striped condition-gradient staging: atomics spread across this many
// copies of the condition tensor to cut per-address contention, then summed.
constexpr int COND_STRIPE_COUNT = 8;
// Writing custom nets in 4.0+ requires a fair bit of code because you are
// responsible for defining your own activation and gradient buffers.
// You usually only ever need a custom Encoder.
typedef void (*init_weights_fn)(void* weights, ulong* seed, cudaStream_t stream);
typedef void (*reg_params_fn)(void* weights, Allocator* alloc);
typedef void (*reg_train_fn)(void* weights, void* buf, Allocator* acts, Allocator* grads, int B_TT);
typedef void (*reg_rollout_fn)(void* weights, void* buf, Allocator* alloc, int B);
typedef void* (*create_weights_fn)(void* self);
typedef Prec (*forward_fn)(void* weights, void* activations, Prec input, cudaStream_t stream);
typedef Prec (*encoder_forward_fn)(void* weights, void* activations,
    PolicyObs input, cudaStream_t stream);
typedef void (*encoder_backward_fn)(void* weights, void* activations,
    Prec grad, cudaStream_t stream);
typedef Prec (*decoder_backward_fn)(void* weights, void* activations,
    Float grad_logits, Float grad_logstd, Float grad_value, cudaStream_t stream);
typedef Prec (*network_forward_fn)(void* weights, Prec x,
    Prec state, void* activations, cudaStream_t stream);
typedef Prec (*network_forward_train_fn)(void* weights, Prec x,
    Prec state, Prec rnn_resets, void* activations, int agent_off,
    cudaStream_t stream);
typedef Prec (*network_backward_fn)(void* weights,
    Prec grad, void* activations, cudaStream_t stream);

struct Encoder {
    encoder_forward_fn forward;
    encoder_backward_fn backward;
    init_weights_fn init_weights;
    reg_params_fn reg_params;
    reg_train_fn reg_train;
    reg_rollout_fn reg_rollout;
    create_weights_fn create_weights;
    int in_dim, out_dim;
    size_t activation_size;  // sizeof(EncoderActivations) or custom override
};

struct EncoderWeights {
    Prec weight;
    int in_dim, out_dim;
};

struct EncoderActivations {
    Prec out, saved_input, decoded_input, wgrad_scratch;
};

struct Decoder {
    forward_fn forward;
    decoder_backward_fn backward;
    init_weights_fn init_weights;
    reg_params_fn reg_params;
    reg_train_fn reg_train;
    reg_rollout_fn reg_rollout;
    create_weights_fn create_weights;
    int hidden_dim, output_dim;
    bool continuous;
    bool ar;
    size_t activation_size;
};

struct DecoderWeights {
    Prec weight, logstd, condition;
    int hidden_dim, output_dim;
    bool continuous;
    bool ar;
};

struct DecoderActivations {
    Prec out, grad_out, saved_input, grad_input, wgrad_scratch, logstd_scratch,
        condition_scratch;
    Float condition_accum;
    Float cond_accum_parts;  // (COND_STRIPE_COUNT, AR_CONDITION_SIZE) atomic staging
};

struct Network {
    network_forward_fn forward;
    network_forward_train_fn forward_train;
    network_backward_fn backward;
    init_weights_fn init_weights;
    reg_params_fn reg_params;
    reg_train_fn reg_train;
    reg_rollout_fn reg_rollout;
    create_weights_fn create_weights;
    int hidden, num_layers, horizon;
};

thread_local cublasHandle_t g_cublas_handle = NULL;
thread_local void* g_cublas_workspace = NULL;
// Side-stream dW (mm_tn) overlaps dX (mm_nn) on main during linear bwd.
thread_local cublasHandle_t g_cublas_dw_handle = NULL;
thread_local cudaEvent_t g_main_ready = NULL;
thread_local void* g_cublas_dw_workspace = NULL;
thread_local cudaStream_t g_dw_stream = NULL;
thread_local cudaEvent_t g_dw_done = NULL;
#ifdef USE_ROCM
thread_local rocblas_handle g_rocblas_handle = NULL;
#endif

static void cublas_init_one(cublasHandle_t* handle, void** workspace) {
    const size_t ws_bytes = 32 * 1024 * 1024;
    assert(cublasCreate(handle) == CUBLAS_STATUS_SUCCESS);
    assert(cudaMalloc(workspace, ws_bytes) == cudaSuccess);
    assert(cublasSetWorkspace(*handle, *workspace, ws_bytes) == CUBLAS_STATUS_SUCCESS);
#ifdef USE_ROCM
    assert(hipblasSetMathMode(*handle, HIPBLAS_DEFAULT_MATH) == HIPBLAS_STATUS_SUCCESS);
#else
    assert(cublasSetMathMode(*handle, CUBLAS_DEFAULT_MATH) == CUBLAS_STATUS_SUCCESS);
#endif
}

void cublas_init_handle() {
    cublas_init_one(&g_cublas_handle, &g_cublas_workspace);
    cublas_init_one(&g_cublas_dw_handle, &g_cublas_dw_workspace);
    cudaStreamCreateWithFlags(&g_dw_stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&g_dw_done, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&g_main_ready, cudaEventDisableTiming);
}

#ifdef USE_ROCM
void rocblas_init_handle() {
    assert(rocblas_create_handle(&g_rocblas_handle) == rocblas_status_success);
    assert(rocblas_set_workspace(g_rocblas_handle, g_cublas_workspace,
        32 * 1024 * 1024) == rocblas_status_success);
}
#endif

// Dense row-major GEMM: C = alpha * op_a(A) @ op_b(B) + beta * C
static void cublasGemmExDense(cublasHandle_t handle,
        cublasOperation_t op_a, cublasOperation_t op_b,
        int M, int N, int K, void* A, void* B, void* C,
        cudaStream_t stream, float alpha = 1.0f, float beta = 0.0f) {
    int lda = (op_a == CUBLAS_OP_N) ? K : M;
    int ldb = (op_b == CUBLAS_OP_N) ? N : K;
    assert(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS);
    cublasStatus_t st = cublasGemmEx(handle, op_b, op_a, N, M, K, &alpha,
        B, CUBLAS_PRECISION, ldb, A, CUBLAS_PRECISION, lda, &beta,
        C, CUBLAS_PRECISION, N, CUBLAS_COMPUTE, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "GEMM failed: %d (%d x %d x %d, ops %d %d, ld %d %d)\n",
            (int)st, M, N, K, (int)op_a, (int)op_b, lda, ldb);
        abort();
    }
}

// out(...,N) = alpha * a(...,K) @ b(N,K)^T + beta * out  — leading dims folded into M
void puf_mm(Prec* a, Prec* b, Prec* out, cudaStream_t stream,
        float alpha = 1.0f, float beta = 0.0f) {
    int M = batch_size(a->shape) * a->shape[ndim(a->shape)-2];
    int K = a->shape[ndim(a->shape)-1];
    int N = b->shape[ndim(b->shape)-2];
    cublasGemmExDense(g_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K,
        a->data, b->data, out->data, stream, alpha, beta);
}

// out(M,N) = alpha * a(...,M)^T @ b(...,N) + beta * out  — leading dims folded into K
void puf_mm_tn(Prec* a, Prec* b, Prec* out, cudaStream_t stream,
        float alpha = 1.0f, float beta = 0.0f,
        cublasHandle_t handle = g_cublas_handle) {
    int M = a->shape[ndim(a->shape)-1];
    int K = batch_size(a->shape) * a->shape[ndim(a->shape)-2];
    int N = b->shape[ndim(b->shape)-1];
    cublasGemmExDense(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
        a->data, b->data, out->data, stream, alpha, beta);
}

// out(...,N) = alpha * a(...,K) @ b(K,N) + beta * out  — leading dims folded into M
void puf_mm_nn(Prec* a, Prec* b, Prec* out, cudaStream_t stream,
        float alpha = 1.0f, float beta = 0.0f) {
    int M = batch_size(a->shape) * a->shape[ndim(a->shape)-2];
    int K = a->shape[ndim(a->shape)-1];
    int N = b->shape[ndim(b->shape)-1];
    cublasGemmExDense(g_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
        a->data, b->data, out->data, stream, alpha, beta);
}

#ifdef USE_ROCM
// ROCm GEMM with distinct residual and output:
// out = alpha * a @ b + beta * residual.
void puf_mm_nn_residual(Prec* a, Prec* b, Prec* residual, Prec* out,
        cudaStream_t stream, float alpha, float beta) {
    int M = batch_size(a->shape) * a->shape[ndim(a->shape)-2];
    int K = a->shape[ndim(a->shape)-1];
    int N = b->shape[ndim(b->shape)-1];
#ifdef PRECISION_FLOAT
    rocblas_datatype type = rocblas_datatype_f32_r;
#else
    rocblas_datatype type = rocblas_datatype_bf16_r;
#endif
    assert(rocblas_set_stream(g_rocblas_handle, stream) == rocblas_status_success);
    rocblas_status status = rocblas_gemm_ex(
        g_rocblas_handle, rocblas_operation_none, rocblas_operation_none,
        N, M, K, &alpha, b->data, type, N, a->data, type, K,
        &beta, residual->data, type, N, out->data, type, N,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    if (status != rocblas_status_success) {
        fprintf(stderr, "residual GEMM failed: %d (%d x %d x %d)\n",
            (int)status, M, N, K);
        abort();
    }
}
#endif


// Queue dW (mm_tn) on side stream once inputs are ready on main. Per-layer
// wgrad buffers are disjoint so multiple dWs can queue; puf_dw_join before muon.
void puf_mm_tn_async_after(Prec* a, Prec* b, Prec* out, cudaStream_t main_stream) {
#ifdef USE_ROCM
    // ROCm hipBLAS stream rebinding is not reliable for the auxiliary handle
    // here (it can return INVALID_VALUE after a promoted worker resumes). Keep
    // the gradient GEMM on the learner stream; correctness matters more than
    // overlapping this small backward-side workload.
    puf_mm_tn(a, b, out, main_stream, 1.0f, 0.0f, g_cublas_handle);
#else
    cudaEventRecord(g_main_ready, main_stream);
    cudaStreamWaitEvent(g_dw_stream, g_main_ready, 0);
    puf_mm_tn(a, b, out, g_dw_stream, 1.0f, 0.0f, g_cublas_dw_handle);
#endif
}

void puf_dw_join(cudaStream_t consumer) {
#ifdef USE_ROCM
    // puf_mm_tn_async_after is synchronous with respect to consumer on ROCm.
    (void)consumer;
#else
    cudaEventRecord(g_dw_done, g_dw_stream);
    cudaStreamWaitEvent(consumer, g_dw_done, 0);
#endif
}


// Weight init (needs cast, grid_size, numel/ndim from pufferl substrate).
__global__ void uniform_scale_kernel(float* data, float bound, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f * bound - bound;
    }
}

// Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))
void puf_kaiming_init(Prec* dst,
        float gain, ulong seed, cudaStream_t stream) {
    assert(ndim(dst->shape) == 2);
    long rows = dst->shape[0], cols = dst->shape[1];
    assert(rows > 0 && cols > 0);
    long n = rows * cols;
    float bound = gain / sqrtf((float)cols);
    float* buf;
    cudaMalloc(&buf, n * sizeof(float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, buf, n);
    curandDestroyGenerator(gen);
    uniform_scale_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(buf, bound, n);
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dst->data, buf, n);
    cudaFree(buf);
}

__device__ __forceinline__ float sigmoid(float x) {
    float z = expf(-fabsf(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}

__device__ __forceinline__ float lerp(float a, float b, float w) {
    float diff = b - a;
    return (fabsf(w) < 0.5f) ? a + w * diff : b - diff * (1.0f - w);
}

// Rollout MinGRU step: h = (1-z)*h + z*h_tilde, highway out = s*h + (1-s)*x.
__global__ void mingru_gate(precision_t* out, precision_t* next_state,
        const precision_t* combined, const precision_t* state_in,
        const precision_t* x_in, int H, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * H;
    if (idx >= N) {
        return;
    }

    int b = idx / H;
    int h = idx % H;

    int combined_base = b * 3 * H;
    float hidden = to_float(combined[combined_base + h]);
    float gate = to_float(combined[combined_base + H + h]);
    float proj = to_float(combined[combined_base + 2*H + h]);
    float state = to_float(state_in[idx]);
    float x = to_float(x_in[idx]);

    float z = sigmoid(gate);
    float h_tilde = (hidden >= 0.0f) ? hidden + 0.5f : sigmoid(hidden);
    float h_out = lerp(state, h_tilde, z);
    next_state[idx] = from_float(h_out);

    float s = sigmoid(proj);
    out[idx] = from_float(s * h_out + (1.0f - s) * x);
}

// Train scan buffers. Cache post-reset h inputs; recompute outputs in bwd.
struct PrefixScan {
    precision_t* combined_ptr = NULL;
    precision_t* state_ptr = NULL;
    precision_t* input_ptr = NULL;  // (B, T, H) pre-projection input (highway)
    precision_t* rnn_resets_ptr = NULL;  // (B, T), reset before observation[t]
    int B = 0, T = 0, H = 0;
    Prec scan_h;  // (B, T, H), recurrent input to each step
    Prec out, next_state;
    Prec grad_combined, grad_state;
    Prec grad_input;
};

// Train scan: match rollout's precision_t recurrent state between steps.
__global__ void mingru_scan_forward_seq(PrefixScan scan) {
    int T_seq = scan.T, H = scan.H, B = scan.B;
    precision_t* __restrict__ out = scan.out.data;
    precision_t* __restrict__ next_state = scan.next_state.data;
    precision_t* __restrict__ scan_h = scan.scan_h.data;
    const precision_t* __restrict__ combined = scan.combined_ptr;
    const precision_t* __restrict__ state = scan.state_ptr;
    const precision_t* __restrict__ input = scan.input_ptr;
    const precision_t* __restrict__ rnn_resets = scan.rnn_resets_ptr;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }

    int b = idx / H;
    int h = idx % H;

    int bH = b * H;
    int H3 = 3 * H;
    int H2 = 2 * H;
    int bHT = bH * T_seq;
    const int out_base = bHT + h;
    int cbase = 3 * bHT;

    // h_0
    float h_t = to_float(state[bH + h]);
    int h_base = b * T_seq * H + h;

    const precision_t* __restrict__ combined_h_base = &combined[cbase + h];
    const precision_t* __restrict__ combined_g_base = &combined[cbase + H + h];
    const precision_t* __restrict__ combined_p_base = &combined[cbase + H2 + h];

    int out_curr = out_base;
    int t_offset = 0;

    for (int t = 0; t < T_seq; t++) {
        // rnn_resets[t] marks that observation[t] starts a new episode.
        if (rnn_resets != NULL && to_float(rnn_resets[b * T_seq + t]) != 0.0f) {
            h_t = 0.0f;
        }
        scan_h[h_base + t * H] = from_float(h_t);

        float hidden_val = to_float(combined_h_base[t_offset]);
        float gate_val = to_float(combined_g_base[t_offset]);
        float proj_val = to_float(combined_p_base[t_offset]);
        float x_val = to_float(input[out_base + t * H]);

        // h = (1-z)*h + z*h_tilde  (exact sigmoid; matches prior train log-space)
        float z = sigmoid(gate_val);
        float h_tilde = (hidden_val >= 0.0f) ? hidden_val + 0.5f : sigmoid(hidden_val);
        h_t = lerp(h_t, h_tilde, z);

        float proj_sigmoid = sigmoid(proj_val);
        out[out_curr] = from_float(proj_sigmoid * h_t + (1.0f - proj_sigmoid) * x_val);
        h_t = to_float(from_float(h_t));

        out_curr += H;
        t_offset += H3;
    }

    next_state[bH + h] = from_float(h_t);
}

// Linear reverse scan. Cache is bf16 h only; a,z,h_tilde recomputed from gates in f32.
__global__ void mingru_scan_backward(PrefixScan scan,
        const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ grad_next_state) {
    int T_seq = scan.T, H = scan.H, B = scan.B;
    precision_t* __restrict__ grad_combined = scan.grad_combined.data;
    precision_t* __restrict__ grad_state = scan.grad_state.data;
    precision_t* __restrict__ grad_input = scan.grad_input.data;
    const precision_t* __restrict__ combined = scan.combined_ptr;
    const precision_t* __restrict__ input = scan.input_ptr;
    const precision_t* __restrict__ rnn_resets = scan.rnn_resets_ptr;
    const precision_t* __restrict__ scan_h = scan.scan_h.data;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }

    int b = idx / H;
    int h = idx % H;

    int bHT = b * H * T_seq;
    int cbase = 3 * bHT;
    int H3 = 3 * H;
    int H2 = 2 * H;
    const int state_idx = b * H + h;
    const int out_base = bHT + h;
    const int h_base = b * T_seq * H + h;

    const precision_t* __restrict__ combined_h_base = &combined[cbase + h];
    const precision_t* __restrict__ combined_g_base = &combined[cbase + H + h];
    const precision_t* __restrict__ combined_p_base = &combined[cbase + H2 + h];

    precision_t* __restrict__ grad_combined_h_base = &grad_combined[cbase + h];
    precision_t* __restrict__ grad_combined_g_base = &grad_combined[cbase + H + h];
    precision_t* __restrict__ grad_combined_p_base = &grad_combined[cbase + H2 + h];

    // dh flowing into h_t from the future (and grad_next at t=T).
    float dh = to_float(grad_next_state[state_idx]);

    for (int t = T_seq; t >= 1; --t) {
        int t0 = t - 1;  // 0-based step
        int t_offset = t0 * H3;
        int input_idx = out_base + t0 * H;

        float h_prev = to_float(scan_h[h_base + (t - 1) * H]);

        float hidden_val = to_float(combined_h_base[t_offset]);
        float gate_val = to_float(combined_g_base[t_offset]);
        float proj_val = to_float(combined_p_base[t_offset]);
        float x_val = to_float(input[input_idx]);
        float grad_out_val = to_float(grad_out[input_idx]);

        float z = sigmoid(gate_val);
        float h_tilde = (hidden_val >= 0.0f) ? hidden_val + 0.5f : sigmoid(hidden_val);
        float h_t = lerp(h_prev, h_tilde, z);
        float proj_sigmoid = sigmoid(proj_val);

        // highway: out = s*h + (1-s)*x
        float dh_from_out = grad_out_val * proj_sigmoid;
        float grad_proj = grad_out_val * (h_t - x_val) * proj_sigmoid * (1.0f - proj_sigmoid);
        grad_input[input_idx] = from_float(grad_out_val * (1.0f - proj_sigmoid));
        grad_combined_p_base[t_offset] = from_float(grad_proj);

        float dh_total = dh + dh_from_out;

        // h = (1-z)*h_prev + z*h_tilde
        float d_h_prev = dh_total * (1.0f - z);
        float d_h_tilde = dh_total * z;
        float d_z = dh_total * (h_tilde - h_prev);

        // z = sigmoid(gate)
        float d_gate = d_z * z * (1.0f - z);
        // h_tilde = hidden+0.5  or  sigmoid(hidden)
        float d_hidden = (hidden_val >= 0.0f)
            ? d_h_tilde
            : d_h_tilde * h_tilde * (1.0f - h_tilde);

        grad_combined_h_base[t_offset] = from_float(d_hidden);
        grad_combined_g_base[t_offset] = from_float(d_gate);

        // terminal before this step zeroed h_prev contribution into this step;
        // after bwd through the step, cut gradient flow into pre-reset state.
        dh = d_h_prev;
        if (rnn_resets != NULL &&
                to_float(rnn_resets[b * T_seq + t0]) != 0.0f) {
            dh = 0.0f;
        }
    }

    grad_state[state_idx] = from_float(dh);
}

__global__ void sum_rows_to_precision_kernel(precision_t* __restrict__ dst,
        const float* __restrict__ src, int R, int C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= C) {
        return;
    }
    float sum = 0.0f;
    for (int r = 0; r < R; r++) {
        sum += src[r * C + col];
    }
    dst[col] = from_float(sum);
}

__global__ void assemble_decoder_grad(
        precision_t* __restrict__ dst, const float* __restrict__ grad_logits,
        const float* __restrict__ grad_value, int B_TT, int od, int od_plus_1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B_TT * od_plus_1) {
        return;
    }
    int row = idx / od_plus_1, col = idx % od_plus_1;
    dst[idx] = from_float((col < od) ? grad_logits[row * od + col] : grad_value[row]);
}

Prec encoder_forward(void* w, void* activations, PolicyObs input, cudaStream_t stream) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
#ifdef PUFFER_PACKED_OBS
    Prec* dense = a->saved_input.data ? &a->saved_input : &a->decoded_input;
    int n = (int)numel(input.shape);
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dense->data, input.data, n);
    puf_mm(dense, &ew->weight, &a->out, stream);
#else
    // Train acts register saved_input; rollout acts leave it null.
    if (a->saved_input.data) {
        puf_copy(&a->saved_input, &input, stream);
    }
    puf_mm(&input, &ew->weight, &a->out, stream);
#endif
    return a->out;
}

void encoder_backward(void* w, void* activations, Prec grad, cudaStream_t stream) {
    EncoderActivations* a = (EncoderActivations*)activations;
    // Last layer before join: nothing left on main to overlap, so sync dW.
    puf_mm_tn(&grad, &a->saved_input, &a->wgrad_scratch, stream);
}

void encoder_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    EncoderWeights* ew = (EncoderWeights*)w;
    Prec wt = {
        .data = ew->weight.data,
        .shape = {ew->out_dim, ew->in_dim},
    };
    puf_kaiming_init(&wt, sqrtf(2.0f), (*seed)++, stream);
}

void encoder_reg_params(void* w, Allocator* alloc) {
    EncoderWeights* ew = (EncoderWeights*)w;
    ew->weight = {.shape = {ew->out_dim, ew->in_dim}};
    alloc_register(alloc, &ew->weight);
}

void encoder_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B_TT) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    *a = (EncoderActivations){
        .out =              {.shape = {B_TT, ew->out_dim}},
        .saved_input =      {.shape = {B_TT, ew->in_dim}},
        .wgrad_scratch =    {.shape = {ew->out_dim, ew->in_dim}},
    };
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->saved_input);
    alloc_register(grads, &a->wgrad_scratch);
}

void encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    a->out = {.shape = {B, ew->out_dim}};
    alloc_register(alloc, &a->out);
#ifdef PUFFER_PACKED_OBS
    a->decoded_input = {.shape = {B, ew->in_dim}};
    alloc_register(alloc, &a->decoded_input);
#endif
}

void* encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    EncoderWeights* ew = (EncoderWeights*)calloc(1, sizeof(EncoderWeights));
    ew->in_dim = e->in_dim; ew->out_dim = e->out_dim;
    return ew;
}

Prec decoder_forward(void* w, void* activations, Prec input, cudaStream_t stream) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    if (a->saved_input.data) {
        puf_copy(&a->saved_input, &input, stream);
    }
    puf_mm(&input, &dw->weight, &a->out, stream);
    return a->out;
}

void decoder_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    DecoderWeights* dw = (DecoderWeights*)w;
    Prec wt = {
        .data = dw->weight.data,
        .shape = {dw->output_dim + 1, dw->hidden_dim},
    };
    puf_kaiming_init(&wt, 1.0f, (*seed)++, stream);
    if (dw->ar) cudaMemsetAsync(dw->condition.data, 0,
        numel(dw->condition.shape) * sizeof(precision_t), stream);
}

void decoder_reg_params(void* w, Allocator* alloc) {
    DecoderWeights* dw = (DecoderWeights*)w;
    dw->weight = {.shape = {dw->output_dim + 1, dw->hidden_dim}};
    alloc_register(alloc, &dw->weight);
    if (dw->continuous) {
        dw->logstd = {.shape = {1, dw->output_dim}};
        alloc_register(alloc,&dw->logstd);
    }
    if (dw->ar) {
        dw->condition = {.shape = {AR_CONDITION_SIZE}};
        alloc_register(alloc, &dw->condition);
    }
}

void decoder_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B_TT) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    int od1 = dw->output_dim + 1;
    *a = (DecoderActivations){
        .out =              {.shape = {B_TT, od1}},
        .grad_out =         {.shape = {B_TT, od1}},
        .saved_input =      {.shape = {B_TT, dw->hidden_dim}},
        .grad_input =       {.shape = {B_TT, dw->hidden_dim}},
        .wgrad_scratch =    {.shape = {od1, dw->hidden_dim}},
        .logstd_scratch =   {.shape = {1, dw->output_dim}},
    };
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->saved_input);
    alloc_register(acts, &a->grad_out);
    alloc_register(acts, &a->grad_input);
    alloc_register(grads, &a->wgrad_scratch);
    if (dw->continuous) {
        alloc_register(grads, &a->logstd_scratch);
    }
    if (dw->ar) {
        a->condition_scratch = {.shape = {AR_CONDITION_SIZE}};
        a->condition_accum = {.shape = {AR_CONDITION_SIZE}};
        a->cond_accum_parts = {.shape = {COND_STRIPE_COUNT, AR_CONDITION_SIZE}};
        alloc_register(grads, &a->condition_scratch);
        alloc_register(acts, &a->condition_accum);
        alloc_register(acts, &a->cond_accum_parts);
    }
}

void decoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    a->out = {.shape = {B, dw->output_dim + 1}};
    alloc_register(alloc, &a->out);
}

void* decoder_create_weights(void* self) {
    Decoder* d = (Decoder*)self;
    DecoderWeights* dw = (DecoderWeights*)calloc(1, sizeof(DecoderWeights));
    dw->hidden_dim = d->hidden_dim;
    dw->output_dim = d->output_dim;
    dw->continuous = d->continuous;
    dw->ar = d->ar;
    return dw;
}

Prec decoder_backward(void* w, void* activations, Float grad_logits,
        Float grad_logstd, Float grad_value, cudaStream_t stream) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    int B_TT = a->saved_input.shape[0];
    int od = dw->output_dim, od1 = od + 1;
    assemble_decoder_grad<<<grid_size(B_TT * od1), BLOCK_SIZE, 0, stream>>>(
        a->grad_out.data, grad_logits.data, grad_value.data, B_TT, od, od1);
    // dW // dX: weight grad on side stream, dX on main (needed for residual chain).
    puf_mm_tn_async_after(&a->grad_out, &a->saved_input, &a->wgrad_scratch, stream);
    if (dw->continuous && grad_logstd.data != NULL) {
        sum_rows_to_precision_kernel<<<grid_size(dw->output_dim), BLOCK_SIZE, 0, stream>>>(
            a->logstd_scratch.data, grad_logstd.data, B_TT, dw->output_dim);
    }
    puf_mm_nn(&a->grad_out, &dw->weight, &a->grad_input, stream);
    return a->grad_input;
}

struct MinGRUActivations {
    // Rollout
    Prec* combined;       // (B rollout, 3*T)[num_layers]
    Prec out;             // (B rollout, T)
    Prec next_state;      // (B rollout, T)
    // Training
    Prec* saved_inputs;   // (B, TT, T)[num_layers]
    PrefixScan* scan_bufs;           // [num_layers]
    Prec* combined_bufs;  // (B*TT, 3*T)[num_layers]
    Prec* wgrad_scratch;  // (3*T, T)[num_layers]
    Prec grad_input_buf;  // (B*TT, T)
    Prec grad_next_state; // (B, 1, T)
};

struct MinGRUWeights {
    int hidden, num_layers, horizon;
    Prec* weights;  // [num_layers]
};

Prec mingru_state_layer(Prec& state, int layer, int agent_off, int B) {
    long A = state.shape[1], H = state.shape[2];
    return {
        .data = state.data + ((long)layer * A + agent_off) * H,
        .shape = {B, H},
    };
}

void mingru_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    for (int i = 0; i < m->num_layers; i++) {
        Prec w2d = {
            .data = m->weights[i].data,
            .shape = {3 * m->hidden, m->hidden},
        };
        puf_kaiming_init(&w2d, 1.0f, (*seed)++, stream);
    }
}

void mingru_reg_params(void* w, Allocator* alloc) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    for (int i = 0; i < m->num_layers; i++) {
        m->weights[i] = {.shape = {3 * m->hidden, m->hidden}};
        alloc_register(alloc, &m->weights[i]);
    }
}

void mingru_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B_TT) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int H = m->hidden, TT = m->horizon, B = B_TT / TT;
    a->saved_inputs = (Prec*)calloc(m->num_layers, sizeof(Prec));
    a->scan_bufs = (PrefixScan*)calloc(m->num_layers, sizeof(PrefixScan));
    a->combined_bufs = (Prec*)calloc(m->num_layers, sizeof(Prec));
    a->wgrad_scratch = (Prec*)calloc(m->num_layers, sizeof(Prec));
    a->grad_input_buf = {.shape = {B_TT, H}};
    a->grad_next_state = {.shape = {B, 1, H}};
    alloc_register(acts, &a->grad_input_buf);
    alloc_register(acts, &a->grad_next_state);
    for (int i = 0; i < m->num_layers; i++) {
        a->scan_bufs[i] = {
            .B = B, .T = TT, .H = H,
            .scan_h =           {.shape = {B, TT, H}},
            .out =              {.shape = {B, TT, H}},
            .next_state =       {.shape = {B, 1, H}},
            .grad_combined =    {.shape = {B, TT, 3 * H}},
            .grad_state =       {.shape = {B, 1, H}},
            .grad_input =       {.shape = {B, TT, H}},
        };
        a->saved_inputs[i]  = {.shape = {B, TT, H}};
        a->combined_bufs[i] = {.shape = {B_TT, 3 * H}};
        a->wgrad_scratch[i] = {.shape = {3 * H, H}};
        alloc_register(acts, &a->saved_inputs[i]);
        alloc_register(acts, &a->combined_bufs[i]);
        alloc_register(acts, &a->scan_bufs[i].out);
        alloc_register(acts, &a->scan_bufs[i].next_state);
        alloc_register(acts, &a->scan_bufs[i].scan_h);
        alloc_register(acts, &a->scan_bufs[i].grad_combined);
        alloc_register(acts, &a->scan_bufs[i].grad_state);
        alloc_register(acts, &a->scan_bufs[i].grad_input);
        alloc_register(grads, &a->wgrad_scratch[i]);
    }
}

void mingru_reg_rollout(void* weights, void* activations,
        Allocator* alloc, int B_inf) {
    MinGRUWeights* w = (MinGRUWeights*)weights;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int H = w->hidden;
    a->combined = (Prec*)calloc(w->num_layers, sizeof(Prec));
    for (int i = 0; i < w->num_layers; i++) {
        a->combined[i] = {.shape = {B_inf, 3 * H}};
        alloc_register(alloc, &a->combined[i]);
    }
    a->out = {.shape = {B_inf, H}};
    a->next_state = {.shape = {B_inf, H}};
    alloc_register(alloc, &a->out);
    alloc_register(alloc, &a->next_state);
}

void* mingru_create_weights(void* self) {
    Network* n = (Network*)self;
    MinGRUWeights* mw = (MinGRUWeights*)calloc(1, sizeof(MinGRUWeights));
    mw->hidden = n->hidden;
    mw->num_layers = n->num_layers;
    mw->horizon = n->horizon;
    mw->weights = (Prec*)calloc(n->num_layers, sizeof(Prec));
    return mw;
}

Prec mingru_forward(void* w, Prec x, Prec state,
        void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int B = state.shape[1];
    int H = state.shape[2];
    for (int i = 0; i < m->num_layers; i++) {
        Prec state_i = mingru_state_layer(state, i, 0, B);
        puf_mm(&x, &m->weights[i], &a->combined[i], stream);
        mingru_gate<<<grid_size(B*H), BLOCK_SIZE, 0, stream>>>(
            a->out.data, a->next_state.data,
            a->combined[i].data, state_i.data, x.data, H, B);
        puf_copy(&state_i, &a->next_state, stream);
        x = a->out;
    }
    return x;
}

Prec mingru_forward_train(void* w, Prec x, Prec state, Prec rnn_resets,
        void* activations, int agent_off, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int B = (int)x.shape[0];
    for (int i = 0; i < m->num_layers; i++) {
        puf_copy(&a->saved_inputs[i], &x, stream);
        Prec state_i = mingru_state_layer(state, i, agent_off, B);
        puf_mm(&x, &m->weights[i], &a->combined_bufs[i], stream);
        PrefixScan& scan = a->scan_bufs[i];
        scan.combined_ptr = a->combined_bufs[i].data;
        scan.state_ptr = state_i.data;
        scan.input_ptr = a->saved_inputs[i].data;
        scan.rnn_resets_ptr = rnn_resets.data;
        mingru_scan_forward_seq<<<grid_size(scan.B * scan.H), BLOCK_SIZE, 0, stream>>>(scan);
        x = scan.out;
    }
    return x;
}

__global__ void add_kernel(precision_t* __restrict__ dst,
        const precision_t* __restrict__ src, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
            idx += blockDim.x * gridDim.x) {
        dst[idx] = from_float(to_float(dst[idx]) + to_float(src[idx]));
    }
}

Prec mingru_backward(void* w, Prec grad, void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    for (int i = m->num_layers - 1; i >= 0; i--) {
        PrefixScan& scan = a->scan_bufs[i];
        mingru_scan_backward<<<grid_size(scan.B*scan.H), BLOCK_SIZE, 0, stream>>>(
            scan, grad.data, a->grad_next_state.data);
        // dW on side stream (per-layer scratch); dX on main continues the bwd chain.
        puf_mm_tn_async_after(&scan.grad_combined, &a->saved_inputs[i],
            &a->wgrad_scratch[i], stream);
        puf_mm_nn(&scan.grad_combined, &m->weights[i], &a->grad_input_buf, stream);
        int n = numel(scan.grad_input.shape);
        add_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            a->grad_input_buf.data, scan.grad_input.data, n);
        grad = a->grad_input_buf;
    }
    return grad;
}

struct Arch {
    Encoder encoder;
    Decoder decoder;
    Network network;
};

struct Activations {
    void* encoder;
    void* decoder;
    void* network;
};

struct Weights {
    void* encoder;
    void* decoder;
    void* network;
};

Prec arch_forward(Arch* p, Weights& w, Activations& activations,
        PolicyObs obs, Prec state, cudaStream_t stream) {
    Prec enc_out = p->encoder.forward(
        w.encoder, activations.encoder, obs, stream);
    Prec h = p->network.forward(
        w.network, enc_out, state, activations.network, stream);
    return p->decoder.forward(w.decoder, activations.decoder, h, stream);
}


void arch_backward(Arch* p, Weights& w,
        Activations& activations, Float grad_logits,
        Float grad_logstd, Float grad_value, cudaStream_t stream) {
    int B = grad_logits.shape[0], TT = grad_logits.shape[1];
    Prec grad_h = p->decoder.backward(w.decoder,
        activations.decoder, *puf_squeeze(&grad_logits, 0),
        grad_logstd, *puf_squeeze(&grad_value, 0), stream);
    grad_h = p->network.backward(w.network,
        *puf_unsqueeze(&grad_h, 0, B, TT), activations.network, stream);
    p->encoder.backward(w.encoder, activations.encoder, grad_h, stream);
    puf_dw_join(stream); // sycn dW GEMMs before muon reads weight grads.
}

Activations arch_reg_train(Arch* p, Weights& w,
        Allocator* acts, Allocator* grads, int B_TT) {
    Activations a;
    a.encoder = calloc(1, p->encoder.activation_size);
    a.decoder = calloc(1, p->decoder.activation_size);
    a.network = calloc(1, sizeof(MinGRUActivations));
    p->encoder.reg_train(w.encoder, a.encoder, acts, grads, B_TT);
    p->decoder.reg_train(w.decoder, a.decoder, acts, grads, B_TT);
    p->network.reg_train(w.network, a.network, acts, grads, B_TT);
    return a;
}

Activations arch_reg_rollout(Arch* p, Weights& w,
        Allocator* acts, int B_inf) {
    Activations a;
    a.encoder = calloc(1, p->encoder.activation_size);
    a.decoder = calloc(1, p->decoder.activation_size);
    a.network = calloc(1, sizeof(MinGRUActivations));
    p->encoder.reg_rollout(w.encoder, a.encoder, acts, B_inf);
    p->decoder.reg_rollout(w.decoder, a.decoder, acts, B_inf);
    p->network.reg_rollout(w.network, a.network, acts, B_inf);
    return a;
}

void weights_init(Arch* p, Weights& w,
        uint64_t* seed, cudaStream_t stream) {
    p->encoder.init_weights(w.encoder, seed, stream);
    p->decoder.init_weights(w.decoder, seed, stream);
    p->network.init_weights(w.network, seed, stream);
}

Weights weights_create(Arch* p, Allocator* params) {
    Weights w;
    w.encoder = p->encoder.create_weights(&p->encoder);
    w.decoder = p->decoder.create_weights(&p->decoder);
    w.network = p->network.create_weights(&p->network);
    p->encoder.reg_params(w.encoder, params);
    p->decoder.reg_params(w.decoder, params);
    p->network.reg_params(w.network, params);
    return w;
}

// Custom architectures for specific envs. Not yet
// happy with the API, but we can't narrow it yet
// because fast, general encoder arch is an
// unsolved research problem.
#include "ocean.cu"

// Build an Arch (ops + dims) for a given env. Encoder/decoder algorithms are
// fixed by the env; hidden_size/num_layers/horizon parameterize shape. Arch
// has no heap state so this returns by value; callers store it wherever.
Arch build_arch(const char* env_name, int input_size, int hidden_size,
        int num_layers, int decoder_output_size, bool is_continuous, int horizon,
        bool use_custom_encoder) {
    Encoder encoder = {
        .forward = encoder_forward,
        .backward = encoder_backward,
        .init_weights = encoder_init_weights,
        .reg_params = encoder_reg_params,
        .reg_train = encoder_reg_train,
        .reg_rollout = encoder_reg_rollout,
        .create_weights = encoder_create_weights,
        .in_dim = input_size, .out_dim = hidden_size,
        .activation_size = sizeof(EncoderActivations),
    };
    if (use_custom_encoder) create_custom_encoder(env_name, &encoder);
    Decoder decoder = {
        .forward = decoder_forward,
        .backward = decoder_backward,
        .init_weights = decoder_init_weights,
        .reg_params = decoder_reg_params,
        .reg_train = decoder_reg_train,
        .reg_rollout = decoder_reg_rollout,
        .create_weights = decoder_create_weights,
        .hidden_dim = hidden_size,
        .output_dim = decoder_output_size,
        .continuous = is_continuous,
        .activation_size = sizeof(DecoderActivations),
    };
    create_custom_decoder(env_name, &decoder);
    Network network = {
        .forward = mingru_forward,
        .forward_train = mingru_forward_train,
        .backward = mingru_backward,
        .init_weights = mingru_init_weights,
        .reg_params = mingru_reg_params,
        .reg_train = mingru_reg_train,
        .reg_rollout = mingru_reg_rollout,
        .create_weights = mingru_create_weights,
        .hidden = hidden_size,
        .num_layers = num_layers,
        .horizon = horizon,
    };
    return Arch{
        .encoder = encoder, .decoder = decoder, .network = network,
    };
}

__global__ void muon_sum_sq_partials(float* __restrict__ partials,
        const precision_t* __restrict__ src, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        float v = to_float(src[i]);
        sum += v * v;
    }
    sdata[tid] = sum;
    block_reduce_sum(sdata, &partials[blockIdx.x], tid, blockDim.x, 1);
}

__global__ void muon_sum_sq_reduce(float* __restrict__ out,
        const float* __restrict__ partials, int num_blocks) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = (tid < num_blocks) ? partials[tid] : 0.0f;
    block_reduce_sum(sdata, out, tid, blockDim.x, 1);
}

// Global grad clip by L2, then Nesterov into f32 momentum buffer.
__global__ void muon_clip_nesterov(float* __restrict__ mb,
        precision_t* __restrict__ gc, const float* __restrict__ sum_sq_ptr,
        float max_norm, float eps, float mu, int n) {
    float clip_coef = fminf(max_norm / (sqrtf(*sum_sq_ptr) + eps), 1.0f);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = to_float(gc[idx]) * clip_coef;
        float m = mu * mb[idx] + g;
        mb[idx] = m;
        gc[idx] = from_float(g + mu * m);
    }
}

// x *= 1 / max(sqrt(sum_sq), eps)  — NS input normalize
__global__ void muon_l2_normalize(precision_t* __restrict__ dst,
        const float* __restrict__ sum_sq_ptr, float eps, int n) {
    float inv_norm = 1.0f / fmaxf(sqrtf(*sum_sq_ptr), eps);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(to_float(dst[idx]) * inv_norm);
    }
}


// NorMuon row-wise second moment and normalization. Each row emits its
// pre/post-normalization squared norms so both matrix norms need one reduction.
__global__ void normuon_scale_rows(precision_t* __restrict__ update,
        float* __restrict__ second_moment, float* __restrict__ row_norm_sums,
        float beta2, int rows, int cols) {
    __shared__ float sdata[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float value = to_float(update[row * cols + col]);
        sum += value * value;
    }
    sdata[tid] = sum;
    block_reduce_sum(sdata, &row_norm_sums[row], tid, blockDim.x, 1);
    if (tid == 0) {
        float mean = sdata[0] / (float)cols;
        float moment = beta2 * second_moment[row] + (1.0f - beta2) * mean;
        second_moment[row] = moment;
        sdata[0] = 1.0f / (sqrtf(moment) + 1e-10f);
    }
    __syncthreads();
    float normalized_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        int idx = row * cols + col;
        precision_t value = from_float(to_float(update[idx]) * sdata[0]);
        update[idx] = value;
        float normalized = to_float(value);
        normalized_sum += normalized * normalized;
    }
    sdata[tid] = normalized_sum;
    block_reduce_sum(sdata, &row_norm_sums[rows + row],
        tid, blockDim.x, 1);
}

#ifdef USE_ROCM
#define NORMUON_FULL_MASK 0xffffffffffffffffULL
#else
#define NORMUON_FULL_MASK 0xffffffffU
#endif
#define NORMUON_WARP_SIZE 32
#define NORMUON_ROWS_PER_BLOCK (256 / NORMUON_WARP_SIZE)

// Narrow rows otherwise waste most of a 256-thread block. A logical 32-thread
// warp handles each row; this works on both CUDA warps and ROCm wave64 halves.
__global__ void normuon_scale_rows_narrow(precision_t* __restrict__ update,
        float* __restrict__ second_moment, float* __restrict__ row_norm_sums,
        float beta2, int rows, int cols) {
    int lane = threadIdx.x % NORMUON_WARP_SIZE;
    int warp = threadIdx.x / NORMUON_WARP_SIZE;
    int row = blockIdx.x * NORMUON_ROWS_PER_BLOCK + warp;
    bool active = row < rows;
    float sum = 0.0f;
    for (int col = lane; active && col < cols; col += NORMUON_WARP_SIZE) {
        float value = to_float(update[row * cols + col]);
        sum += value * value;
    }
    for (int offset = NORMUON_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(NORMUON_FULL_MASK, sum, offset,
            NORMUON_WARP_SIZE);
    }
    float inverse = 0.0f;
    if (active && lane == 0) {
        float mean = sum / (float)cols;
        float moment = beta2 * second_moment[row] + (1.0f - beta2) * mean;
        second_moment[row] = moment;
        row_norm_sums[row] = sum;
        inverse = 1.0f / (sqrtf(moment) + 1e-10f);
    }
    inverse = __shfl_sync(NORMUON_FULL_MASK, inverse, 0,
        NORMUON_WARP_SIZE);
    float normalized_sum = 0.0f;
    for (int col = lane; active && col < cols; col += NORMUON_WARP_SIZE) {
        int idx = row * cols + col;
        precision_t value = from_float(to_float(update[idx]) * inverse);
        update[idx] = value;
        float normalized = to_float(value);
        normalized_sum += normalized * normalized;
    }
    for (int offset = NORMUON_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        normalized_sum += __shfl_down_sync(NORMUON_FULL_MASK,
            normalized_sum, offset, NORMUON_WARP_SIZE);
    }
    if (active && lane == 0) {
        row_norm_sums[rows + row] = normalized_sum;
    }
}

__global__ void normuon_reduce_row_norms(float* __restrict__ norms,
        const float* __restrict__ row_norm_sums, int rows) {
    __shared__ float sdata[2 * 256];
    int tid = threadIdx.x;
    float original_sum = 0.0f;
    float normalized_sum = 0.0f;
    for (int row = tid; row < rows; row += blockDim.x) {
        original_sum += row_norm_sums[row];
        normalized_sum += row_norm_sums[rows + row];
    }
    sdata[tid] = original_sum;
    sdata[blockDim.x + tid] = normalized_sum;
    block_reduce_sum(sdata, norms, tid, blockDim.x, 2);
}

// Restore the pre-normalization Frobenius norm, then apply Muon's aspect scale.
__global__ void normuon_store_update(precision_t* __restrict__ dst,
        const precision_t* __restrict__ src,
        const float* __restrict__ original_norm_sq,
        const float* __restrict__ normalized_norm_sq,
        float aspect_scale, int n) {
    float scale = aspect_scale * sqrtf(*original_norm_sq)
        / (sqrtf(*normalized_norm_sq) + 1e-10f);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(scale * to_float(src[idx]));
    }
}

// wb = wb * (1 - lr*wd) - lr * update  (update already scaled; one call for all params)
__global__ void muon_weight_update(float* __restrict__ wb,
        const precision_t* __restrict__ update,
        const float* __restrict__ lr_ptr, float wd, int n) {
    float lr = *lr_ptr;
    float wd_scale = 1.0f - lr * wd;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        wb[idx] = wb[idx] * wd_scale - lr * to_float(update[idx]);
    }
}

constexpr double ns_coeffs[5][3] = {
    {4.0848, -6.8946, 2.9270},
    {3.9505, -6.3029, 2.6377},
    {3.7418, -5.5913, 2.3037},
    {2.8769, -3.1427, 1.2046},
    {2.8366, -3.0525, 1.2012},
};

// NorMuon optimizer: Muon orthogonalization with neuron-wise adaptive scaling.
struct NorMuon {
    double momentum;
    double beta2;
    // Scalars / scratch: raw device ptrs. Tensors: allocator.
    float* lr;
    float* grad_norm;
    float* ns_norm;        // two scalars; second stores post-row-normalization norm
    float* norm_partials;  // 256
    Float mb;              // flat momentum buffer (param-sized)
    Float second_moment;   // one scalar per matrix row
    Float row_norm_sums;   // pre/post-normalization sum per row, reused per matrix
    Prec gram, gram_buf, x_buf;
    Allocator* param_alloc;
};

void normuon_init(NorMuon* m, Allocator* param_alloc,
        double momentum, double beta2, Allocator* alloc) {
    assert(beta2 >= 0.0 && beta2 < 1.0);
    m->momentum = momentum;
    m->beta2 = beta2;
    m->param_alloc = param_alloc;
    cudaMalloc((void**)&m->lr, sizeof(float));
    cudaMalloc((void**)&m->grad_norm, sizeof(float));
    cudaMalloc((void**)&m->ns_norm, 2 * sizeof(float));
    cudaMalloc((void**)&m->norm_partials, 256 * sizeof(float));
    m->mb = {.shape = {param_alloc->total_elems}};
    alloc_register(alloc, &m->mb);
    long max_M = 0, max_N = 0, max_rows = 0, total_rows = 0;
    for (int _i = 0; _i < param_alloc->num_regs; _i++) {
        AllocEntry& e = param_alloc->regs[_i];
        if (ndim(e.shape) >= 2) {
            long R = e.shape[0], C = numel(e.shape) / R;
            max_M = max(max_M, min(R, C));
            max_N = max(max_N, max(R, C));
            max_rows = max(max_rows, R);
            total_rows += R;
        }
    }
    m->second_moment = {.shape = {total_rows}};
    m->row_norm_sums = {.shape = {2 * max_rows}};
    m->gram =     {.shape = {max_M, max_M}};
    m->gram_buf = {.shape = {max_M, max_M}};
    m->x_buf =    {.shape = {max_M, max_N}};
    alloc_register(alloc, &m->second_moment);
    alloc_register(alloc, &m->row_norm_sums);
    alloc_register(alloc, &m->gram);
    alloc_register(alloc, &m->gram_buf);
    alloc_register(alloc, &m->x_buf);
}



void normuon_step(NorMuon* m, Float weights, Prec grads,
        float max_grad_norm, cudaStream_t stream = 0) {
    int n_grad = (int)numel(grads.shape);
    int sum_blocks = min((int)grid_size(n_grad), 256);
    muon_sum_sq_partials<<<sum_blocks, 256, 0, stream>>>(
        m->norm_partials, grads.data, n_grad);
    muon_sum_sq_reduce<<<1, 256, 0, stream>>>(
        m->grad_norm, m->norm_partials, sum_blocks);
    muon_clip_nesterov<<<grid_size(n_grad), BLOCK_SIZE, 0, stream>>>(
        m->mb.data, grads.data, m->grad_norm,
        max_grad_norm, 1e-6f, (float)m->momentum, n_grad);

    // Per-param NS and row adaptation into workspace; write the update to grads.
    // 1D params retain the Nesterov update in-place.
    long offset = 0;
    long row_offset = 0;
    for (int _i = 0; _i < m->param_alloc->num_regs; _i++) {
        AllocEntry& e = m->param_alloc->regs[_i];
        precision_t* gc_ptr = grads.data + offset;
        long ne = numel(e.shape);
        offset += ne;
        if (ndim(e.shape) < 2) {
            continue;
        }

        long R = e.shape[0], C = ne / R;
        float* second_moment = m->second_moment.data + row_offset;
        row_offset += R;
        long M = min(R, C);
        bool tall = R > C;
        Prec x = {.data = gc_ptr, .shape = {R, C}};
        Prec x_buf = {.data = m->x_buf.data, .shape = {R, C}};
        Prec gram = {.data = m->gram.data, .shape = {M, M}};
        Prec gram_buf = {.data = m->gram_buf.data, .shape = {M, M}};

        int nblk = min((int)grid_size(ne), 256);
        muon_sum_sq_partials<<<nblk, 256, 0, stream>>>(
            m->norm_partials, x.data, (int)ne);
        muon_sum_sq_reduce<<<1, 256, 0, stream>>>(
            m->ns_norm, m->norm_partials, nblk);
        muon_l2_normalize<<<grid_size(ne), BLOCK_SIZE, 0, stream>>>(
            x.data, m->ns_norm, 1e-7f, (int)ne);

        // 5 steps land in x_buf. 4 = you break it.
        for (int i = 0; i < 5; ++i) {
            Prec& src = (i % 2 == 0) ? x : x_buf;
            Prec& dst = (i % 2 == 0) ? x_buf : x;
            if (tall) {
                puf_mm_tn(&src, &src, &gram, stream);
            } else {
                puf_mm(&src, &src, &gram, stream);
            }
#ifdef USE_ROCM
            puf_mm_nn_residual(&gram, &gram, &gram, &gram_buf, stream,
                (float)ns_coeffs[i][2], (float)ns_coeffs[i][1]);
            if (tall) {
                puf_mm_nn_residual(&src, &gram_buf, &src, &dst,
                    stream, 1.0f, (float)ns_coeffs[i][0]);
            } else {
                puf_mm_nn_residual(&gram_buf, &src, &src, &dst,
                    stream, 1.0f, (float)ns_coeffs[i][0]);
            }
#else
            puf_copy(&gram_buf, &gram, stream);
            puf_mm_nn(&gram, &gram, &gram_buf, stream,
                (float)ns_coeffs[i][2], (float)ns_coeffs[i][1]);
            puf_copy(&dst, &src, stream);
            if (tall) {
                puf_mm_nn(&src, &gram_buf, &dst,
                    stream, 1.0f, (float)ns_coeffs[i][0]);
            } else {
                puf_mm_nn(&gram_buf, &src, &dst,
                    stream, 1.0f, (float)ns_coeffs[i][0]);
            }
#endif
        }
        float* row_norm_sums = m->row_norm_sums.data;
        if (C <= 2 * NORMUON_WARP_SIZE) {
            int row_blocks = ((int)R + NORMUON_ROWS_PER_BLOCK - 1)
                / NORMUON_ROWS_PER_BLOCK;
            normuon_scale_rows_narrow<<<row_blocks, 256, 0, stream>>>(
                x_buf.data, second_moment, row_norm_sums,
                (float)m->beta2, (int)R, (int)C);
        } else {
            normuon_scale_rows<<<R, 256, 0, stream>>>(
                x_buf.data, second_moment, row_norm_sums,
                (float)m->beta2, (int)R, (int)C);
        }
        normuon_reduce_row_norms<<<1, 256, 0, stream>>>(
            m->ns_norm, row_norm_sums, (int)R);
        float aspect_scale = sqrtf(fmaxf(1.0f, (float)R / (float)C));
        normuon_store_update<<<grid_size(ne), BLOCK_SIZE, 0, stream>>>(
            gc_ptr, x_buf.data, m->ns_norm, m->ns_norm + 1,
            aspect_scale, (int)ne);
    }
    muon_weight_update<<<grid_size(n_grad), BLOCK_SIZE, 0, stream>>>(
        weights.data, grads.data, m->lr, 0.0f, n_grad);
}

// Train layout is (B, T). Views are sliced each mb; scratch is allocated.
struct TrainGraph {
    Prec mb_state;       // view into train_state (L, A, H); read with agent_off
    PolicyObs mb_obs;    // view (B, T, packed bytes or precision values)
    Prec mb_actions;     // view (B, T, num_atns)
    Prec mb_logprobs;    // view (B, T)
    Prec mb_rnn_resets;  // view (B, T), reset before observation[t]
    Prec mb_transition_dones; // view (B, T), done after action[t]
    Prec mb_rewards;     // view (B, T)
    Prec mb_bootstrap_values; // view (B), V(s_H)
    Prec mb_advantages;  // scratch
    Prec mb_values;      // view: frozen rollout V (vf-clip)
    Prec mb_returns;     // view: aliases mb_gae_v after GAE (V+A)
    Prec mb_action_mask; // view (B, T, mask_size)
    Prec mb_imp;         // scratch
    Prec mb_gae_v;       // scratch: live V in, overwritten with returns
};

void register_train_buffers(TrainGraph& bufs, Allocator* alloc, int B, int T) {
    bufs = (TrainGraph){
        .mb_advantages =    {.shape = {B, T}},
        .mb_imp =           {.shape = {B, T}},
        .mb_gae_v =         {.shape = {B, T}},
    };
    alloc_register(alloc, &bufs.mb_advantages);
    alloc_register(alloc, &bufs.mb_imp);
    alloc_register(alloc, &bufs.mb_gae_v);
}
// TODO: test whether these finite/clamp guards improve continuous-control stability
// or just hide bad logits/actions.
__device__ __forceinline__ float finite_or_clamp(float x, float lo, float hi) {
    if (isnan(x)) {
        return 0.0f;
    }
    if (isinf(x)) {
        return x > 0.0f ? hi : lo;
    }
    return fminf(hi, fmaxf(lo, x));
}

__device__ __forceinline__ float safe_continuous_mean(const precision_t* logits, int idx) {
    return finite_or_clamp(to_float(logits[idx]), -1.0e6f, 1.0e6f);
}

__device__ __forceinline__ float safe_continuous_logstd(const precision_t* logstd, int idx) {
    return finite_or_clamp(to_float(logstd[idx]), -20.0f, 2.0f);
}

enum LossIdx {
    LOSS_PG = 0, LOSS_VF = 1, LOSS_ENT = 2, LOSS_TOTAL = 3,
    LOSS_OLD_APPROX_KL = 4, LOSS_APPROX_KL = 5, LOSS_CLIPFRAC = 6,
    LOSS_IMP = 7,
    LOSS_N = 8, NUM_LOSSES = 9,
};

#ifdef PUFFER_NETHACK
#include "../ocean/nethack/nethack_policy.cu"
#endif

// consumed-head gating is a experimental feature currently used by Nethack only.
#ifndef PUFFER_PROVIDES_HEAD_CONSUME_MAP
extern "C" __attribute__((weak)) const signed char* env_head_consume_map(int*, int*);
#endif
static const signed char* g_hc_dev = NULL;
static int g_hc_stride = 0;
static bool g_hc_init = false;

// Eager host->device upload. Call from create_pufferl (never during capture).
static void init_head_consume_map(void) {
    if (g_hc_init) return;
    g_hc_init = true;
    const char* hg = getenv("PUFFER_HEAD_GATING");
    if (!(hg && hg[0] && hg[0] != '0' && env_head_consume_map)) return;
    int nv = 0, na = 0;
    const signed char* host = env_head_consume_map(&nv, &na);
    if (!(host && nv > 0 && na > 0)) return;
    signed char* dev = NULL;
    assert(cudaMalloc(&dev, (size_t)nv * na) == cudaSuccess
            && "head_consume cudaMalloc failed");
    assert(cudaMemcpy(dev, host, (size_t)nv * na, cudaMemcpyHostToDevice) == cudaSuccess
            && "head_consume cudaMemcpy failed");
    g_hc_dev = dev;
    g_hc_stride = na;
}

static const signed char* get_head_consume_dev(int* stride) {
    // Lazy path for non-cudagraph / tests; create_pufferl pre-inits for graphs.
    if (!g_hc_init) init_head_consume_map();
    *stride = g_hc_stride;
    return g_hc_dev;
}

constexpr int PPO_THREADS = 256;

// Per-env from ENV_HEADER (ocean/<env>/<env>.h).
#ifndef NUM_ATNS
#error "ENV_HEADER must #define NUM_ATNS (number of action heads)"
#endif
#ifndef ACT_SIZES
#error "ENV_HEADER must #define ACT_SIZES { ... } (classes per head)"
#endif
// Exact max head width for this env build — discrete logit cache size.
constexpr int ppo_max_head_classes() {
    constexpr int s[] = ACT_SIZES;
    int m = 0;
    for (int i = 0; i < NUM_ATNS; i++) {
        if (s[i] > m) {
            m = s[i];
        }
    }
    return m > 0 ? m : 1;
}
constexpr int PPO_MAX_HEAD_A = ppo_max_head_classes();

// Fused loss function. PPO clipped loss + value + entropy
// buffers + args are quite complex. We do the entire
// forward + backwards pass for the full loss function in one kernel
struct PPOGraphArgs {
    precision_t* imp;
    const precision_t* actions;
    const precision_t* old_logprobs;
    const precision_t* advantages;
    const precision_t* values;
    const precision_t* returns;
};

struct PPOKernelArgs {
    float* grad_logits;
    float* grad_logstd; // For continuous actions
    float* grad_values_pred;
    const precision_t* logits;
    const precision_t* logstd; // Continuous only
    const precision_t* values_pred;
    const int* act_sizes;
    const precision_t* action_mask; // (N, T, mask_size); always present
    const precision_t* condition;
    float* grad_condition;
    int mask_stride;
    int num_atns;
    float clip_coef, vf_clip_coef, vf_coef;
    const float* ent_coef;  // device ptr — host by-value bakes into CUDA graphs
    int T_seq, A_total, N;
    bool is_continuous;
};

struct PPOBufs {
    Float grad_logits, grad_values, grad_logstd;
    Float ppo_partials;
    float* ent_coef;     // device scalar (graphs cannot bake host by-value)
};

void register_ppo_buffers(PPOBufs& bufs, Allocator* alloc, int N, int T, int A_total, bool is_continuous) {
    long total = (long)N * T;
    int ppo_grid = ((int)total + PPO_THREADS - 1) / PPO_THREADS;
    bufs = (PPOBufs){
        .grad_logits = {.shape = {N, T, A_total}},
        .grad_values = {.shape = {N, T, 1}},
        .grad_logstd = {.shape = {N, T, A_total}},
        .ppo_partials = {.shape = {ppo_grid * LOSS_N}},
        .ent_coef = NULL,
    };
    alloc_register(alloc, &bufs.grad_logits);
    alloc_register(alloc, &bufs.grad_values);
    if (is_continuous) {
        alloc_register(alloc, &bufs.grad_logstd);
    }
    alloc_register(alloc, &bufs.ppo_partials);
    cudaMalloc((void**)&bufs.ent_coef, sizeof(float));
}

// Discrete only. mask is always present (env mask or synthetic all-ones).
__device__ __forceinline__ float load_logit_masked(
        const precision_t* __restrict__ logits, int logits_base,
        int logits_offset, int a,
        const precision_t* __restrict__ mask, int mask_base) {
    float l = to_float(logits[logits_base + logits_offset + a]);
    float m = to_float(mask[mask_base + logits_offset + a]);
    if (m == 0.0f) {
        l = -1e4f;
    }
    return l;
}

// Shared by sample_logits + ppo_loss. Fills cache[0..A) (sized PPO_MAX_HEAD_A).
__device__ __forceinline__ float ppo_discrete_logsumexp(
        const precision_t* __restrict__ logits, int logits_base,
        int logits_offset, int A,
        const precision_t* __restrict__ mask, int mask_base,
        float* __restrict__ cache) {
    float max_logit = -INFINITY;
    float sum = 0.0f;
    for (int a = 0; a < A; ++a) {
        float l = load_logit_masked(
            logits, logits_base, logits_offset, a, mask, mask_base);
        cache[a] = l;
        if (l > max_logit) {
            sum *= __expf(max_logit - l);
            max_logit = l;
        }
        sum += __expf(l - max_logit);
    }
    return max_logit + __logf(sum);
}

#ifdef POLICY_MASK_SIZE
__device__ __forceinline__ int mask_byte(
        const precision_t* m, int base, int off) {
    return __float2int_rn(to_float(m[base + off])) & 255;
}

__device__ __forceinline__ uint64_t mask_u64(
        const precision_t* m, int base, int off) {
    uint64_t v = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        v |= (uint64_t)mask_byte(m, base, off + i) << (8 * i);
    return v;
}

__device__ __forceinline__ bool has_primary(int type) {
    return (type >= ACTION_BUY_CARD && type <= ACTION_SWAP_HAND_RIGHT) ||
        type == ACTION_BUY_AND_USE;
}

// Actions are stored in precision_t buffers. A bad/stale action must not be
// allowed to turn into an unchecked index into the AR condition table.
__device__ __forceinline__ int ar_action_value(
        const precision_t* act, int ab, int rel, int limit) {
    int value = (int)to_float(act[ab + rel]);
    return (unsigned)value < (unsigned)limit ? value : 0;
}

__device__ __forceinline__ int selection_base(
        const precision_t* m, int base, int type, int primary) {
    int e = policy_selection_entry(type, primary);
    if (e < 0) return -1;
    int p = base + POLICY_SELECTION_OFFSET +
        e * POLICY_SELECTION_BYTES;
    return mask_byte(m, 0, p + 1) ? p : -1;
}

// Per-agent autoregressive context. The sampled prefix (type, primary, count,
// cards) is fixed per thread; hoisting it out of the per-option loops removes
// the O(prefix) recomputation that dominated the AR heads (heads 3-7 rebuild
// the selected-card mask for every one of the 64 options).
typedef struct ARContext {
    int type;
    int primary;
    int count;
    bool has_primary;
    uint64_t primary_bits;   // mask_u64 at POLICY_PRIMARY_OFFSET + type*8
    int sb;                  // absolute selection-entry offset, or -1
    int min_count;
    int max_count;
    uint64_t allowed;
    uint64_t required;
    uint64_t selected;       // cards selected before the current position
    int prev;                // last selected card before the current position
    uint64_t required_left;  // required & ~selected
    int allowed_left;        // popcount(allowed & ~selected)
    int pos;                 // current card position (head - 3)
    // Set-synergy stats over `selected` (incremental in ar_ctx_advance):
    // per-suit and per-rank counts plus a rank bitset (ace at bits 1 and 14).
    // The AR heads use them to condition scores on flush / pair / straight
    // completion instead of only the previous card.
    const precision_t* m;    // mask base (card-attr lookups)
    int base;                // per-agent mask offset
    int suit_counts[4];
    int rank_counts[15];
    int rmask;
} ARContext;

// Reads act[0..1] (type, primary) and the selection entry from the mask.
// Call again once head 1 has sampled so `primary` is the real value.
__device__ __forceinline__ ARContext ar_ctx_init(
        const precision_t* m, int base, const precision_t* act, int ab) {
    ARContext c;
    c.type = ar_action_value(act, ab, 0, ACTION_TYPE_COUNT);
    c.has_primary = has_primary(c.type);
    c.primary = c.has_primary ? ar_action_value(act, ab, 1, 64) : 0;
    c.primary_bits = c.has_primary ? mask_u64(m, base,
        POLICY_PRIMARY_OFFSET + c.type * POLICY_PRIMARY_BYTES) : 0;
    c.sb = selection_base(m, base, c.type, c.primary);
    if (c.sb >= 0) {
        c.min_count = mask_byte(m, 0, c.sb);
        c.max_count = mask_byte(m, 0, c.sb + 1);
        c.allowed = mask_u64(m, 0, c.sb + 2);
        c.required = mask_u64(m, 0, c.sb + 10);
    } else {
        c.min_count = 0;
        c.max_count = 0;
        c.allowed = 0;
        c.required = 0;
    }
    c.count = 0;
    c.selected = 0;
    c.prev = -1;
    c.required_left = c.required;
    c.allowed_left = __popcll(c.allowed);
    c.pos = 0;
    c.m = m;
    c.base = base;
    for (int i = 0; i < 4; ++i) c.suit_counts[i] = 0;
    for (int i = 0; i < 15; ++i) c.rank_counts[i] = 0;
    c.rmask = 0;
    return c;
}

// Read the sampled count from act[2] (after head 2 runs).
__device__ __forceinline__ void ar_ctx_set_count(
        ARContext* c, const precision_t* act, int ab) {
    c->count = ar_action_value(act, ab, 2, 6);
}

// Per-option card attribute ((suit << 4) | rank) from the env-written mask
// table. Zero for out-of-range options (masked illegal anyway).
__device__ __forceinline__ int ar_opt_attr(const ARContext* c, int opt) {
    if (opt < 0 || opt >= POLICY_CARD_ATTR_BYTES) return 0;
    return mask_byte(c->m, c->base, POLICY_CARD_ATTR_OFFSET + opt);
}

// Longest consecutive run of set rank bits containing position p (1..14).
// Ace dual is handled by the caller trying both p=1 and p=14.
__device__ __forceinline__ int ar_run_len(int rmask, int p) {
    int left = 0, right = 0;
    while (rmask & (1 << (p + right + 1))) right++;
    while ((p - left - 1) >= 1 && (rmask & (1 << (p - left - 1)))) left++;
    return left + right + 1;
}

// Straight-completion signal: longest consecutive run through opt's rank
// among the selected cards plus opt itself.
__device__ __forceinline__ int ar_run_through(const ARContext* c, int opt) {
    int rank = ar_opt_attr(c, opt) & POLICY_CARD_ATTR_RANK_MASK;
    if (rank < 1 || rank > 14) return 1;
    int bit = (rank == 14) ? ((1 << 1) | (1 << 14)) : (1 << rank);
    int rmask = c->rmask | bit;
    int best = ar_run_len(rmask, rank == 14 ? 1 : rank);
    if (rank == 14) {
        int hi = ar_run_len(rmask, 14);
        if (hi > best) best = hi;
    }
    return best;
}

// Advance the card-selection prefix by one position (call after heads 3..6).
__device__ __forceinline__ void ar_ctx_advance(
        ARContext* c, const precision_t* act, int ab) {
    if (c->pos < MAX_SELECTION) {
        int card = ar_action_value(act, ab, 3 + c->pos, 64);
        c->selected |= UINT64_C(1) << card;
        c->prev = card;
        c->pos++;
        c->required_left = c->required & ~c->selected;
        c->allowed_left = __popcll(c->allowed & ~c->selected);
        int attr = ar_opt_attr(c, card);
        int suit = attr >> POLICY_CARD_ATTR_SUIT_SHIFT;
        int rank = attr & POLICY_CARD_ATTR_RANK_MASK;
        if (suit >= 0 && suit < 4) c->suit_counts[suit]++;
        if (rank >= 1 && rank <= 14) {
            c->rank_counts[rank]++;
            c->rmask |= (rank == 14) ? (1 << 1) | (1 << 14) : (1 << rank);
        }
    }
}

__device__ __forceinline__ bool ar_head_active(const ARContext* c, int head) {
    if (head == 0) return true;
    if (head == 1) return c->has_primary;
    if (head == 2) return c->sb >= 0;
    if (head >= 3 && head <= 7)
        return c->sb >= 0 && head - 3 < c->count;
    return false;
}

__device__ __forceinline__ bool ar_option_legal(
        const ARContext* c, const precision_t* m, int base, int head, int opt) {
    if (head == 0) return mask_byte(m, base, opt) != 0;
    if (head == 1) {
        if (!c->has_primary) return opt == 0;
        return (c->primary_bits & (UINT64_C(1) << opt)) != 0;
    }
    if (head == 2) {
        if (c->sb < 0) return opt == 0;
        return opt >= c->min_count && opt <= c->max_count;
    }
    if (head >= 3 && head <= 7) {
        int pos = head - 3;
        if (c->sb < 0 || pos >= c->count) return opt == 0;
        if (opt <= c->prev || !(c->allowed & (UINT64_C(1) << opt))) return false;
        int left = c->count - pos - 1;
        uint64_t lower = opt ? (UINT64_C(1) << opt) - 1 : 0;
        uint64_t pending = c->required_left;
        if (pending & lower) return false;
        pending &= ~(UINT64_C(1) << opt);
        uint64_t greater = opt == 63 ? 0 :
            ~((UINT64_C(1) << (opt + 1)) - 1);
        if (__popcll(pending & greater) > left) return false;
        return __popcll(c->allowed & ~c->selected & greater) >= left;
    }
    return opt == 0;
}

__device__ __forceinline__ int ar_cond_idx(
        const ARContext* c, int head, int opt, int part) {
    if (head == 1)
        return AR_TYPE_PRIMARY_OFFSET + c->type * 64 + opt;
    if (head == 2)
        return part == 0
            ? AR_TYPE_COUNT_OFFSET + c->type * 6 + opt
            : AR_PRIMARY_COUNT_OFFSET + c->primary * 6 + opt;
    if (head >= 3 && head <= 7) {
        int pos = head - 3;
        if (part == 0) return AR_TYPE_CARD_OFFSET +
            (c->type * 5 + pos) * 64 + opt;
        if (part == 1) return AR_PRIMARY_CARD_OFFSET +
            (c->primary * 5 + pos) * 64 + opt;
        if (part == 2) return AR_COUNT_CARD_OFFSET +
            (c->count * 5 + pos) * 64 + opt;
        if (part == 3 && pos > 0)
            return AR_PREVIOUS_CARD_OFFSET +
                (c->prev * 4 + pos - 1) * 64 + opt;
        if (part == 4) {
            int suit = ar_opt_attr(c, opt) >> POLICY_CARD_ATTR_SUIT_SHIFT;
            int cnt = suit >= 0 && suit < 4 ? c->suit_counts[suit] : 0;
            if (cnt > 4) cnt = 4;
            return AR_SUIT_CARD_OFFSET + (cnt * 5 + pos) * 64 + opt;
        }
        if (part == 5) {
            int rank = ar_opt_attr(c, opt) & POLICY_CARD_ATTR_RANK_MASK;
            int cnt = rank >= 1 && rank <= 14 ? c->rank_counts[rank] : 0;
            if (cnt > 4) cnt = 4;
            return AR_RANK_CARD_OFFSET + (cnt * 5 + pos) * 64 + opt;
        }
        if (part == 6) {
            int run = ar_run_through(c, opt);
            if (run > 5) run = 5;
            return AR_RUN_CARD_OFFSET + ((run - 1) * 5 + pos) * 64 + opt;
        }
    }
    return -1;
}

__device__ __forceinline__ float ar_cond_bias(
        const ARContext* c, const precision_t* cond, int head, int opt) {
    if (!cond || head == 0) return 0.0f;
    int n = head == 1 ? 1 : head == 2 ? 2 : 7;
    float v = 0.0f;
    for (int i = 0; i < n; ++i) {
        int j = ar_cond_idx(c, head, opt, i);
        if (j >= 0) v += to_float(cond[j]);
    }
    return v;
}

__device__ __forceinline__ void ar_cond_add(
        float* g, const ARContext* c, int head, int opt, float v) {
    if (!g || v == 0.0f || head == 0) return;
    int n = head == 1 ? 1 : head == 2 ? 2 : 7;
    for (int i = 0; i < n; ++i) {
        int j = ar_cond_idx(c, head, opt, i);
        if (j >= 0) atomicAdd(g + j, v);
    }
}

// Inline helpers for ppo_loss. The actions are fixed per thread there, so a
// good compiler hoists the loop-invariant prefix; the sampled-actions variant
// in sample_logits needs the explicit ARContext (heads write actions between
// reads, which defeats hoisting).
__device__ __forceinline__ bool head_active(
        const precision_t* m, int base, const precision_t* act,
        int ab, int head) {
    if (head == 0) return true;
    int type = ar_action_value(act, ab, 0, ACTION_TYPE_COUNT);
    if (head == 1) return has_primary(type);
    int primary = has_primary(type) ? ar_action_value(act, ab, 1, 64) : 0;
    int sb = selection_base(m, base, type, primary);
    if (head == 2) return sb >= 0;
    if (head >= 3 && head <= 7)
        return sb >= 0 && head - 3 < ar_action_value(act, ab, 2, 6);
    return false;
}

__device__ __forceinline__ bool option_legal(
        const precision_t* m, int base, const precision_t* act,
        int ab, int head, int opt) {
    if (head == 0) return mask_byte(m, base, opt) != 0;
    int type = ar_action_value(act, ab, 0, ACTION_TYPE_COUNT);
    int primary = has_primary(type) ? ar_action_value(act, ab, 1, 64) : 0;
    if (head == 1) {
        if (!has_primary(type)) return opt == 0;
        uint64_t bits = mask_u64(m, base,
            POLICY_PRIMARY_OFFSET + type * POLICY_PRIMARY_BYTES);
        return (bits & (UINT64_C(1) << opt)) != 0;
    }
    int sb = selection_base(m, base, type, primary);
    if (head == 2) {
        if (sb < 0) return opt == 0;
        return opt >= mask_byte(m, 0, sb) && opt <= mask_byte(m, 0, sb + 1);
    }
    if (head >= 3 && head <= 7) {
        int count = ar_action_value(act, ab, 2, 6);
        int pos = head - 3;
        if (sb < 0 || pos >= count) return opt == 0;
        uint64_t allowed = mask_u64(m, 0, sb + 2);
        uint64_t required = mask_u64(m, 0, sb + 10);
        uint64_t selected = 0;
        int prev = -1;
        for (int i = 0; i < pos; ++i) {
            int card = ar_action_value(act, ab, 3 + i, 64);
            selected |= UINT64_C(1) << card;
            prev = card;
        }
        if (opt <= prev || !(allowed & (UINT64_C(1) << opt))) return false;
        int left = count - pos - 1;
        uint64_t lower = opt ? (UINT64_C(1) << opt) - 1 : 0;
        uint64_t pending = required & ~selected;
        if (pending & lower) return false;
        pending &= ~(UINT64_C(1) << opt);
        uint64_t greater = opt == 63 ? 0 :
            ~((UINT64_C(1) << (opt + 1)) - 1);
        if (__popcll(pending & greater) > left) return false;
        return __popcll(allowed & ~selected & greater) >= left;
    }
    return opt == 0;
}

__device__ __forceinline__ int cond_idx(
        const precision_t* act, int ab, int head, int opt, int part) {
    int type = ar_action_value(act, ab, 0, ACTION_TYPE_COUNT);
    int primary = has_primary(type) ? ar_action_value(act, ab, 1, 64) : 0;
    if (head == 1)
        return AR_TYPE_PRIMARY_OFFSET + type * 64 + opt;
    if (head == 2)
        return part == 0
            ? AR_TYPE_COUNT_OFFSET + type * 6 + opt
            : AR_PRIMARY_COUNT_OFFSET + primary * 6 + opt;
    if (head >= 3 && head <= 7) {
        int pos = head - 3;
        int count = ar_action_value(act, ab, 2, 6);
        if (part == 0) return AR_TYPE_CARD_OFFSET +
            (type * 5 + pos) * 64 + opt;
        if (part == 1) return AR_PRIMARY_CARD_OFFSET +
            (primary * 5 + pos) * 64 + opt;
        if (part == 2) return AR_COUNT_CARD_OFFSET +
            (count * 5 + pos) * 64 + opt;
        if (pos > 0) {
            int prev = ar_action_value(act, ab, 2 + pos, 64);
            return AR_PREVIOUS_CARD_OFFSET +
                (prev * 4 + pos - 1) * 64 + opt;
        }
    }
    return -1;
}

__device__ __forceinline__ float cond_bias(
        const precision_t* c, const precision_t* act,
        int ab, int head, int opt) {
    if (!c || head == 0) return 0.0f;
    int n = head == 1 ? 1 : head == 2 ? 2 : 7;
    float v = 0.0f;
    for (int i = 0; i < n; ++i) {
        int j = cond_idx(act, ab, head, opt, i);
        if (j >= 0) v += to_float(c[j]);
    }
    return v;
}

__device__ __forceinline__ void cond_add(
        float* g, const precision_t* act, int ab,
        int head, int opt, float v) {
    if (!g || v == 0.0f || head == 0) return;
    int n = head == 1 ? 1 : head == 2 ? 2 : 7;
    for (int i = 0; i < n; ++i) {
        int j = cond_idx(act, ab, head, opt, i);
        if (j >= 0) atomicAdd(g + j, v);
    }
}
#endif

__device__ __forceinline__ void ppo_continuous_head(
        float mean, float log_std, float action,
        float* out_logp, float* out_entropy) {
    constexpr float HALF_LOG_2PI = 0.9189385332046727f;
    constexpr float HALF_1_PLUS_LOG_2PI = 1.4189385332046727f;
    float std = __expf(log_std);
    float normalized = (action - mean) / std;
    *out_logp = -0.5f * normalized * normalized - HALF_LOG_2PI - log_std;
    *out_entropy = HALF_1_PLUS_LOG_2PI + log_std;
}

// After decoder GEMM, before GAE. One logit walk: live V, ρ, logp=logit-lse.
// logp is written into grad_logits (overwritten with grads after GAE). For
// POLICY_MASK_SIZE envs the walk applies the AR condition bias and legality
// mask, matching the sampled distribution.
__global__ void cache_imp_and_v(
        Prec dec_out,
        const precision_t* __restrict__ actions,
        const precision_t* __restrict__ old_logprobs,
        const precision_t* __restrict__ action_mask,
        Prec logstd,
        const int* __restrict__ act_sizes,
        const precision_t* __restrict__ condition,
        int mask_stride,
        precision_t* __restrict__ imp_out,
        precision_t* __restrict__ value_out,
        float* __restrict__ logps,
        float* __restrict__ new_lp_out) {
    int NT = (int)dec_out.shape[0] * (int)dec_out.shape[1];
    int fused_cols = (int)dec_out.shape[2];
    int A_total = fused_cols - 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NT) {
        return;
    }
    int logits_base = idx * fused_cols;
    int at_base = idx * A_total;
    int mask_base = idx * mask_stride;
    int action_base = idx * NUM_ATNS;
    precision_t* logits = dec_out.data;
    value_out[idx] = logits[logits_base + A_total];

    float new_lp = 0.0f;
    if (logstd.data) {
        for (int h = 0; h < NUM_ATNS; ++h) {
            float lp, ent;
            ppo_continuous_head(
                safe_continuous_mean(logits, logits_base + h),
                safe_continuous_logstd(logstd.data, h),
                to_float(actions[idx * NUM_ATNS + h]), &lp, &ent);
            new_lp += lp;
        }
    } else {
#ifdef PUFFER_NETHACK
        int verb = (int)actions[idx * NUM_ATNS];
#endif
#ifdef POLICY_MASK_SIZE
        ARContext c = ar_ctx_init(action_mask, mask_base, actions, action_base);
        ar_ctx_set_count(&c, actions, action_base);
#endif
        int logits_offset = 0;
        for (int h = 0; h < NUM_ATNS; ++h) {
            int A = act_sizes[h];
#ifdef PUFFER_NETHACK
            if (!nethack_head_used(verb, h)) {
                logits_offset += A;
                continue;
            }
#endif
#ifdef POLICY_MASK_SIZE
            if (!ar_head_active(&c, h)) {
                logits_offset += A;
                continue;
            }
            int act = ar_action_value(actions, action_base, h, A);
            // Per-option condition bias cached per head: the three passes
            // below would otherwise recompute the (heavier) set-synergy
            // parts for every option.
            float bias_cache[PPO_MAX_HEAD_A];
            for (int a = 0; a < A; ++a)
                bias_cache[a] = ar_cond_bias(&c, condition, h, a);
            float max_l = -INFINITY;
            for (int a = 0; a < A; ++a) {
                float l = ar_option_legal(&c, action_mask, mask_base, h, a)
                    ? to_float(logits[logits_base + logits_offset + a]) +
                        bias_cache[a]
                    : -1e4f;
                if (l > max_l) max_l = l;
            }
            float sum = 0.0f, act_l = 0.0f;
            for (int a = 0; a < A; ++a) {
                float l = ar_option_legal(&c, action_mask, mask_base, h, a)
                    ? to_float(logits[logits_base + logits_offset + a]) +
                        bias_cache[a]
                    : -1e4f;
                float e = __expf(l - max_l);
                sum += e;
                if (a == act) act_l = l;
            }
            float lse = max_l + __logf(sum);
            for (int a = 0; a < A; ++a) {
                float l = ar_option_legal(&c, action_mask, mask_base, h, a)
                    ? to_float(logits[logits_base + logits_offset + a]) +
                        bias_cache[a]
                    : -1e4f;
                logps[at_base + logits_offset + a] = l - lse;
            }
            new_lp += act_l - lse;
            if (h >= 3 && h <= 6) {
                ar_ctx_advance(&c, actions, action_base);
            }
#else
            float cache[PPO_MAX_HEAD_A];
            float lse = ppo_discrete_logsumexp(
                logits, logits_base, logits_offset, A,
                action_mask, mask_base, cache);
            for (int a = 0; a < A; ++a) {
                logps[at_base + logits_offset + a] = cache[a] - lse;
            }
#ifdef PUFFER_NETHACK
            int used = nethack_head_used(verb, h);
            if (used) {
                int act = (int)actions[idx * NUM_ATNS + h];
                if (h == 0) {
                    float mix = 1.0f;
                    new_lp += nethack_verb_train_logp(
                        action_mask + mask_base + logits_offset, A,
                        cache[act] - lse, &mix);
                } else {
                    new_lp += cache[act] - lse;
                }
            }
#else
            new_lp += cache[(int)actions[idx * NUM_ATNS + h]] - lse;
#endif
#endif
            logits_offset += A;
        }
    }
    new_lp_out[idx] = new_lp;
    imp_out[idx] = from_float(__expf(new_lp - to_float(old_logprobs[idx])));
}

Prec arch_forward_train(Arch* p, Weights& w,
        Activations& activations, PolicyObs x,
        Prec state, Prec rnn_resets, int agent_off,
        TrainGraph& g, Prec logstd, Prec condition, int* act_sizes,
        float* logps, float* new_lp, cudaStream_t stream) {
    int B = x.shape[0], TT = x.shape[1];
    Prec h = p->encoder.forward(w.encoder,
        activations.encoder, *puf_squeeze(&x, 0), stream);
    h = p->network.forward_train(w.network, *puf_unsqueeze(&h, 0, B, TT),
        state, rnn_resets, activations.network, agent_off, stream);
    Prec dec_out = p->decoder.forward(
        w.decoder, activations.decoder, *puf_squeeze(&h, 0), stream);
    Prec dec = *puf_unsqueeze(&dec_out, 0, B, TT);
    cache_imp_and_v<<<grid_size(B * TT), BLOCK_SIZE, 0, stream>>>(
        dec, g.mb_actions.data, g.mb_logprobs.data, g.mb_action_mask.data,
        logstd, act_sizes, condition.data, (int)g.mb_action_mask.shape[2],
        g.mb_imp.data, g.mb_gae_v.data, logps, new_lp);
    return dec;
}

__global__ void ppo_loss_compute(
        float* __restrict__ ppo_partials,
        PPOKernelArgs a, PPOGraphArgs g) {
    int tid = threadIdx.x;
    int total_elements = a.N * a.T_seq;
    float inv_NT = 1.0f / float(total_elements);

    __shared__ float block_losses[LOSS_N][PPO_THREADS];
    for (int c = 0; c < LOSS_N; c++) {
        block_losses[c][tid] = 0.0f;
    }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int nt = idx;
        int logits_base = nt * (a.A_total + 1);
        int at_base = nt * a.A_total;
        int mask_base = nt * a.mask_stride;
        int action_base = nt * a.num_atns;

        float adv_for_pg = to_float(g.advantages[nt]);
        float val = to_float(g.values[nt]);
        float ret = to_float(g.returns[nt]);
        float val_pred = to_float(a.values_pred[logits_base]);
        float ent_coef = *a.ent_coef;
        float d_entropy_term = inv_NT * (-ent_coef);
        float logratio = a.grad_values_pred[nt] - to_float(g.old_logprobs[nt]);
        float ratio = __expf(logratio);

        // Value loss + gradient: 0.5 * max((v-r)^2, (v_clip-r)^2).
        // When clipped term wins, v_clip is constant in val_pred -> grad 0.
        float v_error = val_pred - val;
        float v_clipped = val + fmaxf(-a.vf_clip_coef, fminf(a.vf_clip_coef, v_error));
        float v_loss_unclipped = (val_pred - ret) * (val_pred - ret);
        float v_loss_clipped = (v_clipped - ret) * (v_clipped - ret);
        float v_loss = 0.5f * fmaxf(v_loss_unclipped, v_loss_clipped);
        float d_val_pred = (v_loss_clipped > v_loss_unclipped) ? 0.0f : (val_pred - ret);
        a.grad_values_pred[nt] = inv_NT * a.vf_coef * d_val_pred;
        float clip_lo = 1.0f - a.clip_coef;
        float clip_hi = 1.0f + a.clip_coef;
        float ratio_clipped = fmaxf(clip_lo, fminf(clip_hi, ratio));
        float wa = -adv_for_pg;
        float pg_loss1 = wa * ratio;
        float pg_loss2 = wa * ratio_clipped;
        float pg_loss = fmaxf(pg_loss1, pg_loss2);
        float d_ratio = wa * inv_NT;
        if (pg_loss2 > pg_loss1 && (ratio <= clip_lo || ratio >= clip_hi)) {
            d_ratio = 0.0f;
        }
        float d_new_logp = d_ratio * ratio;
        float total_entropy = 0.0f;

        if (a.is_continuous) {
            for (int h = 0; h < a.num_atns; ++h) {
                float mean = safe_continuous_mean(a.logits, logits_base + h);
                float c_logstd = safe_continuous_logstd(a.logstd, h);
                float c_action = finite_or_clamp(
                    g.actions[nt * a.num_atns + h], -1.0e6f, 1.0e6f);
                float lp, ent;
                ppo_continuous_head(mean, c_logstd, c_action, &lp, &ent);
                total_entropy += ent;
                float std = __expf(c_logstd);
                float var = std * std;
                float diff = c_action - mean;
                a.grad_logits[at_base + h] = d_new_logp * diff / var;
                a.grad_logstd[nt * a.num_atns + h] =
                    d_new_logp * (diff * diff / var - 1.0f) + d_entropy_term;
            }
        } else {
#ifdef PUFFER_NETHACK
            int verb = (int)g.actions[nt * a.num_atns];
            float verb_mix_scale = 1.0f;
#endif
#ifdef POLICY_MASK_SIZE
            // Balatro AR heads: context hoisted once per row; heads 0..7 are
            // active per the sampled prefix, head 8 is dead (never active).
            // Cached logps already carry the AR condition bias (see
            // cache_imp_and_v); here we only gate and add the condition grad.
            ARContext c = ar_ctx_init(
                a.action_mask, mask_base, g.actions, action_base);
            ar_ctx_set_count(&c, g.actions, action_base);
#endif
            int logits_offset = 0;
            for (int h = 0; h < a.num_atns; ++h) {
                int A = a.act_sizes[h];
#ifdef PUFFER_NETHACK
                if (!nethack_head_used(verb, h)) {
                    for (int j = 0; j < A; ++j) {
                        a.grad_logits[at_base + logits_offset + j] = 0.0f;
                    }
                    logits_offset += A;
                    continue;
                }
#endif
#ifdef POLICY_MASK_SIZE
                if (!ar_head_active(&c, h)) {
                    for (int j = 0; j < A; ++j) {
                        a.grad_logits[at_base + logits_offset + j] = 0.0f;
                    }
                    logits_offset += A;
                    continue;
                }
                int raw_act = (int)to_float(g.actions[action_base + h]);
                int act = (unsigned)raw_act < (unsigned)A ? raw_act : 0;
#else
                int act = (int)g.actions[nt * a.num_atns + h];
#endif
                float ent = 0.0f;
                for (int j = 0; j < A; ++j) {
                    float logp = a.grad_logits[at_base + logits_offset + j];
                    ent -= __expf(logp) * logp;
                }
                total_entropy += ent;
#ifdef PUFFER_NETHACK
                float d_logp = d_new_logp;
                if (h == 0) {
                    nethack_verb_train_logp(
                        a.action_mask + at_base + logits_offset, A,
                        a.grad_logits[at_base + act],
                        &verb_mix_scale);
                    d_logp *= verb_mix_scale;
                }
#else
                float d_logp = d_new_logp;
#endif
                for (int j = 0; j < A; ++j) {
                    float logp = a.grad_logits[at_base + logits_offset + j];
                    float p = __expf(logp);
                    a.grad_logits[at_base + logits_offset + j] =
                        ((j == act ? 1.0f : 0.0f) - p) * d_logp
                        + d_entropy_term * p * (-ent - logp);
#ifdef POLICY_MASK_SIZE
                    if (ar_option_legal(&c, a.action_mask, mask_base, h, j)) {
                        ar_cond_add(
                            a.grad_condition + (nt & (COND_STRIPE_COUNT - 1))
                                * AR_CONDITION_SIZE,
                            &c, h, j,
                            a.grad_logits[at_base + logits_offset + j]);
                    }
#endif
                }
#ifdef POLICY_MASK_SIZE
                // Advance the card-selection prefix for the next AR head.
                if (h >= 3 && h <= 6) {
                    ar_ctx_advance(&c, g.actions, action_base);
                }
#endif
                logits_offset += A;
            }
        }

        float thread_loss = (pg_loss + a.vf_coef * v_loss
            - ent_coef * total_entropy) * inv_NT;

        block_losses[LOSS_PG][tid] = pg_loss * inv_NT;
        block_losses[LOSS_VF][tid] = v_loss * inv_NT;
        block_losses[LOSS_ENT][tid] = total_entropy * inv_NT;
        block_losses[LOSS_TOTAL][tid] = thread_loss;
        block_losses[LOSS_OLD_APPROX_KL][tid] = (-logratio) * inv_NT;
        block_losses[LOSS_APPROX_KL][tid] = ((ratio - 1.0f) - logratio) * inv_NT;
        block_losses[LOSS_CLIPFRAC][tid] =
            (fabsf(ratio - 1.0f) > a.clip_coef ? 1.0f : 0.0f) * inv_NT;
        block_losses[LOSS_IMP][tid] = ratio * inv_NT;
    }

    block_reduce_sum(&block_losses[0][0], &ppo_partials[blockIdx.x * LOSS_N],
        tid, PPO_THREADS, LOSS_N);
}


// Deterministic reduction of per-block PPO loss partials + count increment
__global__ void ppo_loss_reduce(
        float* __restrict__ losses_acc,
        const float* __restrict__ partials,
        int num_blocks) {
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int b = 0; b < num_blocks; b++) {
        sum += partials[b * LOSS_N + tid];
    }
    losses_acc[tid] += sum;
    if (tid == 0) {
        losses_acc[LOSS_N] += 1.0f;
    }
}

__global__ void puf_float_to_precision_kernel(
        precision_t* __restrict__ dst, const float* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = from_float(src[idx]);
}

// Sum the COND_STRIPE_COUNT row-striped condition-gradient staging copies.
__global__ void cond_parts_sum(
        float* __restrict__ dst, const float* __restrict__ parts,
        int stripes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float sum = 0.0f;
    for (int k = 0; k < stripes; ++k) sum += parts[k * n + idx];
    dst[idx] = sum;
}


void ppo_loss_fwd_bwd(
        Prec& dec_out,    // (N, T, fused_cols) - fused logits+value from decoder
        Prec& logstd,     // continuous logstd or empty
        TrainGraph& graph,
        int* act_sizes, float* losses_acc,
        float clip_coef, float vf_clip_coef, float vf_coef, const float* ent_coef,
        PPOBufs& bufs, bool is_continuous,
        Prec condition, Float condition_accum, Prec condition_grad,
        Float condition_parts, cudaStream_t stream) {
    int N = dec_out.shape[0], T = dec_out.shape[1], fused_cols = dec_out.shape[2];
    int A_total = fused_cols - 1;  // last column is value
    int total = N * T;
    int ppo_grid = (total + PPO_THREADS - 1) / PPO_THREADS;
    if (condition_accum.data) {
        cudaMemsetAsync(condition_accum.data, 0,
            numel(condition_accum.shape) * sizeof(float), stream);
    }
    if (condition_parts.data) {
        cudaMemsetAsync(condition_parts.data, 0,
            numel(condition_parts.shape) * sizeof(float), stream);
    }

    PPOGraphArgs graph_args = {
        .imp = graph.mb_imp.data,
        .actions = graph.mb_actions.data,
        .old_logprobs = graph.mb_logprobs.data,
        .advantages = graph.mb_advantages.data,
        .values = graph.mb_values.data,
        .returns = graph.mb_returns.data,
    };

    PPOKernelArgs args = {
        .grad_logits = bufs.grad_logits.data,
        .grad_logstd = is_continuous ? bufs.grad_logstd.data : NULL,
        .grad_values_pred = bufs.grad_values.data,
        .logits = dec_out.data,
        .logstd = is_continuous ? logstd.data : NULL,
        .values_pred = dec_out.data + A_total,
        .act_sizes = act_sizes,
        .action_mask = graph.mb_action_mask.data,
        .condition = condition.data,
        .grad_condition = condition_parts.data,
        .mask_stride = (int)graph.mb_action_mask.shape[2],
        .num_atns = NUM_ATNS,
        .clip_coef = clip_coef, .vf_clip_coef = vf_clip_coef,
        .vf_coef = vf_coef,
        .ent_coef = ent_coef,
        .T_seq = T, .A_total = A_total, .N = N,
        .is_continuous = is_continuous,
    };
    ppo_loss_compute<<<ppo_grid, PPO_THREADS, 0, stream>>>(
            bufs.ppo_partials.data, args, graph_args);
    if (condition_parts.data) {
        // Sum the row-striped condition-gradient staging copies (contention
        // reduced by spreading atomics across COND_STRIPE_COUNT tensors).
        int cn = (int)numel(condition_accum.shape);
        cond_parts_sum<<<grid_size(cn), BLOCK_SIZE, 0, stream>>>(
            condition_accum.data, condition_parts.data, COND_STRIPE_COUNT, cn);
    }
    if (condition_accum.data) {
        int n = numel(condition_accum.shape);
        puf_float_to_precision_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            condition_grad.data, condition_accum.data, n);
    }
    ppo_loss_reduce<<<1, LOSS_N, 0, stream>>>(
        losses_acc, bufs.ppo_partials.data, ppo_grid);
}

// Puffer advantage (GAE + V-trace ρ/c clips). One thread per row.
// 16B segment I/O (H % ADV_VEC_WIDTH == 0). Bench: tests/bench_puff_advantage.cu
constexpr int ADV_VEC_WIDTH = 16 / (int)sizeof(precision_t);

// Wide load/store of one ADV_VEC_WIDTH segment into float regs.
__device__ __forceinline__ void adv_ld(const precision_t* p, float* o) {
    if constexpr (sizeof(precision_t) == sizeof(float)) {
        float4 v = *(const float4*)p;
        o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w;
    } else {
        uint4 u = *(const uint4*)p;
        auto* b = (const __nv_bfloat16*)&u;
        #pragma unroll
        for (int i = 0; i < 8; i++) o[i] = __bfloat162float(b[i]);
    }
}
__device__ __forceinline__ void adv_st(precision_t* p, const float* o) {
    if constexpr (sizeof(precision_t) == sizeof(float)) {
        *(float4*)p = make_float4(o[0], o[1], o[2], o[3]);
    } else {
        __nv_bfloat16 b[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) b[i] = __float2bfloat16(o[i]);
        *(uint4*)p = *(const uint4*)b;
    }
}

__global__ void puff_advantage(const precision_t* values,
        const precision_t* rewards, const precision_t* dones,
        const precision_t* bootstrap_values,
        const precision_t* importance, precision_t* advantages,
        precision_t* returns,
        float gamma, float lambda, float rho_clip, float c_clip,
        int num_steps, int horizon) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_steps) return;
    int off = row * horizon;
    float lastlam = 0.f;
    float next_v = to_float(bootstrap_values[row]);

    for (int seg = horizon / ADV_VEC_WIDTH - 1; seg >= 0; seg--) {
        int base = off + seg * ADV_VEC_WIDTH;
        float v[ADV_VEC_WIDTH], r[ADV_VEC_WIDTH], d[ADV_VEC_WIDTH], imp[ADV_VEC_WIDTH];
        float adv[ADV_VEC_WIDTH] = {};
        float ret[ADV_VEC_WIDTH];
        adv_ld(values + base, v);
        adv_ld(rewards + base, r);
        adv_ld(dones + base, d);
        if (importance) {
            adv_ld(importance + base, imp);
        } else {
            #pragma unroll
            for (int i = 0; i < ADV_VEC_WIDTH; i++) {
                imp[i] = 1.f;
            }
        }
        #pragma unroll
        for (int i = ADV_VEC_WIDTH - 1; i >= 0; i--) {
            float nnt = 1.f - d[i];
            float rho_t = fminf(imp[i], rho_clip), c_t = fminf(imp[i], c_clip);
            float delta = rho_t * (r[i] + gamma * next_v * nnt - v[i]);
            lastlam = delta + gamma * lambda * c_t * lastlam * nnt;
            adv[i] = lastlam;
            next_v = v[i];
        }
        #pragma unroll
        for (int i = 0; i < ADV_VEC_WIDTH; i++) {
            ret[i] = v[i] + adv[i];
        }
        adv_st(advantages + base, adv);
        adv_st(returns + base, ret);
    }
}
