// Custom ocean env encoders. Included by algo.cu.
// Per-env nets live under ocean/<env>/<env>.cu and are pulled in below.

#ifdef USE_ROCM
#define PUF_WARP_MASK 0xffffffffffffffffULL
#else
#define PUF_WARP_MASK 0xffffffffU
#endif

// Normal(0, std). Used by custom ocean encoders for embeddings.
void puf_normal_init(Prec* dst, float std, ulong seed, cudaStream_t stream) {
    long n = numel(dst->shape);
    assert(n > 0);
    long rand_count = (n % 2 == 0) ? n : n + 1;
    float* buf;
    cudaMalloc(&buf, rand_count * sizeof(float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, buf, rand_count, 0.0f, std);
    curandDestroyGenerator(gen);
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dst->data, buf, n);
    cudaFree(buf);
}

#ifdef USE_ROCM
#ifndef PUFFER_PACKED_OBS
#include "../ocean/nmmo3/nmmo3.hip"
#include "../ocean/minimal/minimal.hip"
#endif
#ifdef PUFFER_BALATRO
#include "../ocean/balatro/balatro.hip"
#endif
#ifdef PUFFER_NETHACK
#include "../ocean/nethack/nethack.hip"
#endif
#else
#ifndef PUFFER_PACKED_OBS
#include "../ocean/nmmo3/nmmo3.cu"
#include "../ocean/minimal/minimal.cu"
#endif
#ifdef PUFFER_BALATRO
#include "../ocean/balatro/balatro.cu"
#endif
#ifdef PUFFER_NETHACK
#include "../ocean/nethack/nethack.cu"
#endif
#endif
// Override encoder vtable for known ocean environments. No-op for unknown envs.
static void create_custom_encoder(const char* env_name, Encoder* enc) {
#ifdef PUFFER_NETHACK
    if (strcmp(env_name, "nethack") == 0) {
        create_nethack_encoder(enc);
        return;
    }
#endif
#ifndef PUFFER_PACKED_OBS
    if (strcmp(env_name, "nmmo3") == 0) {
        create_nmmo3_encoder(enc);
        return;
    }
    if (strcmp(env_name, "minimal") == 0) {
        create_minimal_encoder(enc);
        return;
    }
#endif
#ifdef PUFFER_BALATRO
    if (strcmp(env_name, "balatro") == 0) {
        create_balatro_encoder(enc);
        return;
    }
#endif
}

static void create_custom_decoder(const char* env_name, Decoder* dec) {
#ifdef PUFFER_BALATRO
    if (strcmp(env_name, "balatro") == 0) {
        create_balatro_decoder(dec);
        return;
    }
#endif
#ifdef POLICY_MASK_SIZE
    dec->ar = true;
    return;
#endif
#ifdef PUFFER_NETHACK
    if (strcmp(env_name, "nethack") == 0) {
        create_nethack_decoder(dec);
        return;
    }
#endif
}
