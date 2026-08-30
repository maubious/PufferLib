#include "balatro_core.h"
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum TargetEffect {
    TARGET_CUSTOM,
    TARGET_ENHANCEMENT,
    TARGET_SUIT,
    TARGET_RANK_UP,
    TARGET_SEAL
} TargetEffect;

typedef enum VoucherEffect {
    VOUCHER_NONE,
    VOUCHER_SHOP_SIZE,
    VOUCHER_TAROT_RATE,
    VOUCHER_PLANET_RATE,
    VOUCHER_EDITION_RATE,
    VOUCHER_PLAYING_RATE,
    VOUCHER_CONSUMABLE_SLOT,
    VOUCHER_DISCOUNT,
    VOUCHER_REROLL,
    VOUCHER_INTEREST,
    VOUCHER_HANDS,
    VOUCHER_HAND_SIZE,
    VOUCHER_DISCARDS,
    VOUCHER_JOKER_SLOT,
    VOUCHER_ANTE_HANDS,
    VOUCHER_ANTE_DISCARDS
} VoucherEffect;

typedef struct CenterDefinition {
    uint16_t id;
    uint8_t set;
    uint8_t rarity;
    int16_t cost;
    float weight;
    uint8_t base_available;
    uint8_t kind;
    uint8_t pack_extra;
    uint8_t pack_choose;
    uint16_t requires;
    float extra;
    uint8_t target_effect;
    uint8_t target_value;
    uint8_t target_max;
    uint8_t target_min;
    uint8_t voucher_effect;
} CenterDefinition;

typedef struct PlayingCardDefinition {
    uint8_t suit;
    uint8_t rank;
} PlayingCardDefinition;

typedef struct ScoreResult {
    uint8_t hand_type;
    uint8_t scoring_mask;
    uint8_t destroyed_mask;
    uint8_t reserved;
    double chips;
    double mult;
    double total;
    int32_t dollars;
} ScoreResult;

const CenterDefinition centers[CENTER_COUNT];
PlayingCardDefinition playing_card(uint8_t index);
const uint16_t *center_pool(uint8_t set, size_t *count);
const uint16_t *joker_pool(uint8_t rarity, size_t *count);

static inline uint64_t radix_key(const uint8_t *item, size_t offset, size_t bytes) {
    uint64_t key = 0;
    memcpy(&key, item + offset, bytes);
    return key;
}

static inline void radix_sort(void *values, void *scratch, size_t count, size_t stride,
                              size_t key_offset, size_t key_bytes) {
    if (count < 2) return;
    uint8_t *source = values;
    uint8_t *destination = scratch;
    for (size_t shift = 0; shift < key_bytes * 8; shift += 8) {
        uint16_t offsets[256] = {0};
        for (size_t i = 0; i < count; ++i) {
            uint64_t key = radix_key(source + i * stride, key_offset, key_bytes);
            offsets[(key >> shift) & 0xffu]++;
        }
        uint16_t position = 0;
        for (size_t bucket = 0; bucket < 256; ++bucket) {
            uint16_t size = offsets[bucket];
            offsets[bucket] = position;
            position = (uint16_t)(position + size);
        }
        for (size_t i = 0; i < count; ++i) {
            uint64_t key = radix_key(source + i * stride, key_offset, key_bytes);
            uint8_t bucket = (uint8_t)((key >> shift) & 0xffu);
            memcpy(destination + offsets[bucket]++ * stride, source + i * stride, stride);
        }
        uint8_t *swap = source;
        source = destination;
        destination = swap;
    }
    if (source != values) memcpy(values, source, count * stride);
}

/* Aggregate deck composition; same layout as the observation summaries. */
typedef ObservationDeckSummary DeckSummary;

enum CenterSet {
    SET_DEFAULT = 1,
    SET_PLAYING = 1,
    SET_ENHANCED = 2,
    SET_JOKER = 3,
    SET_TAROT = 4,
    SET_PLANET = 5,
    SET_SPECTRAL = 6,
    SET_VOUCHER = 7,
    SET_BOOSTER = 8
};

enum PackKind {
    PACK_STANDARD = 0,
    PACK_TAROT = 1,
    PACK_PLANET = 2,
    PACK_SPECTRAL = 3,
    PACK_JOKER = 5
};

static inline uint16_t playing_card_count(const State *state) {
    return (uint16_t)(state->deck_count + state->hand_count + state->discard_count);
}

static inline int can_add_playing_cards(const State *state, uint16_t count) {
    return (uint32_t)playing_card_count(state) + count <= OBS_MAX_PLAYING_CARDS;
}

/* Capacity for one more card of a set. Unknown sets report capacity (their
   rejection is handled by add_owned_card). */
static int can_own(const State *state, uint8_t set, uint8_t edition) {
    if (set == SET_JOKER)
        return state->joker_count < MAX_JOKERS &&
               (state->joker_count < state->joker_slots || edition == EDITION_NEGATIVE);
    if (set >= SET_TAROT && set <= SET_SPECTRAL)
        return state->consumable_count < MAX_CONSUMABLES &&
               (state->consumable_count < state->consumable_slots || edition == EDITION_NEGATIVE);
    return (set == SET_DEFAULT || set == SET_ENHANCED) &&
           state->deck_count < MAX_DECK && can_add_playing_cards(state, 1);
}

#define ZONE_REMOVE(zone, count, index) do { \
    memmove(&(zone)[index], &(zone)[(index) + 1], \
            ((count) - (index) - 1) * sizeof(*(zone))); \
    (count)--; \
} while (0)

static inline int action_has_primary(uint8_t type) {
    return type >= ACTION_BUY_CARD && type <= ACTION_SWAP_HAND_RIGHT;
}

typedef struct KeyBuilder {
    char *data;
    size_t capacity;
    size_t length;
} KeyBuilder;

static inline void key_begin(KeyBuilder *builder, char *data, size_t capacity) {
    *builder = (KeyBuilder){.data = data, .capacity = capacity};
    if (capacity) data[0] = '\0';
}

static inline void key_append(KeyBuilder *builder, const char *text) {
    if (!builder->capacity) return;
    while (*text && builder->length + 1 < builder->capacity) builder->data[builder->length++] = *text++;
    builder->data[builder->length] = '\0';
}

static inline void key_append_u64(KeyBuilder *builder, uint64_t value) {
    char reversed[20];
    size_t count = 0;
    do {
        reversed[count++] = (char)('0' + value % 10);
        value /= 10;
    } while (value);
    while (count && builder->length + 1 < builder->capacity) builder->data[builder->length++] = reversed[--count];
    if (builder->capacity) builder->data[builder->length] = '\0';
}

static inline void key_with_u64(char *data, size_t capacity, const char *prefix, uint64_t value) {
    KeyBuilder builder;
    key_begin(&builder, data, capacity);
    key_append(&builder, prefix);
    key_append_u64(&builder, value);
}

/* Stream key with the shared "_resample" retry suffix. */
static void key_resample(char *stream, size_t capacity, const char *key, unsigned attempt) {
    KeyBuilder builder;
    key_begin(&builder, stream, capacity);
    key_append(&builder, key);
    if (attempt) {
        key_append(&builder, "_resample");
        key_append_u64(&builder, attempt + 1);
    }
}

static void sort_indices(const Card *cards, uint8_t *indices, uint8_t count) {
    for (uint8_t i = 1; i < count; ++i) {
        uint8_t value = indices[i], j = i;
        while (j && cards[indices[j - 1]].sort_id > cards[value].sort_id) {
            indices[j] = indices[j - 1];
            --j;
        }
        indices[j] = value;
    }
}

/* Cross-references below their definitions; everything else in this file is
   defined before its first use. */
static void draw_to_hand(State *state);
static void initialize_joker_card(State *state, Card *card);
static void complete_pack_pick(State *state, uint8_t index);
static uint8_t most_played_hand(const State *state);
static void joker_added(State *state, const Card *joker);
static void joker_removed(State *state, const Card *joker);
static void consumable_added(State *state, const Card *card);
static void consumable_removed(State *state, const Card *card);
static void playing_card_added(State *state, uint8_t count);
static void clear_card_debuffs(State *state);
static int add_pooled_consumable(State *state, uint8_t set, const char *append, uint8_t edition);
static int add_specific_consumable(State *state, uint16_t center_id);
static int add_joker_rarity(State *state, uint8_t rarity, const char *append, int legendary);
static void remove_joker_at(State *state, uint8_t index);
static void remove_hand_index(State *state, uint8_t index);
static uint8_t random_sorted_hand_index(State *state, const char *stream);
static void add_spectral_cards(State *state, const uint8_t *ranks, size_t rank_count, uint8_t count, const char *stream);
static double probability_normal(const State *state, double base);
static void sort_hand_desc(State *state);
static void refresh_card_debuff(const State *state, Card *card);
static int remove_discard_sort_id(State *state, uint16_t sort_id);
static int blind_debuffs_card(const State *state, const Card *card);
static uint8_t resolved_joker_source(const State *state, uint8_t start);
static uint16_t choose_boss(State *state);
static int legal_masks(const State *state, LegalMasks *out);
static void roll_joker_edition(State *state, const char *append, Card *card);

/* Generated content metadata. Do not hand-edit. */

const CenterDefinition centers[CENTER_COUNT] = {
#define CENTER(name, id, ...) [id] = {id, __VA_ARGS__},
#include "balatro_content.def"
#undef CENTER
};

PlayingCardDefinition playing_card(uint8_t index) {
    static const uint8_t suits[] = {2, 1, 0, 3};
    static const uint8_t ranks[] = {2, 3, 4, 5, 6, 7, 8, 9, 14, 11, 13, 12, 10};
    return (PlayingCardDefinition){suits[index / 13], ranks[index % 13]};
}

static const uint16_t pool_4[] = {33, 46, 39, 30, 29, 36, 45, 22, 44, 37, 67, 59, 35, 24, 62, 26, 63, 58, 50, 60, 42, 68};
static const uint16_t pool_5[] = {49, 66, 27, 47, 43, 55, 65, 51, 54, 53, 21, 31};
static const uint16_t pool_6[] = {32, 34, 41, 61, 18, 69, 56, 52, 28, 40, 17, 25, 38, 64, 48, 23, 57, 20};
static const uint16_t pool_7[] = {284, 285, 270, 278, 276, 273, 293, 292, 271, 283, 298, 282, 274, 281, 299, 291,
                                  296, 297, 289, 290, 295, 280, 269, 268, 279, 277, 275, 288, 272, 294, 286, 287};
static const uint16_t pool_8[] = {237, 238, 239, 240, 233, 234, 235, 236, 249, 250, 251, 252, 245, 246, 247, 248,
                                  261, 262, 263, 264, 257, 258, 259, 260, 243, 244, 241, 242, 255, 256, 253, 254};
static const uint16_t pool_2[] = {225, 229, 232, 226, 230, 231, 227, 228};
static const uint16_t joker_1[] = {146, 133, 152, 222, 131, 147, 224, 153, 108, 115, 192, 221, 104, 111, 107, 137, 109, 81,  163, 75,  161,
                                   173, 102, 184, 76,  110, 135, 121, 165, 185, 93,  203, 177, 119, 182, 143, 197, 86,  122, 134, 204, 209,
                                   99,  175, 198, 178, 171, 176, 155, 138, 127, 148, 116, 132, 172, 219, 194, 207, 205, 139, 189};
static const uint16_t joker_2[] = {200, 128, 160, 100, 156, 149, 118, 124, 199, 136, 169, 196, 91,  84,  191, 106,
                                   140, 96,  154, 186, 218, 190, 142, 105, 180, 159, 150, 129, 216, 120, 208, 201,
                                   151, 90,  112, 210, 125, 215, 174, 188, 98,  162, 77,  195, 214, 101, 193, 206,
                                   181, 85,  79,  166, 130, 179, 126, 158, 167, 144, 187, 157, 183, 97,  80,  88};
static const uint16_t joker_3[] = {113, 217, 82, 164, 83, 78, 95, 87, 220, 141, 117, 213, 123, 168, 211, 202, 145, 89, 114, 92};
static const uint16_t joker_4[] = {94, 212, 223, 103, 170};

typedef struct CenterPool {
    const uint16_t *values;
    uint16_t count;
} CenterPool;

static const CenterPool pools[11] = {
    [2] = {pool_2, sizeof(pool_2) / sizeof(*pool_2)},
    [4] = {pool_4, sizeof(pool_4) / sizeof(*pool_4)},
    [5] = {pool_5, sizeof(pool_5) / sizeof(*pool_5)},
    [6] = {pool_6, sizeof(pool_6) / sizeof(*pool_6)},
    [7] = {pool_7, sizeof(pool_7) / sizeof(*pool_7)},
    [8] = {pool_8, sizeof(pool_8) / sizeof(*pool_8)},
};

static const CenterPool joker_pools[5] = {
    [1] = {joker_1, sizeof(joker_1) / sizeof(*joker_1)},
    [2] = {joker_2, sizeof(joker_2) / sizeof(*joker_2)},
    [3] = {joker_3, sizeof(joker_3) / sizeof(*joker_3)},
    [4] = {joker_4, sizeof(joker_4) / sizeof(*joker_4)},
};

const uint16_t *center_pool(uint8_t set, size_t *count) {
    assert(count);
    if (set >= sizeof(pools) / sizeof(*pools) || !pools[set].values) {
        *count = 0;
        return NULL;
    }
    *count = pools[set].count;
    return pools[set].values;
}

const uint16_t *joker_pool(uint8_t rarity, size_t *count) {
    assert(count);
    if (rarity >= sizeof(joker_pools) / sizeof(*joker_pools) || !joker_pools[rarity].values) {
        *count = 0;
        return NULL;
    }
    *count = joker_pools[rarity].count;
    return joker_pools[rarity].values;
}

static const double PI = 3.14159265358979323846;
static uint64_t bits_of_double(double value);

static double positive_fraction(double value) {
    return value - floor(value);
}

static double round_decimal13_exact(double value) {
#if defined(__SIZEOF_INT128__)
    if (!(value > 0.0)) return 0.0;
    uint64_t bits = bits_of_double(value);
    unsigned exponent = (unsigned)((bits >> 52) & 0x7ffu);
    if (!exponent) return 0.0;
    uint64_t significand = (bits & UINT64_C(0x000fffffffffffff)) | UINT64_C(0x0010000000000000);
    unsigned shift = 1075u - exponent;
    __uint128_t scaled = (__uint128_t)significand * UINT64_C(10000000000000);
    uint64_t rounded;
    if (shift >= 128) {
        rounded = 0;
    } else {
        __uint128_t quotient = scaled >> shift;
        __uint128_t remainder = scaled - (quotient << shift);
        __uint128_t halfway = (__uint128_t)1 << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (quotient & 1))) quotient++;
        rounded = (uint64_t)quotient;
    }
    return (double)rounded / 10000000000000.0;
#else
    char rounded[32];
    (void)snprintf(rounded, sizeof(rounded), "%.13f", value);
    return fabs(strtod(rounded, NULL));
#endif
}

double round_decimal13(double value) {
    if (!(value > 0.0)) return 0.0;
    double scaled = value * 10000000000000.0;
    double rounded = floor(scaled);
    double fraction = scaled - rounded;
    if (fraction > 0.499 && fraction < 0.501) return round_decimal13_exact(value);
    if (fraction >= 0.5) rounded++;
    return rounded / 10000000000000.0;
}

static uint64_t bits_of_double(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static uint64_t hash_key(const char *text) {
    uint64_t hash = UINT64_C(1469598103934665603);
    while (*text) {
        hash ^= (uint8_t)*text++;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t tw223_step(uint64_t state[4]) {
    uint64_t z = state[0];
    z = (((z << 31) ^ z) >> 45) ^ ((z & (UINT64_MAX << 1)) << 18);
    uint64_t result = state[0] = z;
    z = state[1];
    z = (((z << 19) ^ z) >> 30) ^ ((z & (UINT64_MAX << 6)) << 28);
    result ^= state[1] = z;
    z = state[2];
    z = (((z << 24) ^ z) >> 48) ^ ((z & (UINT64_MAX << 9)) << 7);
    result ^= state[2] = z;
    z = state[3];
    z = (((z << 21) ^ z) >> 39) ^ ((z & (UINT64_MAX << 17)) << 8);
    result ^= state[3] = z;
    return result;
}

static void luajit_seed(double seed, uint64_t state[4]) {
    static const uint64_t minimum[4] = {UINT64_C(2), UINT64_C(64), UINT64_C(512), UINT64_C(131072)};
    double value = seed;
    for (size_t i = 0; i < 4; ++i) {
        value = value * 3.14159265358979323846 + 2.7182818284590452354;
        state[i] = bits_of_double(value);
        if (state[i] < minimum[i]) state[i] += minimum[i];
    }
    for (size_t i = 0; i < 10; ++i) tw223_step(state);
}

static double luajit_next(uint64_t state[4]) {
    uint64_t bits = (tw223_step(state) & UINT64_C(0x000fffffffffffff)) | UINT64_C(0x3ff0000000000000);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value - 1.0;
}

double pseudohash(const char *text) {
    size_t length = strlen(text);
    double number = 1.0;
    for (size_t i = length; i > 0; --i) {
        number = positive_fraction((1.1239285023 / number) * (uint8_t)text[i - 1] * PI + PI * (double)i);
    }
    return number;
}

void rng_reset(State *state) {
    state->hashed_seed = pseudohash(state->seed);
    state->rng_count = 0;
}

static _Thread_local uint16_t rng_stream_cache[256];

/* Fast RNG path (config.fast_rng): splitmix64 per stream. Deterministic given
   (seed, stream) and statistically stronger than the LuaJIT emulation, but not
   bit-compatible with the reference game — use only for RL training. */
static inline uint64_t rng_splitmix(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static inline double rng_splitmix_unit(uint64_t *state) {
    uint64_t z = rng_splitmix(state);
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

static double advance_seed(State *state, const char *stream) {
    uint64_t key = hash_key(stream);
    uint16_t *cached = &rng_stream_cache[key & 0xffu];
    RngStream *slot = NULL;
    if (*cached < state->rng_count && state->rng[*cached].key_hash == key) {
        slot = &state->rng[*cached];
    } else {
        for (uint16_t i = state->rng_count; i > 0; --i) {
            RngStream *candidate = &state->rng[i - 1];
            if (candidate->key_hash == key) {
                slot = candidate;
                *cached = (uint16_t)(i - 1);
                break;
            }
        }
    }
    if (!slot) {
        if (state->rng_count >= MAX_RNG_STREAMS) return 0.0;
        slot = &state->rng[state->rng_count++];
        slot->key_hash = key;
        *cached = (uint16_t)(state->rng_count - 1);
        char combined[96];
        size_t stream_length = strlen(stream);
        size_t seed_length = strlen(state->seed);
        if (stream_length + seed_length >= sizeof(combined)) return 0.0;
        memcpy(combined, stream, stream_length);
        memcpy(combined + stream_length, state->seed, seed_length + 1);
        if (state->config.fast_rng) {
            uint64_t initial = hash_key(combined);
            uint64_t seeded = rng_splitmix(&initial);
            memcpy(&slot->value, &seeded, sizeof(seeded));
        } else {
            slot->value = pseudohash(combined);
        }
    }
    if (state->config.fast_rng) {
        uint64_t split_state;
        memcpy(&split_state, &slot->value, sizeof(split_state));
        double value = rng_splitmix_unit(&split_state);
        memcpy(&slot->value, &split_state, sizeof(split_state));
        return value;
    }
    double raw = positive_fraction(2.134453429141 + slot->value * 1.72431234);
    slot->value = round_decimal13(raw);
    return (slot->value + state->hashed_seed) / 2.0;
}

double pseudorandom(State *state, const char *stream) {
    if (state->config.fast_rng) return advance_seed(state, stream);
    uint64_t random_state[4];
    luajit_seed(advance_seed(state, stream), random_state);
    return luajit_next(random_state);
}

void shuffle(State *state, Card *cards, size_t count, const char *stream) {
    if (count < 2) return;
    Card sorted[MAX_DECK];
    int ordered = 1;
    for (size_t i = 1; i < count; ++i)
        if (cards[i - 1].sort_id > cards[i].sort_id) {
            ordered = 0;
            break;
        }
    if (!ordered) radix_sort(cards, sorted, count, sizeof(*cards), offsetof(Card, sort_id), sizeof(uint16_t));
    if (state->config.fast_rng) {
        double fast_seed = advance_seed(state, stream);
        uint64_t split_state;
        memcpy(&split_state, &fast_seed, sizeof(split_state));
        for (size_t i = count - 1; i > 0; --i) {
            uint64_t z = rng_splitmix(&split_state);
            size_t j = (size_t)(((z >> 11) * (uint64_t)(i + 1)) >> 53);
            Card tmp = cards[i];
            cards[i] = cards[j];
            cards[j] = tmp;
        }
        return;
    }
    uint64_t tw_state[4];
    luajit_seed(advance_seed(state, stream), tw_state);
    for (size_t i = count - 1; i > 0; --i) {
        size_t j = (size_t)floor(luajit_next(tw_state) * (double)(i + 1));
        Card tmp = cards[i];
        cards[i] = cards[j];
        cards[j] = tmp;
    }
}

int state_layout_valid(const State *state) {
    return state && state->rng_count <= MAX_RNG_STREAMS && state->deck_count <= MAX_DECK &&
           state->hand_count <= MAX_HAND && state->discard_count <= MAX_DECK && state->joker_count <= MAX_JOKERS &&
           state->consumable_count <= MAX_CONSUMABLES &&
           state->shop_main_count <= OBS_MAX_SHOP_MAIN &&
           state->shop_voucher_count <= OBS_MAX_SHOP_VOUCHERS &&
           state->shop_booster_count <= OBS_MAX_SHOP_BOOSTERS &&
           playing_card_count(state) <= OBS_MAX_PLAYING_CARDS &&
           state->pack_count <= MAX_PACK_CARDS && state->phase <= PHASE_GAME_OVER &&
           memchr(state->seed, 0, sizeof(state->seed));
}

static int joker_cache_bit(uint16_t center_id) {
    switch (center_id) {
    case CENTER_J_FOUR_FINGERS: return 0;
    case CENTER_J_SHORTCUT: return 1;
    case CENTER_J_SMEARED: return 2;
    case CENTER_J_PAREIDOLIA: return 3;
    case CENTER_J_SPLASH: return 4;
    case CENTER_J_OOPS: return 5;
    case CENTER_J_DNA: return 6;
    case CENTER_J_RING_MASTER: return 7;
    case CENTER_J_CERTIFICATE: return 8;
    case CENTER_J_SIXTH_SENSE: return 9;
    case CENTER_J_SEANCE: return 10;
    case CENTER_J_SUPERPOSITION: return 11;
    case CENTER_J_VAGABOND: return 12;
    case CENTER_J_PERKEO: return 13;
    default: return -1;
    }
}

void refresh_joker_cache(State *state) {
    state->joker_flags = 0;
    state->joker_active_flags = 0;
    for (uint8_t i = 0; i < state->joker_count; ++i) {
        int bit = joker_cache_bit(state->jokers[i].center_id);
        if (bit >= 0) {
            uint64_t flag = UINT64_C(1) << bit;
            state->joker_flags |= flag;
            if (!(state->jokers[i].flags & CARD_DEBUFFED)) state->joker_active_flags |= flag;
        }
    }
}

int joker_active(const State *state, uint16_t center_id) {
    int bit = joker_cache_bit(center_id);
    if (bit >= 0) return (state->joker_active_flags & (UINT64_C(1) << bit)) != 0;
    for (uint8_t i = 0; i < state->joker_count; ++i)
        if (!(state->jokers[i].flags & CARD_DEBUFFED) && state->jokers[i].center_id == center_id) return 1;
    return 0;
}

static uint64_t hash_bytes(uint64_t hash, const void *data, size_t length) {
    const uint8_t *bytes = data;
    for (size_t i = 0; i < length; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

uint64_t state_hash(const State *state) {
    if (!state_layout_valid(state)) return 0;
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = hash_bytes(hash, state, offsetof(State, rng));
    hash = hash_bytes(hash, state->rng, (size_t)state->rng_count * sizeof(state->rng[0]));
    hash = hash_bytes(hash, &state->rng_count, offsetof(State, deck) - offsetof(State, rng_count));
    const Card *zones[] = {state->deck, state->hand, state->discard, state->jokers,
                           state->consumables, state->shop_main, state->shop_vouchers,
                           state->shop_boosters, state->pack_cards};
    const uint16_t counts[] = {state->deck_count, state->hand_count, state->discard_count,
                               state->joker_count, state->consumable_count, state->shop_main_count,
                               state->shop_voucher_count, state->shop_booster_count, state->pack_count};
    for (int zone = 0; zone < 9; ++zone)
        hash = hash_bytes(hash, zones[zone], (size_t)counts[zone] * sizeof(Card));
    return hash_bytes(hash, &state->deck_count, sizeof(*state) - offsetof(State, deck_count));
}

static const int16_t base_chips[HAND_COUNT] = {160, 140, 120, 100, 60, 40, 35, 30, 30, 20, 10, 5};
static const int16_t base_mult[HAND_COUNT] = {16, 14, 12, 8, 7, 4, 4, 4, 3, 2, 2, 1};
static const int16_t level_chips[HAND_COUNT] = {50, 40, 35, 40, 30, 25, 15, 30, 20, 20, 15, 10};
static const int16_t level_mult[HAND_COUNT] = {3, 4, 3, 4, 3, 2, 2, 3, 2, 1, 1, 1};

HandType classify_hand_rules(const Card *cards, size_t count, uint8_t *scoring_mask,
    int four_fingers, int shortcut, int smeared) {
    uint8_t suits[4] = {0};
    uint8_t rank_masks[15] = {0};
    uint8_t flush_mask = 0;
    memset(scoring_mask, 0, sizeof(*scoring_mask));
    assert(count && count <= MAX_SELECTION);
    for (size_t i = 0; i < count; ++i) {
        if (cards[i].enhancement == ENHANCEMENT_STONE) continue;
        uint8_t rank = cards[i].rank;
        if (rank >= 2 && rank <= 14) rank_masks[rank] |= (uint8_t)(1u << i);
        if (cards[i].enhancement == ENHANCEMENT_WILD)
            for (int suit = 0; suit < 4; ++suit) suits[suit]++;
        else if (cards[i].suit < 4)
            suits[smeared ? (cards[i].suit == HEARTS || cards[i].suit == DIAMONDS ? 0 : 1)
                          : cards[i].suit]++;
    }
    int flush = -1;
    int need = four_fingers ? 4 : 5;
    for (int s = 0; s < (smeared ? 2 : 4); ++s)
        if (suits[s] >= need) flush = s;
    if (flush >= 0)
        for (size_t i = 0; i < count; ++i) {
            if (cards[i].enhancement == ENHANCEMENT_STONE) continue;
            if (cards[i].enhancement == ENHANCEMENT_WILD ||
                (smeared ? (cards[i].suit == HEARTS || cards[i].suit == DIAMONDS ? 0 : 1)
                         : cards[i].suit) == flush)
                flush_mask |= (uint8_t)(1u << i);
        }
    uint8_t straight_mask = 0;
    for (int high = 14; high >= need && !straight_mask; --high) {
        uint8_t mask = 0;
        int d = 0;
        while (d < need && rank_masks[high - d]) mask |= rank_masks[high - d++];
        if (d == need) straight_mask = mask;
    }
    if (!straight_mask && rank_masks[14]) {
        uint8_t wheel_mask = rank_masks[14];
        int wheel_ok = 1;
        for (int rank = 2; rank <= need; ++rank) {
            if (!rank_masks[rank]) wheel_ok = 0;
            wheel_mask |= rank_masks[rank];
        }
        if (wheel_ok) straight_mask = wheel_mask;
    }
    if (!straight_mask && shortcut) {
        static const uint8_t order[] = {14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
        int length = 0, last = -3;
        uint8_t mask = 0;
        for (int position = 0; position < (int)sizeof(order); ++position) {
            uint8_t rank = order[position];
            if (!rank_masks[rank]) continue;
            if (length && position - last > 2) {
                length = 0;
                mask = 0;
            }
            if (!(mask & rank_masks[rank])) {
                mask |= rank_masks[rank];
                length++;
            }
            last = position;
            if (length >= need) {
                straight_mask = mask;
                break;
            }
        }
    }
    int five = 0, four = 0, three = 0, pair1 = 0, pair2 = 0;
    for (int rank = 14; rank >= 2; --rank) {
        uint8_t rank_count = (uint8_t)__builtin_popcount(rank_masks[rank]);
        if (rank_count == 5)
            five = rank;
        else if (rank_count == 4)
            four = rank;
        else if (rank_count == 3 && !three)
            three = rank;
        else if (rank_count == 2) {
            if (!pair1)
                pair1 = rank;
            else if (!pair2)
                pair2 = rank;
        }
    }
    if (five) {
        *scoring_mask = rank_masks[five];
        return flush >= 0 ? FLUSH_FIVE : FIVE_OF_A_KIND;
    }
    if (three && pair1) {
        *scoring_mask = rank_masks[three] | rank_masks[pair1];
        return flush >= 0 ? FLUSH_HOUSE : FULL_HOUSE;
    }
    if (straight_mask && flush >= 0 && (straight_mask & flush_mask) == straight_mask) {
        *scoring_mask = straight_mask;
        return STRAIGHT_FLUSH;
    }
    if (four) {
        *scoring_mask = rank_masks[four];
        return FOUR_OF_A_KIND;
    }
    if (flush >= 0) {
        *scoring_mask = flush_mask;
        return FLUSH;
    }
    if (straight_mask) {
        *scoring_mask = straight_mask;
        return STRAIGHT;
    }
    if (three) {
        *scoring_mask = rank_masks[three];
        return THREE_OF_A_KIND;
    }
    if (pair2) {
        *scoring_mask = rank_masks[pair1] | rank_masks[pair2];
        return TWO_PAIR;
    }
    if (pair1) {
        *scoring_mask = rank_masks[pair1];
        return PAIR;
    }
    int high = 14;
    while (high && !rank_masks[high]) high--;
    if (high) *scoring_mask = rank_masks[high] & (uint8_t)(-(int8_t)rank_masks[high]);
    return HIGH_CARD;
}

static HandType classify_hand(const Card *cards, size_t count, uint8_t *scoring_mask) {
    return classify_hand_rules(cards, count, scoring_mask, 0, 0, 0);
}

static double blind_amount(uint8_t ante, uint8_t scaling) {
    static const int32_t amounts[3][8] = {
        {300, 800, 2000, 5000, 11000, 20000, 35000, 50000},
        {300, 900, 2600, 8000, 20000, 36000, 60000, 100000},
        {300, 1000, 3200, 9000, 25000, 60000, 110000, 200000},
    };
    if (ante < 1) return 100;
    if (scaling < 1 || scaling > 3) scaling = 1;
    if (ante <= 8) return amounts[scaling - 1][ante - 1];
    double c = (double)ante - 8.0;
    double d = 1.0 + 0.2 * c;
    double raw = floor(amounts[scaling - 1][7] * pow(1.6 + pow(0.75 * c, d), c));
    double rounding = pow(10.0, floor(log10(raw) - 1.0));
    return raw - fmod(raw, rounding);
}

static int32_t calculate_round_earnings(const State *state) {
    if (!state) return 0;
    int32_t interest = 0;
    if (state->dollars >= 5 && state->config.deck != CENTER_B_GREEN) {
        int32_t brackets = state->dollars / 5;
        int32_t cap = state->interest_cap / 5;
        if (brackets > cap) brackets = cap;
        interest = brackets * state->interest_amount;
    }
    int bonus = 0;
    int nines = 0;
    const Card *playing_zones[] = {state->deck, state->hand, state->discard};
    const uint16_t playing_counts[] = {state->deck_count, state->hand_count, state->discard_count};
    for (int zone = 0; zone < 3; ++zone)
        for (uint16_t i = 0; i < playing_counts[zone]; ++i)
            nines += playing_zones[zone][i].rank == 9;
    for (uint8_t i = 0; i < state->joker_count; ++i) {
        const Card *joker = &state->jokers[i];
        if (joker->flags & CARD_DEBUFFED) continue;
        switch (joker->center_id) {
        case CENTER_J_GOLDEN:
            bonus += 4;
            break;
        case CENTER_J_CLOUD_9:
            bonus += nines;
            break;
        case CENTER_J_ROCKET:
            bonus += joker->state[0] > 0 ? joker->state[0] : 1;
            break;
        case CENTER_J_DELAYED_GRAT:
            if (!state->discards_used && state->discards_left) bonus += state->discards_left * 2;
            break;
        case CENTER_J_SATELLITE:
            for (uint16_t mask = state->planet_usage_mask; mask; mask &= (uint16_t)(mask - 1)) bonus++;
            break;
        default:
            break;
        }
    }
    int hand_bonus = state->hands_left * (state->config.deck == CENTER_B_GREEN ? 2 : 1);
    int discard_bonus = state->config.deck == CENTER_B_GREEN ? state->discards_left : 0;
    return state->blind_reward + hand_bonus + discard_bonus + interest + bonus;
}

uint8_t card_set(const Card *card) {
    if (card->center_id >= CENTER_COUNT) return 0;
    return centers[card->center_id].set;
}

static const uint16_t planet_centers[HAND_COUNT] = {
    CENTER_C_ERIS,  CENTER_C_CERES,  CENTER_C_PLANET_X, CENTER_C_NEPTUNE,
    CENTER_C_MARS,  CENTER_C_EARTH,  CENTER_C_JUPITER,  CENTER_C_SATURN,
    CENTER_C_VENUS, CENTER_C_URANUS, CENTER_C_MERCURY,  CENTER_C_PLUTO,
};

uint16_t planet_center(uint8_t hand) {
    return planet_centers[hand];
}

HandType planet_hand(uint16_t center_id) {
    for (uint8_t hand = 0; hand < HAND_COUNT; ++hand)
        if (planet_centers[hand] == center_id) return (HandType)hand;
    return HAND_COUNT;
}

int center_used(const State *state, uint16_t id) {
    return (state->used_centers[id / 8] >> (id % 8)) & 1u;
}

void mark_center_used(State *state, uint16_t id) {
    state->used_centers[id / 8] |= (uint8_t)(1u << (id % 8));
}

void unmark_center_used(State *state, uint16_t id) {
    state->used_centers[id / 8] &= (uint8_t)~(1u << (id % 8));
}

static int showman_active(const State *state) {
    for (uint8_t i = 0; i < state->joker_count; ++i)
        if (!(state->jokers[i].flags & CARD_DEBUFFED) && state->jokers[i].center_id == CENTER_J_RING_MASTER) return 1;
    return 0;
}

static int center_used_for_pool(const State *state, uint16_t id) {
    if (center_used(state, id)) return 1;
    const Card *zones[] = {state->shop_main, state->shop_vouchers,
                           state->shop_boosters, state->pack_cards};
    const uint8_t counts[] = {state->shop_main_count, state->shop_voucher_count,
                              state->shop_booster_count, state->pack_count};
    for (int zone = 0; zone < 4; ++zone)
        for (uint8_t i = 0; i < counts[zone]; ++i)
            if (zones[zone][i].center_id == id) return 1;
    return 0;
}

static int center_available(const State *state, uint16_t id) {
    if (id == CENTER_J_GROS_MICHEL) return !state->gros_michel_extinct;
    if (id == CENTER_J_CAVENDISH) return state->gros_michel_extinct;
    uint8_t gate = id == CENTER_J_STEEL_JOKER ? ENHANCEMENT_STEEL
                   : id == CENTER_J_STONE     ? ENHANCEMENT_STONE
                   : id == CENTER_J_LUCKY_CAT ? ENHANCEMENT_LUCKY
                   : id == CENTER_J_TICKET    ? ENHANCEMENT_GOLD
                   : id == CENTER_J_GLASS     ? ENHANCEMENT_GLASS
                                                      : 0;
    if (gate) {
        const Card *zones[] = {state->deck, state->hand, state->discard};
        const uint16_t counts[] = {state->deck_count, state->hand_count, state->discard_count};
        for (int zone = 0; zone < 3; ++zone)
            for (uint16_t i = 0; i < counts[zone]; ++i)
                if (zones[zone][i].enhancement == gate) return 1;
        return 0;
    }
    return centers[id].base_available;
}

static uint16_t pick_pool(State *state, uint8_t set, uint8_t rarity, const char *key) {
    size_t count = 0;
    const uint16_t *pool = set == SET_JOKER ? joker_pool(rarity, &count) : center_pool(set, &count);
    if (!pool || !count) return CENTER_NONE;
    int ring_master = showman_active(state);
    for (unsigned attempt = 0; attempt < 21; ++attempt) {
        char stream[64];
        key_resample(stream, sizeof(stream), key, attempt);
        size_t index = (size_t)floor(pseudorandom(state, stream) * count);
        uint16_t id = pool[index];
        int available = center_available(state, id);
        if (set == SET_PLANET) {
            uint8_t hand = id == CENTER_C_PLANET_X ? FIVE_OF_A_KIND
                           : id == CENTER_C_CERES  ? FLUSH_HOUSE
                           : id == CENTER_C_ERIS   ? FLUSH_FIVE
                                                           : HAND_COUNT;
            if (hand < HAND_COUNT && !state->hand_plays[hand]) available = 0;
        }
        if ((!center_used_for_pool(state, id) || ring_master) && available) return id;
    }
    if (set == SET_JOKER) {
        for (size_t i = 0; i < count; ++i) {
            uint16_t id = pool[i];
            int available = center_available(state, id);
            if ((!center_used_for_pool(state, id) || ring_master) && available) return id;
        }
        return CENTER_J_JOKER;
    }
    return pool[0];
}

static uint8_t choose_shop_set(State *state) {
    char stream[32];
    key_with_u64(stream, sizeof(stream), "cdt", state->ante);
    double total = state->joker_rate + state->tarot_rate + state->planet_rate + state->spectral_rate + state->playing_card_rate;
    double poll = pseudorandom(state, stream) * total;
    uint8_t playing_set = SET_DEFAULT;
    if (center_used(state, CENTER_V_ILLUSION) && pseudorandom(state, "illusion") > 0.6)
        playing_set = SET_ENHANCED;
    if (poll < state->joker_rate) return SET_JOKER;
    poll -= state->joker_rate;
    if (poll < state->tarot_rate) return SET_TAROT;
    poll -= state->tarot_rate;
    if (poll < state->planet_rate) return SET_PLANET;
    poll -= state->planet_rate;
    if (poll < state->playing_card_rate) return playing_set;
    return SET_SPECTRAL;
}

static uint16_t create_pool_center(State *state, uint8_t set, const char *append, int pack_area) {
    char stream[64];
    if (set == SET_JOKER) {
        if (!pack_area && state->tag_force_rarity) {
            uint8_t rarity = state->tag_force_rarity;
            if (state->tag_force_rarity_count > 1)
                state->tag_force_rarity_count--;
            else {
                state->tag_force_rarity = 0;
                state->tag_force_rarity_count = 0;
            }
            KeyBuilder key;
            key_begin(&key, stream, sizeof(stream));
            key_append(&key, "Joker");
            key_append_u64(&key, rarity);
            key_append(&key, append);
            key_append_u64(&key, state->ante);
            return pick_pool(state, set, rarity, stream);
        }
        KeyBuilder key;
        key_begin(&key, stream, sizeof(stream));
        key_append(&key, "rarity");
        key_append_u64(&key, state->ante);
        key_append(&key, append);
        double roll = pseudorandom(state, stream);
        uint8_t rarity = roll > 0.95 ? 3 : roll > 0.7 ? 2 : 1;
        key_begin(&key, stream, sizeof(stream));
        key_append(&key, "Joker");
        key_append_u64(&key, rarity);
        key_append(&key, append);
        key_append_u64(&key, state->ante);
        return pick_pool(state, set, rarity, stream);
    }
    const char *name = set == SET_TAROT ? "Tarot" : set == SET_PLANET ? "Planet" : "Spectral";
    if (set == SET_TAROT && pack_area) {
        key_with_u64(stream, sizeof(stream), "soul_Tarot", state->ante);
        if ((!center_used_for_pool(state, CENTER_C_SOUL) || showman_active(state)) && pseudorandom(state, stream) > 0.997)
            return CENTER_C_SOUL;
    }
    if (set == SET_PLANET && pack_area) {
        key_with_u64(stream, sizeof(stream), "soul_Planet", state->ante);
        if ((!center_used_for_pool(state, CENTER_C_BLACK_HOLE) || showman_active(state)) &&
            pseudorandom(state, stream) > 0.997)
            return CENTER_C_BLACK_HOLE;
    }
    if (set == SET_SPECTRAL && pack_area) {
        key_with_u64(stream, sizeof(stream), "soul_Spectral", state->ante);
        if ((!center_used_for_pool(state, CENTER_C_SOUL) || showman_active(state)) && pseudorandom(state, stream) > 0.997)
            return CENTER_C_SOUL;
        if ((!center_used_for_pool(state, CENTER_C_BLACK_HOLE) || showman_active(state)) &&
            pseudorandom(state, stream) > 0.997)
            return CENTER_C_BLACK_HOLE;
    }
    KeyBuilder key;
    key_begin(&key, stream, sizeof(stream));
    key_append(&key, name);
    key_append(&key, append);
    key_append_u64(&key, state->ante);
    return pick_pool(state, set, 0, stream);
}

void price_card(const State *state, Card *card) {
    int cost = centers[card->center_id].cost;
    int edition_cost = card->edition == EDITION_FOIL ? 2 : card->edition == EDITION_HOLO ? 3 : card->edition == EDITION_POLYCHROME || card->edition == EDITION_NEGATIVE ? 5 : 0;
    int raw = cost + edition_cost;
    int free_planet = 0;
    if (card_set(card) == SET_PLANET)
        for (uint8_t i = 0; i < state->joker_count; ++i)
            if (state->jokers[i].center_id == CENTER_J_ASTRONOMER && !(state->jokers[i].flags & CARD_DEBUFFED)) {
                raw = 0;
                free_planet = 1;
            }
    int discount = state->discount_percent > 100 ? 100 : state->discount_percent;
    int discounted = (int)floor((raw + 0.5) * (100 - discount) / 100.0);
    card->cost = (int16_t)(free_planet ? 0 : (discounted > 0 ? discounted : 1));
    card->sell_cost = (int16_t)(free_planet ? 0 : (card->cost / 2 > 0 ? card->cost / 2 : 1));
}

static void reprice_all_cards(State *state) {
    Card *zones[] = {state->jokers, state->consumables, state->shop_main,
                     state->shop_vouchers, state->shop_boosters, state->pack_cards};
    const uint8_t counts[] = {state->joker_count, state->consumable_count,
                              state->shop_main_count, state->shop_voucher_count,
                              state->shop_booster_count, state->pack_count};
    for (int zone = 0; zone < 6; ++zone)
        for (uint8_t i = 0; i < counts[zone]; ++i) {
            Card *card = &zones[zone][i];
            int old_base_sell = card->cost == 0 ? 0 : (card->cost / 2 > 0 ? card->cost / 2 : 1);
            int extra_value = card->cost == 0 ? 0 : card->sell_cost - old_base_sell;
            int couponed = zone > 1 && zone < 5 && card->cost == 0;
            price_card(state, card);
            if (card->flags & CARD_RENTAL) {
                card->cost = 1;
                card->sell_cost = 1;
            }
            if (extra_value > 0) card->sell_cost += (int16_t)extra_value;
            if (couponed) card->cost = 0;
        }
}

Card create_pooled_card(State *state, uint8_t set, const char *append, int pack_area) {
    Card card = {0};
    uint8_t forced_edition = EDITION_NONE;
    if (set == SET_DEFAULT)
        card.center_id = CENTER_C_BASE;
    else
        card.center_id = create_pool_center(state, set, append, pack_area);
    card.sort_id = ++state->next_sort_id;
    if (set == SET_DEFAULT || set == SET_ENHANCED) {
        char stream[32];
        KeyBuilder key;
        key_begin(&key, stream, sizeof(stream));
        key_append(&key, "front");
        key_append(&key, append);
        key_append_u64(&key, state->ante);
    size_t front = (size_t)floor(pseudorandom(state, stream) * 52.0);
        PlayingCardDefinition definition = playing_card((uint8_t)front);
        card.suit = definition.suit;
        card.rank = definition.rank;
        switch (card.center_id) {
        case CENTER_M_BONUS:
            card.enhancement = ENHANCEMENT_BONUS;
            break;
        case CENTER_M_MULT:
            card.enhancement = ENHANCEMENT_MULT;
            break;
        case CENTER_M_WILD:
            card.enhancement = ENHANCEMENT_WILD;
            break;
        case CENTER_M_GLASS:
            card.enhancement = ENHANCEMENT_GLASS;
            break;
        case CENTER_M_STEEL:
            card.enhancement = ENHANCEMENT_STEEL;
            break;
        case CENTER_M_STONE:
            card.enhancement = ENHANCEMENT_STONE;
            break;
        case CENTER_M_GOLD:
            card.enhancement = ENHANCEMENT_GOLD;
            break;
        case CENTER_M_LUCKY:
            card.enhancement = ENHANCEMENT_LUCKY;
            break;
        }
    }
    if (set == SET_JOKER) {
        initialize_joker_card(state, &card);
        char stream[32];
        key_with_u64(stream, sizeof(stream), pack_area ? "packetper" : "etperpoll", state->ante);
        double ep = pseudorandom(state, stream);
        if (state->config.stake >= 4 && ep > 0.7)
            card.flags |= CARD_ETERNAL;
        else if (state->config.stake >= 7 && ep > 0.4)
            card.flags |= CARD_PERISHABLE;
        key_with_u64(stream, sizeof(stream), pack_area ? "packssjr" : "ssjr", state->ante);
        if (state->config.stake >= 8 && pseudorandom(state, stream) > 0.7) card.flags |= CARD_RENTAL;
        roll_joker_edition(state, append, &card);
        if (!pack_area && state->tag_force_edition && !card.edition) {
            forced_edition = state->tag_force_edition;
            card.edition = forced_edition;
            if (state->tag_force_edition_count > 1)
                state->tag_force_edition_count--;
            else {
                state->tag_force_edition = EDITION_NONE;
                state->tag_force_edition_count = 0;
            }
        }
        if (card.flags & CARD_PERISHABLE) card.state[3] = 5;
    }
    price_card(state, &card);
    if (forced_edition) card.cost = 0;
    return card;
}

/* Random joker edition roll on the shared "edi"+append stream. */
static void roll_joker_edition(State *state, const char *append, Card *card) {
    char stream[32];
    KeyBuilder key;
    key_begin(&key, stream, sizeof(stream));
    key_append(&key, "edi");
    key_append(&key, append);
    key_append_u64(&key, state->ante);
    double edition = pseudorandom(state, stream);
    if (edition > 0.997)
        card->edition = EDITION_NEGATIVE;
    else if (edition > 1.0 - 0.006 * state->edition_rate)
        card->edition = EDITION_POLYCHROME;
    else if (edition > 1.0 - 0.02 * state->edition_rate)
        card->edition = EDITION_HOLO;
    else if (edition > 1.0 - 0.04 * state->edition_rate)
        card->edition = EDITION_FOIL;
}

static Card create_shop_card(State *state) {
    if (state->tag_force_rarity) {
        uint8_t rarity = state->tag_force_rarity;
        const char *append = rarity == 3 ? "rta" : "uta";
        Card card = create_pooled_card(state, SET_JOKER, append, 0);
        card.cost = 0;
        return card;
    }
    uint8_t set = choose_shop_set(state);
    Card card = create_pooled_card(state, set, "sho", 0);
    if ((set == SET_DEFAULT || set == SET_ENHANCED) && center_used(state, CENTER_V_ILLUSION) &&
        pseudorandom(state, "illusion") > 0.8) {
        double edition = pseudorandom(state, "illusion");
        card.edition = edition > 0.85 ? EDITION_POLYCHROME
                       : edition > 0.5 ? EDITION_HOLO
                                       : EDITION_FOIL;
        price_card(state, &card);
    }
    return card;
}

/* Fill the shop's joker slots from a clean count of zero. */
static void fill_shop_jokers(State *state) {
    for (uint8_t i = 0; i < state->shop_joker_max; ++i) {
        Card card = create_shop_card(state);
        if (state->shop_main_count < OBS_MAX_SHOP_MAIN)
            state->shop_main[state->shop_main_count++] = card;
    }
}

static Card center_card(State *state, uint16_t center_id) {
    Card card = {0};
    card.center_id = center_id;
    card.sort_id = ++state->next_sort_id;
    price_card(state, &card);
    return card;
}

static uint16_t pick_voucher_stream(State *state, const char *key) {
    size_t count = 0;
    const uint16_t *pool = center_pool(SET_VOUCHER, &count);
    for (unsigned attempt = 0; attempt < 64; ++attempt) {
        char stream[64];
        key_resample(stream, sizeof(stream), key, attempt);
        size_t index = (size_t)floor(pseudorandom(state, stream) * count);
        uint16_t id = pool[index];
        uint16_t requirement = centers[id].requires;
        if (!center_used(state, id) && (!requirement || center_used(state, requirement))) return id;
    }
    return CENTER_V_BLANK;
}

uint16_t pick_voucher(State *state) {
    char key[32];
    key_with_u64(key, sizeof(key), "Voucher", state->ante);
    return pick_voucher_stream(state, key);
}

uint16_t pick_voucher_from_tag(State *state) {
    return pick_voucher_stream(state, "Voucher_fromtag");
}

static uint16_t pick_booster(State *state, int first) {
    if (first) return CENTER_P_BUFFOON_NORMAL_1;
    size_t count = 0;
    const uint16_t *pool = center_pool(SET_BOOSTER, &count);
    double total = 0.0;
    for (size_t i = 0; i < count; ++i) total += centers[pool[i]].weight;
    char stream[32];
    key_with_u64(stream, sizeof(stream), "shop_pack", state->ante);
    double poll = pseudorandom(state, stream) * total;
    double cumulative = 0.0;
    for (size_t i = 0; i < count; ++i) {
        cumulative += centers[pool[i]].weight;
        if (poll <= cumulative) return pool[i];
    }
    return pool[count - 1];
}

void populate_shop(State *state) {
    state->shop_main_count = 0;
    state->shop_voucher_count = 0;
    state->shop_booster_count = 0;
    state->tag_d_six_active = 0;
    if (state->tag_d_six_pending) {
        state->tag_d_six_pending = 0;
        state->tag_d_six_active = 1;
        state->free_rerolls = 0;
        state->reroll_increase = 0;
        state->reroll_cost = 0;
    }
    int coupon = state->tag_coupon_pending != 0;
    state->tag_coupon_pending = 0;
    fill_shop_jokers(state);
    if (state->next_voucher_id && state->shop_voucher_count < OBS_MAX_SHOP_VOUCHERS)
        state->shop_vouchers[state->shop_voucher_count++] = center_card(state, state->next_voucher_id);
    while (state->tag_voucher_pending && state->shop_voucher_count < OBS_MAX_SHOP_VOUCHERS) {
        Card free_voucher = center_card(state, pick_voucher_from_tag(state));
        free_voucher.cost = 0;
        free_voucher.sell_cost = 0;
        state->shop_vouchers[state->shop_voucher_count++] = free_voucher;
        state->tag_voucher_pending--;
    }
    state->shop_boosters[state->shop_booster_count++] = center_card(state, pick_booster(state, !state->first_shop_buffoon));
    state->first_shop_buffoon = 1;
    state->shop_boosters[state->shop_booster_count++] = center_card(state, pick_booster(state, 0));
    if (coupon) {
        for (uint8_t i = 0; i < state->shop_main_count; ++i) state->shop_main[i].cost = 0;
        for (uint8_t i = 0; i < state->shop_booster_count; ++i) state->shop_boosters[i].cost = 0;
    }
}

int can_afford(const State *state, int32_t cost) {
    int32_t bankrupt_at = 0;
    for (uint8_t j = 0; j < state->joker_count; ++j)
        if (state->jokers[j].center_id == CENTER_J_CREDIT_CARD && !(state->jokers[j].flags & CARD_DEBUFFED))
            bankrupt_at -= 20;
    return (int64_t)state->dollars - cost >= bankrupt_at;
}

static int add_owned_card(State *state, Card card, uint8_t set) {
    if (set == SET_JOKER) {
        state->jokers[state->joker_count++] = card;
        mark_center_used(state, card.center_id);
        joker_added(state, &state->jokers[state->joker_count - 1]);
    } else if (set >= SET_TAROT && set <= SET_SPECTRAL) {
        state->consumables[state->consumable_count++] = card;
        consumable_added(state, &card);
    } else if (set == SET_DEFAULT || set == SET_ENHANCED) {
        memmove(&state->deck[1], &state->deck[0], state->deck_count * sizeof(Card));
        state->deck[0] = card;
        state->deck_count++;
        playing_card_added(state, 1);
    } else {
        return 0;
    }
    return 1;
}

int buy_shop_card(State *state, uint8_t index) {
    if (!state || index >= state->shop_main_count) return ERR_ACTION;
    Card card = state->shop_main[index];
    uint8_t set = card_set(&card);
    if (!can_afford(state, card.cost)) return ERR_ACTION;
    if (!can_own(state, set, card.edition)) return ERR_CAPACITY;
    state->dollars -= card.cost;
    if (!add_owned_card(state, card, set)) return ERR_ACTION;
    ZONE_REMOVE(state->shop_main, state->shop_main_count, index);
    return OK;
}

int sell_joker(State *state, uint8_t index) {
    if (!state || index >= state->joker_count || (state->jokers[index].flags & CARD_ETERNAL)) return ERR_ACTION;
    Card card = state->jokers[index];
    state->dollars += card.sell_cost;
    joker_removed(state, &card);
    if (!(card.flags & CARD_DEBUFFED) && card.center_id == CENTER_J_DIET_COLA) state->double_tag = 1;
    if (!(card.flags & CARD_DEBUFFED) && card.center_id == CENTER_J_INVISIBLE && card.state[0] >= 2 && state->joker_count > 0 &&
        state->joker_count - 1 < state->joker_slots) {
        uint8_t eligible[MAX_JOKERS], eligible_count = 0;
        for (uint8_t j = 0; j < state->joker_count; ++j)
            if (j != index) eligible[eligible_count++] = j;
        if (eligible_count) {
            sort_indices(state->jokers, eligible, eligible_count);
            size_t pick = (size_t)floor(pseudorandom(state, "invisible") * eligible_count);
            if (pick >= eligible_count) pick = eligible_count - 1;
            Card copy = state->jokers[eligible[pick]];
            copy.sort_id = ++state->next_sort_id;
            if (copy.center_id == CENTER_J_INVISIBLE) copy.state[0] = 0;
            state->jokers[state->joker_count++] = copy;
            joker_added(state, &state->jokers[state->joker_count - 1]);
        }
    }
    int luchador_disables =
        !(card.flags & CARD_DEBUFFED) && card.center_id == CENTER_J_LUCHADOR && state->blind_on_deck == 2;
    if (!state->blind_disabled && (luchador_disables || state->blind_id == BLIND_BL_FINAL_LEAF)) {
        state->blind_disabled = 1;
        clear_card_debuffs(state);
    }
    for (uint8_t j = 0; j < state->joker_count; ++j)
        if (j != index && !(state->jokers[j].flags & CARD_DEBUFFED) && state->jokers[j].center_id == CENTER_J_CAMPFIRE)
            state->jokers[j].state[0] = (state->jokers[j].state[0] > 100 ? state->jokers[j].state[0] : 100) + 25;
    ZONE_REMOVE(state->jokers, state->joker_count, index);
    refresh_joker_cache(state);
    return OK;
}

int reroll_shop(State *state) {
    if (!state || (state->free_rerolls == 0 && !can_afford(state, state->reroll_cost))) return ERR_ACTION;
    int free = state->free_rerolls != 0;
    if (free) {
        state->free_rerolls--;
    } else
        state->dollars -= state->reroll_cost;
    if (!free) state->reroll_increase++;
    state->reroll_cost = state->free_rerolls ? 0 : (state->tag_d_six_active ? 0 : state->reroll_base) + state->reroll_increase;
    state->shop_main_count = 0;
    fill_shop_jokers(state);
    return OK;
}

static void apply_voucher(State *state, uint16_t id) {
    const CenterDefinition *voucher = &centers[id];
    const float extra = voucher->extra;
    switch (voucher->voucher_effect) {
    case VOUCHER_SHOP_SIZE: {
        state->shop_joker_max++;
        while (state->shop_main_count < state->shop_joker_max &&
               state->shop_main_count < OBS_MAX_SHOP_MAIN) {
            Card card = create_shop_card(state);
            state->shop_main[state->shop_main_count++] = card;
        }
        break;
    }
    case VOUCHER_TAROT_RATE:
        state->tarot_rate = 4.0f * extra;
        break;
    case VOUCHER_PLANET_RATE:
        state->planet_rate = 4.0f * extra;
        break;
    case VOUCHER_EDITION_RATE:
        state->edition_rate = extra;
        break;
    case VOUCHER_PLAYING_RATE:
        state->playing_card_rate = extra;
        break;
    case VOUCHER_CONSUMABLE_SLOT:
        state->consumable_slots++;
        break;
    case VOUCHER_DISCOUNT:
        state->discount_percent = (uint8_t)extra;
        reprice_all_cards(state);
        break;
    case VOUCHER_REROLL:
        state->reroll_base = state->reroll_base > (uint8_t)extra ? (uint8_t)(state->reroll_base - (uint8_t)extra) : 0;
        state->reroll_cost -= (int32_t)extra;
        if (state->reroll_cost < 0) state->reroll_cost = 0;
        break;
    case VOUCHER_INTEREST:
        state->interest_cap = (int16_t)extra;
        break;
    case VOUCHER_HANDS:
        state->hands_per_round += (uint8_t)extra;
        break;
    case VOUCHER_HAND_SIZE:
        state->hand_size++;
        state->base_hand_size++;
        break;
    case VOUCHER_DISCARDS:
        state->discards_per_round += (uint8_t)extra;
        break;
    case VOUCHER_JOKER_SLOT:
        state->joker_slots++;
        break;
    case VOUCHER_ANTE_HANDS:
        if (state->ante > 1) state->ante--;
        if (state->hands_per_round > (uint8_t)extra) state->hands_per_round -= (uint8_t)extra;
        break;
    case VOUCHER_ANTE_DISCARDS:
        if (state->ante > 1) state->ante--;
        if (state->discards_per_round > (uint8_t)extra) state->discards_per_round -= (uint8_t)extra;
        break;
    default:
        break;
    }
}

int redeem_voucher(State *state, uint8_t index) {
    if (!state || index >= state->shop_voucher_count) return ERR_ACTION;
    Card card = state->shop_vouchers[index];
    if (!can_afford(state, card.cost)) return ERR_ACTION;
    state->dollars -= card.cost;
    if (state->next_voucher_id == card.center_id) state->next_voucher_id = 0;
    mark_center_used(state, card.center_id);
    apply_voucher(state, card.center_id);
    ZONE_REMOVE(state->shop_vouchers, state->shop_voucher_count, index);
    return OK;
}

static uint8_t sorted_editionless_jokers(const State *state, uint8_t indices[MAX_JOKERS]) {
    uint8_t count = 0;
    for (uint8_t i = 0; i < state->joker_count; ++i)
        if (!(state->jokers[i].flags & CARD_DEBUFFED) && state->jokers[i].edition == EDITION_NONE) indices[count++] = i;
    sort_indices(state->jokers, indices, count);
    return count;
}

void apply_consumable(State *state, const Action *action, Card card) {
    uint8_t set = card_set(&card);
    HandType planet = planet_hand(card.center_id);
    if (planet < HAND_COUNT) {
        state->planet_usage_mask |= (uint16_t)(1u << planet);
        if (state->hand_levels[planet] < 255) state->hand_levels[planet]++;
        for (uint8_t j = 0; j < state->joker_count; ++j)
            if (state->jokers[j].center_id == CENTER_J_CONSTELLATION)
                state->jokers[j].state[0] = (state->jokers[j].state[0] > 100 ? state->jokers[j].state[0] : 100) + 10;
    } else if (card.center_id == CENTER_C_HERMIT) {
        int gain = state->dollars > 0 ? (state->dollars < 20 ? state->dollars : 20) : 0;
        state->dollars += gain;
    } else if (card.center_id == CENTER_C_TEMPERANCE) {
        int total = 0;
        for (uint8_t i = 0; i < state->joker_count; ++i) total += state->jokers[i].sell_cost;
        state->dollars += total > 50 ? 50 : total;
    } else if (card.center_id == CENTER_C_BLACK_HOLE) {
        for (uint8_t i = 0; i < HAND_COUNT; ++i)
            if (state->hand_levels[i] < 255) state->hand_levels[i]++;
    } else if (card.center_id == CENTER_C_SIGIL) {
        static const uint8_t suits[] = {SPADES, HEARTS, DIAMONDS, CLUBS};
        uint8_t suit = suits[(size_t)floor(pseudorandom(state, "sigil") * 4.0) % 4];
        for (uint8_t i = 0; i < state->hand_count; ++i) state->hand[i].suit = suit;
    } else if (card.center_id == CENTER_C_OUIJA) {
        uint8_t rank = (uint8_t)(2 + floor(pseudorandom(state, "ouija") * 13.0));
        for (uint8_t i = 0; i < state->hand_count; ++i) state->hand[i].rank = rank;
        if (state->hand_size > 1) state->hand_size--;
        if (state->base_hand_size > 1) state->base_hand_size--;
    } else if (card.center_id == CENTER_C_IMMOLATE) {
        Card shuffled[MAX_HAND];
        memcpy(shuffled, state->hand, (size_t)state->hand_count * sizeof(Card));
        shuffle(state, shuffled, state->hand_count, "immolate");
        uint8_t remove = state->hand_count < 5 ? state->hand_count : 5;
        uint16_t destroyed_ids[5];
        for (uint8_t i = 0; i < remove; ++i) destroyed_ids[i] = shuffled[i].sort_id;
        for (uint8_t i = remove; i-- > 0;) {
            for (uint8_t h = 0; h < state->hand_count; ++h) {
                if (state->hand[h].sort_id == destroyed_ids[i]) {
                    remove_hand_index(state, h);
                    break;
                }
            }
        }
        state->dollars += 20;
    } else if (card.center_id == CENTER_C_FAMILIAR || card.center_id == CENTER_C_GRIM ||
               card.center_id == CENTER_C_INCANTATION) {
        if (state->hand_count) remove_hand_index(state, random_sorted_hand_index(state, "random_destroy"));
        static const uint8_t face_ranks[] = {11, 12, 13};
        static const uint8_t ace_ranks[] = {14};
        static const uint8_t number_ranks[] = {2, 3, 4, 5, 6, 7, 8, 9, 10};
        const uint8_t *ranks = card.center_id == CENTER_C_FAMILIAR ? face_ranks
                               : card.center_id == CENTER_C_GRIM   ? ace_ranks
                                                                           : number_ranks;
        size_t rank_count = card.center_id == CENTER_C_FAMILIAR ? sizeof(face_ranks)
                            : card.center_id == CENTER_C_GRIM   ? sizeof(ace_ranks)
                                                                        : sizeof(number_ranks);
        uint8_t create_count = card.center_id == CENTER_C_FAMILIAR ? 3 : card.center_id == CENTER_C_GRIM ? 2 : 4;
        const char *stream = card.center_id == CENTER_C_FAMILIAR ? "familiar_create"
                             : card.center_id == CENTER_C_GRIM   ? "grim_create"
                                                                         : "incantation_create";
        add_spectral_cards(state, ranks, rank_count, create_count, stream);
    } else if (card.center_id == CENTER_C_CRYPTID && action->selection_count) {
        Card source = state->hand[action->selection[0]];
        for (uint8_t i = 0; i < 2; ++i) {
            if (!can_add_playing_cards(state, 1)) break;
            Card copy = source;
            copy.sort_id = ++state->next_sort_id;
            int added = 0;
            if ((state->phase == PHASE_SELECTING_HAND || state->phase == PHASE_PACK_OPENING) &&
                state->hand_count < MAX_HAND)
                state->hand[state->hand_count++] = copy, added = 1;
            else if (state->deck_count < MAX_DECK)
                state->deck[state->deck_count++] = copy, added = 1;
            if (added) playing_card_added(state, 1);
        }
        if (state->phase == PHASE_SELECTING_HAND) sort_hand_desc(state);
    } else if (card.center_id == CENTER_C_AURA && action->selection_count) {
        double roll = pseudorandom(state, "aura");
        uint8_t edition = roll > 0.85 ? EDITION_POLYCHROME : roll > 0.50 ? EDITION_HOLO : EDITION_FOIL;
        state->hand[action->selection[0]].edition = edition;
    } else if (card.center_id == CENTER_C_ECTOPLASM) {
        uint8_t eligible[MAX_JOKERS];
        uint8_t count = sorted_editionless_jokers(state, eligible);
        if (count) {
            size_t pick = (size_t)floor(pseudorandom(state, "ectoplasm") * count);
            state->jokers[eligible[pick]].edition = EDITION_NEGATIVE;
            price_card(state, &state->jokers[eligible[pick]]);
            if (state->joker_slots < UINT8_MAX) state->joker_slots++;
            uint8_t penalty = state->ecto_penalty ? state->ecto_penalty : 1;
            if (state->hand_size > penalty)
                state->hand_size -= penalty;
            else
                state->hand_size = 1;
            if (state->base_hand_size > penalty)
                state->base_hand_size -= penalty;
            else
                state->base_hand_size = 1;
            state->ecto_penalty = penalty < UINT8_MAX ? (uint8_t)(penalty + 1) : UINT8_MAX;
        }
    } else if (card.center_id == CENTER_C_HEX) {
        uint8_t eligible[MAX_JOKERS];
        uint8_t count = sorted_editionless_jokers(state, eligible);
        if (count) {
            size_t pick = (size_t)floor(pseudorandom(state, "hex") * count);
            uint8_t target = eligible[pick];
            state->jokers[target].edition = EDITION_POLYCHROME;
            price_card(state, &state->jokers[target]);
            for (uint8_t i = state->joker_count; i-- > 0;) {
                if (i != target && !(state->jokers[i].flags & CARD_ETERNAL)) remove_joker_at(state, i);
            }
        }
    } else if (card.center_id == CENTER_C_WHEEL_OF_FORTUNE) {
        uint8_t eligible[MAX_JOKERS];
        uint8_t count = sorted_editionless_jokers(state, eligible);
        if (count && pseudorandom(state, "wheel_of_fortune") < probability_normal(state, 0.25)) {
            size_t pick = (size_t)floor(pseudorandom(state, "wheel_of_fortune") * count);
            double edition = pseudorandom(state, "wheel_of_fortune");
            state->jokers[eligible[pick]].edition = edition > 0.85 ? EDITION_POLYCHROME
                                               : edition > 0.50 ? EDITION_HOLO : EDITION_FOIL;
            price_card(state, &state->jokers[eligible[pick]]);
        }
    } else if (card.center_id == CENTER_C_FOOL) {
        if (state->last_tarot_planet && state->last_tarot_planet != CENTER_C_FOOL) {
            uint8_t saved_slots = state->consumable_slots;
            if (action->type == ACTION_USE_CONSUMABLE && state->consumable_slots < UINT8_MAX) state->consumable_slots++;
            (void)add_specific_consumable(state, state->last_tarot_planet);
            state->consumable_slots = saved_slots;
        }
    } else if (card.center_id == CENTER_C_EMPEROR) {
        uint8_t saved_slots = state->consumable_slots;
        if (action->type == ACTION_USE_CONSUMABLE && state->consumable_slots < UINT8_MAX) state->consumable_slots++;
        uint8_t room = state->consumable_slots - state->consumable_count;
        if (room) (void)add_pooled_consumable(state, SET_TAROT, "emp", 0);
        if (room > 1) (void)add_pooled_consumable(state, SET_TAROT, "emp", 0);
        state->consumable_slots = saved_slots;
    } else if (card.center_id == CENTER_C_HIGH_PRIESTESS) {
        uint8_t saved_slots = state->consumable_slots;
        if (action->type == ACTION_USE_CONSUMABLE && state->consumable_slots < UINT8_MAX) state->consumable_slots++;
        uint8_t room = state->consumable_slots - state->consumable_count;
        if (room) (void)add_pooled_consumable(state, SET_PLANET, "pri", 0);
        if (room > 1) (void)add_pooled_consumable(state, SET_PLANET, "pri", 0);
        state->consumable_slots = saved_slots;
    } else if (card.center_id == CENTER_C_JUDGEMENT) {
        if (state->joker_count < state->joker_slots && state->joker_count < MAX_JOKERS) {
            Card joker = create_pooled_card(state, SET_JOKER, "jud", 0);
            state->jokers[state->joker_count++] = joker;
            joker_added(state, &state->jokers[state->joker_count - 1]);
            mark_center_used(state, joker.center_id);
        }
    } else if (card.center_id == CENTER_C_DEATH && action->selection_count >= 2) {
        uint8_t left = action->selection[0], right = action->selection[1];
        if (left < state->hand_count && right < state->hand_count && left != right) {
            Card source = state->hand[right];
            Card *target = &state->hand[left];
            target->center_id = source.center_id;
            target->suit = source.suit;
            target->rank = source.rank;
            target->enhancement = source.enhancement;
            target->edition = source.edition;
            target->seal = source.seal;
            target->perma_bonus = source.perma_bonus;
            refresh_card_debuff(state, target);
            target->cost = source.cost;
            target->sell_cost = source.sell_cost;
        }
    } else if (card.center_id == CENTER_C_HANGED_MAN && action->selection_count) {
        uint8_t glass = 0;
        for (uint8_t i = 0; i < action->selection_count; ++i)
            if (action->selection[i] < state->hand_count && state->hand[action->selection[i]].enhancement == ENHANCEMENT_GLASS) glass++;
        if (glass)
            for (uint8_t j = 0; j < state->joker_count; ++j)
                if (!(state->jokers[j].flags & CARD_DEBUFFED) && state->jokers[j].center_id == CENTER_J_GLASS)
                    state->jokers[j].state[0] = (state->jokers[j].state[0] > 100 ? state->jokers[j].state[0] : 100) + glass * 75;
        for (uint8_t i = action->selection_count; i-- > 0;) {
            if (action->selection[i] < state->hand_count) remove_hand_index(state, action->selection[i]);
        }
    } else if (card.center_id == CENTER_C_WRAITH) {
        (void)add_joker_rarity(state, 3, "wra", 0);
        state->dollars = 0;
    } else if (card.center_id == CENTER_C_SOUL) {
        (void)add_joker_rarity(state, 4, "sou", 1);
    } else if (card.center_id == CENTER_C_ANKH && state->joker_count) {
        uint8_t order[MAX_JOKERS];
        for (uint8_t i = 0; i < state->joker_count; ++i) order[i] = i;
        sort_indices(state->jokers, order, state->joker_count);
        size_t pick = (size_t)floor(pseudorandom(state, "ankh_choice") * state->joker_count);
        Card copy = state->jokers[order[pick]];
        for (uint8_t i = state->joker_count; i-- > 0;) {
            if (i != order[pick] && !(state->jokers[i].flags & CARD_ETERNAL)) remove_joker_at(state, i);
        }
        if (state->joker_count < state->joker_slots && state->joker_count < MAX_JOKERS) {
            copy.sort_id = ++state->next_sort_id;
            if (copy.edition == EDITION_NEGATIVE) copy.edition = EDITION_NONE;
            price_card(state, &copy);
            state->jokers[state->joker_count++] = copy;
            joker_added(state, &state->jokers[state->joker_count - 1]);
        }
    } else if (action->selection_count) {
        const CenterDefinition *definition = &centers[card.center_id];
        for (uint8_t i = 0; i < action->selection_count; ++i) {
            uint8_t index = action->selection[i];
            Card *target = &state->hand[index];
            if (definition->target_effect == TARGET_ENHANCEMENT)
                target->enhancement = definition->target_value;
            else if (definition->target_effect == TARGET_SUIT)
                target->suit = definition->target_value;
            else if (definition->target_effect == TARGET_SEAL)
                target->seal = definition->target_value;
            else if (definition->target_effect == TARGET_RANK_UP)
                target->rank = target->rank == 14 ? 2 : target->rank < 14 ? target->rank + 1 : target->rank;
            refresh_card_debuff(state, target);
        }
    }
    if (set == SET_TAROT) {
        state->tarots_used++;
        state->last_tarot_planet = card.center_id;
    } else if (set == SET_PLANET)
        state->last_tarot_planet = card.center_id;
}

void use_consumable(State *state, const Action *action) {
    Card card = state->consumables[action->primary];
    apply_consumable(state, action, card);
    consumable_removed(state, &card);
    ZONE_REMOVE(state->consumables, state->consumable_count, action->primary);
}

void sell_consumable(State *state, uint8_t index) {
    Card card = state->consumables[index];
    state->dollars += card.sell_cost;
    consumable_removed(state, &card);
    for (uint8_t j = 0; j < state->joker_count; ++j)
        if (!(state->jokers[j].flags & CARD_DEBUFFED) && state->jokers[j].center_id == CENTER_J_CAMPFIRE)
            state->jokers[j].state[0] = (state->jokers[j].state[0] > 100 ? state->jokers[j].state[0] : 100) + 25;
    ZONE_REMOVE(state->consumables, state->consumable_count, index);
}

static int open_pack_center(State *state, uint16_t center_id) {
    const CenterDefinition *definition = &centers[center_id];
    uint8_t kind = definition->kind;
    state->pack_count = 0;
    state->pack_choices = definition->pack_choose;
    state->pack_kind = kind;
    if (state->phase != PHASE_PACK_OPENING) state->shop_return_phase = state->phase;
    uint8_t set = SET_DEFAULT;
    const char *append = "sta";
    switch (kind) {
    case PACK_TAROT: set = SET_TAROT; append = "ar1"; break;
    case PACK_PLANET: set = SET_PLANET; append = "pl1"; break;
    case PACK_SPECTRAL: set = SET_SPECTRAL; append = "spe"; break;
    case PACK_JOKER: set = SET_JOKER; append = "buf"; break;
    default: break;
    }
    uint16_t temporary[MAX_PACK_CARDS];
    uint8_t temporary_count = 0;
    for (uint8_t i = 0; i < definition->pack_extra; ++i) {
        if (state->pack_count >= MAX_PACK_CARDS) return ERR_CAPACITY;
        Card card = {0};
        uint16_t telescope = 0;
        if (kind == PACK_PLANET && i == 0 && center_used(state, CENTER_V_TELESCOPE)) {
            uint8_t hand = most_played_hand(state);
            if (state->hand_plays[hand]) telescope = planet_center(hand);
        }
        uint8_t card_set = set;
        const char *card_append = append;
        if (kind == PACK_TAROT && center_used(state, CENTER_V_OMEN_GLOBE) &&
            pseudorandom(state, "omen_globe") > 0.8) {
            card_set = SET_SPECTRAL;
            card_append = "ar2";
        }
        if (telescope) {
            card.center_id = telescope;
            card.sort_id = ++state->next_sort_id;
            card.cost = centers[telescope].cost;
            card.sell_cost = card.cost / 2 > 0 ? card.cost / 2 : 1;
        } else if (card_set == SET_DEFAULT) {
            char stream[32];
            key_with_u64(stream, sizeof(stream), "stdset", state->ante);
            int enhanced = pseudorandom(state, stream) > 0.6;
            key_with_u64(stream, sizeof(stream), "frontsta", state->ante);
            PlayingCardDefinition playing =
                playing_card((uint8_t)(pseudorandom(state, stream) * 52.0));
            card.suit = playing.suit;
            card.rank = playing.rank;
            card.center_id = CENTER_C_BASE;
            if (enhanced) {
                size_t count = 0;
                const uint16_t *pool = center_pool(SET_ENHANCED, &count);
                key_with_u64(stream, sizeof(stream), "Enhancedsta", state->ante);
                size_t index = (size_t)(pseudorandom(state, stream) * count);
                card.center_id = pool[index];
                card.enhancement = (uint8_t)(index + 1);
            }
            key_with_u64(stream, sizeof(stream), "standard_edition", state->ante);
            double edition = pseudorandom(state, stream);
            if (edition > 1.0 - 0.012 * state->edition_rate)
                card.edition = EDITION_POLYCHROME;
            else if (edition > 1.0 - 0.04 * state->edition_rate)
                card.edition = EDITION_HOLO;
            else if (edition > 1.0 - 0.08 * state->edition_rate)
                card.edition = EDITION_FOIL;
            key_with_u64(stream, sizeof(stream), "stdseal", state->ante);
            if (pseudorandom(state, stream) > 0.8) {
                key_with_u64(stream, sizeof(stream), "stdsealtype", state->ante);
                double seal = pseudorandom(state, stream);
                card.seal = seal > 0.75 ? SEAL_RED
                    : seal > 0.5 ? SEAL_BLUE
                    : seal > 0.25 ? SEAL_GOLD
                    : SEAL_PURPLE;
            }
            card.sort_id = ++state->next_sort_id;
            card.cost = card.sell_cost = 1;
        } else
            card = create_pooled_card(state, card_set, card_append, 1);
        state->pack_cards[state->pack_count++] = card;
        if (card.center_id != CENTER_C_BASE && !center_used(state, card.center_id)) {
            mark_center_used(state, card.center_id);
            temporary[temporary_count++] = card.center_id;
        }
    }
    for (uint8_t i = 0; i < temporary_count; ++i) unmark_center_used(state, temporary[i]);
    for (uint8_t start = 0; start < state->joker_count; ++start) {
        uint8_t source = resolved_joker_source(state, start);
        int hallucination = source != UINT8_MAX &&
                            state->jokers[source].center_id == CENTER_J_HALLUCINATION;
        if (hallucination && state->consumable_count < state->consumable_slots &&
            state->consumable_count < MAX_CONSUMABLES) {
            double chance = 0.5;
            for (uint8_t j = 0; j < state->joker_count; ++j)
                if (!(state->jokers[j].flags & CARD_DEBUFFED) &&
                    state->jokers[j].center_id == CENTER_J_OOPS)
                    chance *= 2.0;
            char stream[32];
            key_with_u64(stream, sizeof(stream), "halu", state->ante);
            if (pseudorandom(state, stream) < chance) {
                Card tarot = create_pooled_card(state, SET_TAROT, "hal", 0);
                state->consumables[state->consumable_count++] = tarot;
                consumable_added(state, &tarot);
            }
        }
    }
    if (kind == PACK_TAROT || kind == PACK_SPECTRAL) draw_to_hand(state);
    state->phase = PHASE_PACK_OPENING;
    return OK;
}

int open_booster(State *state, uint8_t index) {
    assert(state);
    if (index >= state->shop_booster_count) return ERR_ACTION;
    Card booster = state->shop_boosters[index];
    if (!can_afford(state, booster.cost)) return ERR_ACTION;
    state->dollars -= booster.cost;
    ZONE_REMOVE(state->shop_boosters, state->shop_booster_count, index);
    return open_pack_center(state, booster.center_id);
}

int open_free_pack(State *state, uint16_t center_id) {
    assert(state);
    if (center_id >= CENTER_COUNT || centers[center_id].set != SET_BOOSTER) return ERR_ACTION;
    if (state->phase == PHASE_PACK_OPENING) {
        if (state->pending_free_pack_id) return ERR_CAPACITY;
        state->pending_free_pack_id = center_id;
        return OK;
    }
    return open_pack_center(state, center_id);
}

static void finish_pack(State *state) {
    uint8_t return_hand = state->pack_kind == PACK_TAROT || state->pack_kind == PACK_SPECTRAL;
    state->pack_count = 0;
    state->pack_choices = 0;
    state->pack_kind = 0;
    if (return_hand && state->hand_count) {
        uint8_t returning = state->hand_count;
        if ((size_t)state->deck_count + returning > MAX_DECK)
            returning = (uint8_t)(MAX_DECK - state->deck_count);
        memmove(&state->deck[returning], &state->deck[0], state->deck_count * sizeof(Card));
        for (uint8_t i = 0; i < returning; ++i)
            state->deck[i] = state->hand[state->hand_count - 1 - i];
        state->deck_count += returning;
        state->hand_count = 0;
    }
    if (state->pending_free_pack_id) {
        uint16_t next = state->pending_free_pack_id;
        state->pending_free_pack_id = 0;
        if (open_pack_center(state, next) == OK) return;
    }
    state->phase = state->shop_return_phase;
}

int pick_pack_card(State *state, uint8_t index) {
    assert(state);
    if (state->phase != PHASE_PACK_OPENING || index >= state->pack_count) return ERR_ACTION;
    Card card = state->pack_cards[index];
    uint8_t set = card_set(&card);
    if (!can_own(state, set, card.edition)) return ERR_CAPACITY;
    if (!add_owned_card(state, card, set)) return ERR_ACTION;
    complete_pack_pick(state, index);
    return OK;
}

static void complete_pack_pick(State *state, uint8_t index) {
    ZONE_REMOVE(state->pack_cards, state->pack_count, index);
    if (--state->pack_choices == 0 || state->pack_count == 0) finish_pack(state);
}

int skip_pack(State *state) {
    assert(state);
    if (state->phase != PHASE_PACK_OPENING) return ERR_ACTION;
    finish_pack(state);
    return OK;
}

void reset_round_rerolls(State *state) {
    uint8_t chaos = 0;
    for (uint8_t j = 0; j < state->joker_count; ++j)
        if (!(state->jokers[j].flags & CARD_DEBUFFED) && state->jokers[j].center_id == CENTER_J_CHAOS && chaos < UINT8_MAX)
            chaos++;
    state->free_rerolls = chaos;
    state->reroll_increase = 0;
    state->reroll_cost = chaos ? 0 : (state->tag_d_six_active ? 0 : state->reroll_base);
}

static const uint8_t lua_hand_order[] = {
    FLUSH_HOUSE, FULL_HOUSE, FLUSH,      PAIR,           HIGH_CARD,       STRAIGHT_FLUSH,
    STRAIGHT,    TWO_PAIR,   FLUSH_FIVE, FIVE_OF_A_KIND, THREE_OF_A_KIND, FOUR_OF_A_KIND,
};

static int hand_visible(const State *state, uint8_t hand) {
    return hand >= STRAIGHT_FLUSH || state->hand_plays[hand] != 0;
}

static uint8_t visible_hands(const State *state, int excluded, uint8_t out[HAND_COUNT]) {
    uint8_t count = 0;
    for (size_t i = 0; i < sizeof(lua_hand_order); ++i)
        if (hand_visible(state, lua_hand_order[i]) && (int)lua_hand_order[i] != excluded)
            out[count++] = lua_hand_order[i];
    return count;
}

uint8_t choose_to_do_hand(State *state, int excluded) {
    uint8_t candidates[HAND_COUNT];
    size_t count = visible_hands(state, excluded, candidates);
    size_t index = (size_t)floor(pseudorandom(state, "to_do") * count);
    return candidates[index];
}

void choose_orbital_hands(State *state) {
    uint8_t visible[HAND_COUNT];
    uint8_t count = visible_hands(state, -1, visible);
    for (uint8_t blind = 0; blind < 3; ++blind) {
        size_t pick = (size_t)floor(pseudorandom(state, "orbital") * count);
        state->orbital_hands[blind] = visible[pick];
    }
}

static void initialize_joker_card(State *state, Card *card) {
    if (card->center_id == CENTER_J_TODO_LIST && card->state[2] == 0) {
        card->state[1] = choose_to_do_hand(state, -1);
        card->state[2] = 1;
    }
}

static uint8_t choose_tag(State *state) {
    static const uint8_t tags[] = {
        TAG_TAG_UNCOMMON,   TAG_TAG_RARE,       TAG_TAG_NEGATIVE, TAG_TAG_FOIL,    TAG_TAG_HOLO,
        TAG_TAG_POLYCHROME, TAG_TAG_INVESTMENT, TAG_TAG_VOUCHER,  TAG_TAG_BOSS,    TAG_TAG_STANDARD,
        TAG_TAG_CHARM,      TAG_TAG_METEOR,     TAG_TAG_BUFFOON,  TAG_TAG_HANDY,   TAG_TAG_GARBAGE,
        TAG_TAG_ETHEREAL,   TAG_TAG_COUPON,     TAG_TAG_DOUBLE,   TAG_TAG_JUGGLE,  TAG_TAG_D_SIX,
        TAG_TAG_TOP_UP,     TAG_TAG_SKIP,       TAG_TAG_ORBITAL,  TAG_TAG_ECONOMY,
    };
    static const uint8_t minimum[] = {
        1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1,
    };
    uint8_t available[sizeof(tags)];
    uint8_t ante = state->ante;
    for (size_t i = 0; i < sizeof(tags); ++i) {
        available[i] = ante >= minimum[i];
    }
    char stream[32];
    char base[32];
    for (unsigned attempt = 0; attempt <= 20; ++attempt) {
        key_with_u64(base, sizeof(base), "Tag", state->ante);
        key_resample(stream, sizeof(stream), base, attempt);
        size_t index = (size_t)floor(pseudorandom(state, stream) * sizeof(tags));
        if (available[index]) return tags[index];
    }
    return TAG_TAG_HANDY;
}

void assign_blind_tags(State *state) {
    state->blind_tags[0] = choose_tag(state);
    state->blind_tags[1] = choose_tag(state);
}

void apply_skip_tag(State *state, uint8_t tag) {
    state->active_tag = tag;
    switch (tag) {
    case TAG_TAG_UNCOMMON:
    case TAG_TAG_RARE: {
        uint8_t rarity = tag == TAG_TAG_RARE ? 3 : 2;
        if (state->tag_force_rarity == rarity) {
            if (state->tag_force_rarity_count < UINT8_MAX) state->tag_force_rarity_count++;
        } else {
            state->tag_force_rarity = rarity;
            state->tag_force_rarity_count = 1;
        }
        break;
    }
    case TAG_TAG_FOIL:
    case TAG_TAG_HOLO:
    case TAG_TAG_POLYCHROME:
    case TAG_TAG_NEGATIVE: {
        uint8_t edition;
        switch (tag) {
        case TAG_TAG_FOIL: edition = EDITION_FOIL; break;
        case TAG_TAG_HOLO: edition = EDITION_HOLO; break;
        case TAG_TAG_POLYCHROME: edition = EDITION_POLYCHROME; break;
        default: edition = EDITION_NEGATIVE; break;
        }
        if (state->tag_force_edition == edition) {
            if (state->tag_force_edition_count < UINT8_MAX) state->tag_force_edition_count++;
        } else {
            state->tag_force_edition = edition;
            state->tag_force_edition_count = 1;
        }
        break;
    }
    case TAG_TAG_INVESTMENT:
        if (state->tag_investment_pending < UINT8_MAX) state->tag_investment_pending++;
        break;
    case TAG_TAG_VOUCHER:
        if (state->tag_voucher_pending < UINT8_MAX) state->tag_voucher_pending++;
        break;
    case TAG_TAG_DOUBLE:
        state->double_tag = 1;
        break;
    case TAG_TAG_ECONOMY: {
        int gain = state->dollars < 40 ? state->dollars : 40;
        state->dollars += gain;
        break;
    }
    case TAG_TAG_GARBAGE:
        state->dollars += state->unused_discards;
        state->unused_discards = 0;
        break;
    case TAG_TAG_HANDY:
        state->dollars += (int32_t)state->run_hands_played;
        break;
    case TAG_TAG_SKIP:
        state->dollars += state->skips * 5;
        break;
    case TAG_TAG_TOP_UP:
        (void)add_joker_rarity(state, 1, "top", 0);
        (void)add_joker_rarity(state, 1, "top", 0);
        break;
    case TAG_TAG_ORBITAL: {
        uint8_t hand = state->orbital_hands[state->blind_on_deck];
        state->hand_levels[hand] = (uint8_t)(state->hand_levels[hand] > 252 ? 255 : state->hand_levels[hand] + 3);
        break;
    }
    case TAG_TAG_JUGGLE:
        state->tag_hand_bonus = (uint8_t)(state->tag_hand_bonus > UINT8_MAX - 3 ? UINT8_MAX : state->tag_hand_bonus + 3);
        break;
    case TAG_TAG_D_SIX:
        state->tag_d_six_pending = 1;
        break;
    case TAG_TAG_COUPON:
        state->tag_coupon_pending = 1;
        break;
    case TAG_TAG_BOSS:
        state->boss_rerolled = 1;
        state->next_boss_id = choose_boss(state);
        break;
    case TAG_TAG_STANDARD:
        (void)open_free_pack(state, CENTER_P_STANDARD_MEGA_1);
        break;
    case TAG_TAG_CHARM:
        (void)open_free_pack(state, CENTER_P_ARCANA_MEGA_1);
        break;
    case TAG_TAG_METEOR:
        (void)open_free_pack(state, CENTER_P_CELESTIAL_MEGA_1);
        break;
    case TAG_TAG_BUFFOON:
        (void)open_free_pack(state, CENTER_P_BUFFOON_MEGA_1);
        break;
    case TAG_TAG_ETHEREAL:
        (void)open_free_pack(state, CENTER_P_SPECTRAL_NORMAL_1);
        break;
    default:
        break;
    }
}

static int add_pooled_consumable(State *state, uint8_t set, const char *append, uint8_t edition) {
    Card card = create_pooled_card(state, set, append, 0);
    card.edition = edition;
    if (state->consumable_count >= MAX_CONSUMABLES || (state->consumable_count >= state->consumable_slots && card.edition != EDITION_NEGATIVE))
        return 0;
    state->consumables[state->consumable_count++] = card;
    consumable_added(state, &card);
    return 1;
}

static int add_specific_consumable(State *state, uint16_t center_id) {
    if (state->consumable_count >= state->consumable_slots || state->consumable_count >= MAX_CONSUMABLES) return 0;
    Card card = {0};
    card.center_id = center_id;
    card.sort_id = ++state->next_sort_id;
    card.cost = centers[center_id].cost;
    card.sell_cost = card.cost / 2 > 0 ? card.cost / 2 : 1;
    state->consumables[state->consumable_count++] = card;
    consumable_added(state, &card);
    return 1;
}

static void joker_added(State *state, const Card *joker) {
    Card *mutable_joker = (Card *)joker;
    if (joker->edition == EDITION_NEGATIVE && state->joker_slots < UINT8_MAX) state->joker_slots++;
    if (joker->center_id == CENTER_J_TO_THE_MOON) state->interest_amount++;
    initialize_joker_card(state, mutable_joker);
    if (joker->center_id == CENTER_J_IDOL) {
        int copied_target = 0;
        for (uint8_t i = 0; i + 1 < state->joker_count; ++i)
            if (&state->jokers[i] != joker && !(state->jokers[i].flags & CARD_DEBUFFED) && state->jokers[i].center_id == CENTER_J_IDOL) {
                mutable_joker->state[0] = state->jokers[i].state[0];
                mutable_joker->state[1] = state->jokers[i].state[1];
                copied_target = 1;
                break;
            }
        if (!copied_target && state->idol_rank) {
            mutable_joker->state[0] = state->idol_rank;
            mutable_joker->state[1] = state->idol_suit;
        }
    }
    if (joker->center_id == CENTER_J_MAIL && state->mail_rank) mutable_joker->state[1] = state->mail_rank;
    if (joker->center_id == CENTER_J_CASTLE && state->castle_suit < 4) mutable_joker->state[1] = state->castle_suit;
    if (joker->center_id == CENTER_J_LOYALTY_CARD && joker->state[2] == 0) {
        mutable_joker->state[1] = (int32_t)state->run_hands_played;
        mutable_joker->state[2] = 1;
    }
    uint8_t old_hand_size = state->hand_size;
    if (joker->center_id == CENTER_J_JUGGLER) {
        state->hand_size++;
    } else if (joker->center_id == CENTER_J_MERRY_ANDY) {
        if (state->hand_size > 1) state->hand_size--;
        state->discards_left = state->discards_left > UINT8_MAX - 3 ? UINT8_MAX : (uint8_t)(state->discards_left + 3);
    } else if (joker->center_id == CENTER_J_DRUNKARD) {
        if (state->discards_left < UINT8_MAX) state->discards_left++;
    } else if (joker->center_id == CENTER_J_TROUBADOUR) {
        state->hand_size = state->hand_size > UINT8_MAX - 2 ? UINT8_MAX : (uint8_t)(state->hand_size + 2);
    } else if (joker->center_id == CENTER_J_TURTLE_BEAN) {
        if (!mutable_joker->state[0]) mutable_joker->state[0] = 5;
        uint8_t bonus = (uint8_t)mutable_joker->state[0];
        state->hand_size = state->hand_size > UINT8_MAX - bonus ? UINT8_MAX : (uint8_t)(state->hand_size + bonus);
    }
    if (state->phase == PHASE_SELECTING_HAND && state->hand_size > old_hand_size) draw_to_hand(state);
    if (joker->center_id == CENTER_J_STUNTMAN) {
        state->hand_size = state->hand_size > 2 ? state->hand_size - 2 : 1;
        state->base_hand_size = state->base_hand_size > 2 ? state->base_hand_size - 2 : 1;
    }
    if (joker->center_id == CENTER_J_CHAOS) {
        if (state->free_rerolls < UINT8_MAX) state->free_rerolls++;
        state->reroll_cost = 0;
    }
    refresh_joker_cache(state);
}

static void joker_removed(State *state, const Card *joker) {
    uint8_t matching = 0;
    for (uint8_t i = 0; i < state->joker_count; ++i)
        if (state->jokers[i].center_id == joker->center_id) matching++;
    if (matching <= 1) unmark_center_used(state, joker->center_id);
    if (joker->edition == EDITION_NEGATIVE && state->joker_slots > 0) state->joker_slots--;
    if (joker->center_id == CENTER_J_TO_THE_MOON && state->interest_amount > 0) state->interest_amount--;
    if (joker->center_id == CENTER_J_STUNTMAN) {
        state->hand_size += 2;
        state->base_hand_size += 2;
    }
    if (state->phase == PHASE_SELECTING_HAND) {
        if (joker->center_id == CENTER_J_JUGGLER) {
            if (state->hand_size > 1) state->hand_size--;
        } else if (joker->center_id == CENTER_J_MERRY_ANDY) {
            if (state->hand_size < UINT8_MAX) state->hand_size++;
            state->discards_left = state->discards_left >= 3 ? (uint8_t)(state->discards_left - 3) : 0;
        } else if (joker->center_id == CENTER_J_DRUNKARD) {
            if (state->discards_left) state->discards_left--;
        } else if (joker->center_id == CENTER_J_TROUBADOUR) {
            state->hand_size = state->hand_size > 2 ? state->hand_size - 2 : 1;
        } else if (joker->center_id == CENTER_J_TURTLE_BEAN) {
            uint8_t bonus = joker->state[0] > 0 ? (uint8_t)joker->state[0] : 5;
            state->hand_size = state->hand_size > bonus ? (uint8_t)(state->hand_size - bonus) : 1;
        }
    }
    if (joker->center_id == CENTER_J_CHAOS) {
        if (state->free_rerolls) state->free_rerolls--;
        state->reroll_cost = state->free_rerolls ? 0 : (state->tag_d_six_active ? 0 : state->reroll_base) + state->reroll_increase;
    }
}

static void consumable_added(State *state, const Card *card) {
    mark_center_used(state, card->center_id);
    if (card->edition == EDITION_NEGATIVE && state->consumable_slots < UINT8_MAX) state->consumable_slots++;
}

static void consumable_removed(State *state, const Card *card) {
    if (card->edition == EDITION_NEGATIVE && state->consumable_slots > 0) state->consumable_slots--;
    uint8_t matching = 0;
    for (uint8_t i = 0; i < state->consumable_count; ++i)
        if (state->consumables[i].center_id == card->center_id) matching++;
    if (matching <= 1) unmark_center_used(state, card->center_id);
}

static void playing_card_added(State *state, uint8_t count) {
    for (uint8_t j = 0; j < state->joker_count; ++j)
        if (!(state->jokers[j].flags & CARD_DEBUFFED) && state->jokers[j].center_id == CENTER_J_HOLOGRAM)
            state->jokers[j].state[0] = (state->jokers[j].state[0] > 100 ? state->jokers[j].state[0] : 100) + count * 25;
}

static int add_joker_rarity(State *state, uint8_t rarity, const char *append, int legendary) {
    if (state->joker_count >= state->joker_slots || state->joker_count >= MAX_JOKERS) return 0;
    size_t count = 0;
    const uint16_t *pool = joker_pool(rarity, &count);
    if (!pool || !count) return 0;
    uint16_t center_id = 0;
    for (unsigned attempt = 0; attempt < 64; ++attempt) {
        char stream[64];
        char base[64];
        KeyBuilder key;
        key_begin(&key, base, sizeof(base));
        key_append(&key, "Joker");
        key_append_u64(&key, rarity);
        if (!legendary) {
            key_append(&key, append);
            key_append_u64(&key, state->ante);
        }
        key_resample(stream, sizeof(stream), base, attempt);
        size_t index = (size_t)floor(pseudorandom(state, stream) * count);
        uint16_t candidate = pool[index];
        int ring_master = joker_active(state, CENTER_J_RING_MASTER);
        if (centers[candidate].base_available && (!center_used(state, candidate) || ring_master)) {
            center_id = candidate;
            break;
        }
    }
    if (!center_id) return 0;
    Card card = {0};
    card.center_id = center_id;
    card.sort_id = ++state->next_sort_id;
    initialize_joker_card(state, &card);
    roll_joker_edition(state, append, &card);
    price_card(state, &card);
    state->jokers[state->joker_count++] = card;
    joker_added(state, &state->jokers[state->joker_count - 1]);
    mark_center_used(state, card.center_id);
    return 1;
}

void notify_removed_playing_cards(State *state, const Card *cards, uint8_t count, int shattered) {
    uint8_t faces = 0, glass = 0;
    for (uint8_t i = 0; i < count; ++i) {
        if (cards[i].rank >= 11 && cards[i].rank <= 13 && !(cards[i].flags & CARD_DEBUFFED)) faces++;
        if (shattered && cards[i].enhancement == ENHANCEMENT_GLASS && !(cards[i].flags & CARD_DEBUFFED)) glass++;
    }
    if (!faces && !glass) return;
    for (uint8_t j = 0; j < state->joker_count; ++j) {
        Card *joker = &state->jokers[j];
        if (joker->flags & CARD_DEBUFFED) continue;
        if (joker->center_id == CENTER_J_CAINO && faces)
            joker->state[0] = (joker->state[0] > 100 ? joker->state[0] : 100) + faces * 100;
        else if (joker->center_id == CENTER_J_GLASS && glass)
            joker->state[0] = (joker->state[0] > 100 ? joker->state[0] : 100) + glass * 75;
    }
}

static void remove_hand_index(State *state, uint8_t index) {
    notify_removed_playing_cards(state, &state->hand[index], 1, 0);
    ZONE_REMOVE(state->hand, state->hand_count, index);
}

Card random_playing_card(State *state, const char *stream) {
    Card card = {0};
    size_t front = (size_t)floor(pseudorandom(state, stream) * 52.0);
    card.center_id = CENTER_C_BASE;
    card.sort_id = ++state->next_sort_id;
    PlayingCardDefinition definition = playing_card((uint8_t)front);
    card.suit = definition.suit;
    card.rank = definition.rank;
    card.cost = card.sell_cost = 1;
    return card;
}

static uint8_t random_sorted_hand_index(State *state, const char *stream) {
    uint8_t order[MAX_HAND];
    for (uint8_t i = 0; i < state->hand_count; ++i) order[i] = i;
    sort_indices(state->hand, order, state->hand_count);
    if (!state->hand_count) return 0;
    size_t index = (size_t)floor(pseudorandom(state, stream) * state->hand_count);
    return order[index];
}

typedef struct TargetCard {
    uint16_t sort_id;
    uint8_t rank;
    uint8_t suit;
} TargetCard;

static size_t pick_target_index(State *state, uint16_t count, const char *prefix) {
    char stream[32];
    KeyBuilder key;
    key_begin(&key, stream, sizeof(stream));
    key_append(&key, prefix);
    key_append_u64(&key, state->ante);
    size_t pick = (size_t)floor(pseudorandom(state, stream) * count);
    return pick < count ? pick : (size_t)(count - 1);
}

void reset_round_targets(State *state) {
    TargetCard candidates[MAX_DECK * 2 + MAX_HAND];
    TargetCard scratch[MAX_DECK * 2 + MAX_HAND];
    uint16_t count = 0;
    const Card *zones[] = {state->deck, state->hand, state->discard};
    const uint16_t counts[] = {state->deck_count, state->hand_count, state->discard_count};
    for (int zone = 0; zone < 3; ++zone)
        for (uint16_t i = 0; i < counts[zone]; ++i) {
            const Card *card = &zones[zone][i];
            if (card->enhancement != ENHANCEMENT_STONE)
                candidates[count++] = (TargetCard){card->sort_id, card->rank, card->suit};
        }
    radix_sort(candidates, scratch, count, sizeof(*candidates),
               offsetof(TargetCard, sort_id), sizeof(uint16_t));
    uint8_t idol_rank = 14, idol_suit = SPADES;
    if (count) {
        size_t pick = pick_target_index(state, count, "idol");
        idol_rank = candidates[pick].rank;
        idol_suit = candidates[pick].suit;
    }
    uint8_t mail_rank = 14, castle_suit = SPADES;
    if (count) {
        size_t mail = pick_target_index(state, count, "mail");
        size_t castle = pick_target_index(state, count, "cas");
        mail_rank = candidates[mail].rank;
        castle_suit = candidates[castle].suit;
    }
    state->idol_rank = idol_rank;
    state->idol_suit = idol_suit;
    state->mail_rank = mail_rank;
    state->castle_suit = castle_suit;
    for (uint8_t i = 0; i < state->joker_count; ++i) {
        if (state->jokers[i].flags & CARD_DEBUFFED) continue;
        if (state->jokers[i].center_id == CENTER_J_IDOL) {
            state->jokers[i].state[0] = idol_rank;
            state->jokers[i].state[1] = idol_suit;
        } else if (state->jokers[i].center_id == CENTER_J_MAIL)
            state->jokers[i].state[1] = mail_rank;
        else if (state->jokers[i].center_id == CENTER_J_CASTLE)
            state->jokers[i].state[1] = castle_suit;
    }
    uint8_t suits[4] = {SPADES, HEARTS, CLUBS, DIAMONDS};
    uint8_t choices[4], choice_count = 0;
    for (uint8_t i = 0; i < 4; ++i)
        if (suits[i] != state->ancient_suit) choices[choice_count++] = suits[i];
    size_t pick = pick_target_index(state, choice_count, "anc");
    state->ancient_suit = choices[pick];
}

static void add_spectral_cards(State *state, const uint8_t *ranks, size_t rank_count, uint8_t count, const char *stream) {
    static const uint8_t suits[] = {SPADES, HEARTS, DIAMONDS, CLUBS};
    static const uint8_t enhancements[] = {1, 2, 3, 4, 5, 7, 8};
    while (count--) {
        if (!can_add_playing_cards(state, 1)) break;
        size_t rank = rank_count > 1 ? (size_t)floor(pseudorandom(state, stream) * rank_count) : 0;
        size_t suit = (size_t)floor(pseudorandom(state, stream) * 4.0);
        size_t enhancement = (size_t)floor(pseudorandom(state, "spe_card") * (sizeof(enhancements) / sizeof(enhancements[0])));
        if (rank >= rank_count) rank = rank_count - 1;
        if (suit >= sizeof(suits)) suit = sizeof(suits) - 1;
        if (enhancement >= sizeof(enhancements)) enhancement = sizeof(enhancements) - 1;
        Card card = {
            .center_id = CENTER_C_BASE,
            .sort_id = ++state->next_sort_id,
            .rank = ranks[rank],
            .suit = suits[suit],
            .enhancement = enhancements[enhancement],
            .cost = 1,
            .sell_cost = 1,
        };
        if ((state->phase == PHASE_SELECTING_HAND || state->phase == PHASE_PACK_OPENING) &&
            state->hand_count < MAX_HAND)
            state->hand[state->hand_count++] = card;
        else if (state->deck_count < MAX_DECK)
            state->deck[state->deck_count++] = card;
        else
            continue;
        playing_card_added(state, 1);
    }
    if (state->phase == PHASE_SELECTING_HAND) sort_hand_desc(state);
}

static void build_starting_deck(State *state) {
    state->deck_count = 0;
    for (uint8_t front = 0; front < 52; ++front) {
        PlayingCardDefinition definition = playing_card(front);
        if (state->config.deck == CENTER_B_ABANDONED &&
            (definition.rank == 11 || definition.rank == 12 || definition.rank == 13)) continue;
        Card card = {.center_id = CENTER_C_BASE, .sort_id = ++state->next_sort_id,
                     .suit = definition.suit, .rank = definition.rank};
        if (state->config.deck == CENTER_B_CHECKERED) {
            if (card.suit == CLUBS) card.suit = SPADES;
            else if (card.suit == DIAMONDS) card.suit = HEARTS;
        }
        if (state->config.deck == CENTER_B_ERRATIC) {
            size_t pick = (size_t)floor(pseudorandom(state, "erratic") * 52.0);
            definition = playing_card((uint8_t)pick);
            card.suit = definition.suit;
            card.rank = definition.rank;
        }
        state->deck[state->deck_count++] = card;
    }
}

void draw_after_play(State *state) {
    uint8_t target = state->hand_size;
    if (!state->blind_disabled && state->blind_id == BLIND_BL_SERPENT) {
        target = state->hand_count + 3;
    }
    uint8_t first = state->hand_count;
    while (state->hand_count < target && state->deck_count > 0 && state->hand_count < MAX_HAND)
        state->hand[state->hand_count++] = state->deck[--state->deck_count];
    for (uint8_t i = first; i < state->hand_count; ++i)
        if (blind_debuffs_card(state, &state->hand[i])) state->hand[i].flags |= CARD_DEBUFFED;
    sort_hand_desc(state);
}

void apply_discard_effects(State *state, const Card *cards, uint8_t count, int first_discard, int hook) {
    if (!hook && first_discard) {
        for (uint8_t j = 0; j < state->joker_count; ++j)
            if (!(state->jokers[j].flags & CARD_DEBUFFED) && state->jokers[j].center_id == CENTER_J_BURNT) {
                uint8_t mask = 0;
                HandType hand = classify_hand(cards, count, &mask);
                if (state->hand_levels[hand] < 255) state->hand_levels[hand]++;
            }
    }
    int pareidolia = joker_active(state, CENTER_J_PAREIDOLIA);
    int face_count = 0, jacks = 0;
    for (uint8_t k = 0; k < count; ++k) {
        if (!(cards[k].flags & CARD_DEBUFFED)) {
            face_count += pareidolia || (cards[k].rank >= 11 && cards[k].rank <= 13);
            jacks += cards[k].rank == 11;
        }
    }
    Card removed[MAX_SELECTION] = {0};
    uint8_t removed_count = 0;
    for (uint8_t j = 0; j < state->joker_count; ++j) {
        Card *joker = &state->jokers[j];
        if (joker->flags & CARD_DEBUFFED) continue;
        if (joker->center_id == CENTER_J_FACELESS && face_count >= 3) {
            state->dollars += 5;
        } else if (joker->center_id == CENTER_J_MAIL) {
            for (uint8_t k = 0; k < count; ++k)
                if (!(cards[k].flags & CARD_DEBUFFED) && cards[k].rank == (uint8_t)joker->state[1]) state->dollars += 5;
        } else if (joker->center_id == CENTER_J_TRADING && first_discard && count == 1) {
            state->dollars += 3;
            if (remove_discard_sort_id(state, cards[0].sort_id)) removed[removed_count++] = cards[0];
        } else if (joker->center_id == CENTER_J_CASTLE) {
            for (uint8_t k = 0; k < count; ++k)
                if (!(cards[k].flags & CARD_DEBUFFED) && (cards[k].enhancement == ENHANCEMENT_WILD || cards[k].suit == (uint8_t)joker->state[1]))
                    joker->state[0] += 3;
        }
    }
    for (uint8_t k = 0; k < count; ++k)
        if (!(cards[k].flags & CARD_DEBUFFED) && cards[k].seal == SEAL_PURPLE && state->consumable_count < state->consumable_slots)
            (void)add_pooled_consumable(state, SET_TAROT, "8ba", 0);
    for (uint8_t j = 0; j < state->joker_count; ++j) {
        Card *joker = &state->jokers[j];
        if (joker->flags & CARD_DEBUFFED) continue;
        if (joker->center_id == CENTER_J_GREEN_JOKER && joker->state[0] > 0) joker->state[0]--;
        if (joker->center_id == CENTER_J_HIT_THE_ROAD) joker->state[0] += jacks * 50;
        if (joker->center_id == CENTER_J_RAMEN) {
            int x = joker->state[0] > 100 ? joker->state[0] : 200;
            x -= count;
            if (x <= 100)
                joker->state[1] = 1;
            else
                joker->state[0] = x;
        }
        if (joker->center_id == CENTER_J_YORICK) {
            int remaining = joker->state[1] > 0 ? joker->state[1] : 23;
            for (uint8_t k = 0; k < count; ++k) {
                if (--remaining <= 0) {
                    remaining = 23;
                    joker->state[0] = (joker->state[0] > 100 ? joker->state[0] : 100) + 100;
                }
            }
            joker->state[1] = remaining;
        }
    }
    for (uint8_t j = state->joker_count; j-- > 0;)
        if (state->jokers[j].center_id == CENTER_J_RAMEN && state->jokers[j].state[1]) remove_joker_at(state, j);
    if (removed_count && hook) notify_removed_playing_cards(state, removed, removed_count, 0);
}

void apply_hook_discard(State *state) {
    if (state->blind_disabled || state->blind_id != BLIND_BL_HOOK) return;
    Card cards[2];
    uint8_t count = 0;
    for (uint8_t draw = 0; draw < 2 && state->hand_count; ++draw) {
        uint8_t index = random_sorted_hand_index(state, "hook");
        cards[count++] = state->hand[index];
        if (state->discard_count < MAX_DECK) state->discard[state->discard_count++] = state->hand[index];
        ZONE_REMOVE(state->hand, state->hand_count, index);
    }
    if (count) apply_discard_effects(state, cards, count, state->discards_used == 0, 1);
}

static int remove_discard_sort_id(State *state, uint16_t sort_id) {
    for (uint16_t i = 0; i < state->discard_count; ++i)
        if (state->discard[i].sort_id == sort_id) {
            ZONE_REMOVE(state->discard, state->discard_count, i);
            return 1;
        }
    return 0;
}

void apply_drawn_to_hand_boss(State *state, int crimson_prepped) {
    if (state->blind_disabled) return;
    if (state->blind_id == BLIND_BL_FINAL_BELL && state->hand_count) {
        for (uint8_t i = 0; i < state->hand_count; ++i)
            if (state->hand[i].flags & CARD_FORCED) return;
        uint8_t order[MAX_HAND];
        for (uint8_t i = 0; i < state->hand_count; ++i) order[i] = i;
        sort_indices(state->hand, order, state->hand_count);
        size_t sorted_index = (size_t)floor(pseudorandom(state, "cerulean_bell") * state->hand_count);
        state->hand[order[sorted_index]].flags |= CARD_FORCED;
    } else if (crimson_prepped && state->blind_id == BLIND_BL_FINAL_HEART && state->joker_count) {
        uint8_t eligible[MAX_JOKERS], count = 0;
        for (uint8_t i = 0; i < state->joker_count; ++i) {
            if (!(state->jokers[i].flags & CARD_DEBUFFED) || state->joker_count < 2) eligible[count++] = i;
            state->jokers[i].flags &= (uint8_t)~CARD_DEBUFFED;
        }
        sort_indices(state->jokers, eligible, count);
        if (count) {
            size_t sorted_index = (size_t)floor(pseudorandom(state, "crimson_heart") * count);
            state->jokers[eligible[sorted_index]].flags |= CARD_DEBUFFED;
        }
    }
}

static double probability_normal(const State *state, double base) {
    for (uint8_t i = 0; i < state->joker_count; ++i)
        if (!(state->jokers[i].flags & CARD_DEBUFFED) && state->jokers[i].center_id == CENTER_J_OOPS) base *= 2.0;
    return base;
}

static void draw_to_hand(State *state) {
    while (state->hand_count < state->hand_size && state->deck_count > 0 && state->hand_count < MAX_HAND)
        state->hand[state->hand_count++] = state->deck[--state->deck_count];
    sort_hand_desc(state);
}

void sort_hand_mode(State *state, int by_suit) {
    int keys[MAX_HAND];
    for (uint8_t i = 0; i < state->hand_count; ++i) {
        const Card *card = &state->hand[i];
        int suit_value = card->suit == SPADES ? 4 : card->suit == HEARTS ? 3 : card->suit == CLUBS ? 2 : 1;
        int nominal = card->rank == 14 ? 11 : card->rank >= 11 ? 10 : card->rank;
        int face = card->rank >= 11 ? (card->rank == 14 ? 4 : card->rank - 10) : 0;
        int base = nominal * 100 + face * 10;
        keys[i] = card->enhancement == ENHANCEMENT_STONE
            ? base - suit_value * 10000
            : by_suit ? suit_value * 10000 + base : base * 10 + suit_value;
    }
    for (uint8_t i = 1; i < state->hand_count; ++i) {
        Card card = state->hand[i];
        int key = keys[i];
        uint8_t j = i;
        while (j > 0) {
            if (keys[j - 1] >= key) break;
            state->hand[j] = state->hand[j - 1];
            keys[j] = keys[j - 1];
            --j;
        }
        state->hand[j] = card;
        keys[j] = key;
    }
}

static void sort_hand_desc(State *state) {
    sort_hand_mode(state, state->hand_sort_suit != 0);
}

static uint16_t choose_boss(State *state) {
    if (state->config.win_ante && state->ante >= 2 && state->ante % state->config.win_ante == 0) {
        static const uint16_t final_ids[] = {
            BLIND_BL_FINAL_ACORN, BLIND_BL_FINAL_BELL,   BLIND_BL_FINAL_HEART,
            BLIND_BL_FINAL_LEAF,  BLIND_BL_FINAL_VESSEL,
        };
        uint8_t min_use = UINT8_MAX;
        for (size_t i = 0; i < sizeof(final_ids) / sizeof(final_ids[0]); ++i)
            if (state->boss_usage[final_ids[i]] < min_use) min_use = state->boss_usage[final_ids[i]];
        uint16_t eligible[sizeof(final_ids) / sizeof(final_ids[0])];
        size_t count = 0;
        for (size_t i = 0; i < sizeof(final_ids) / sizeof(final_ids[0]); ++i)
            if (state->boss_usage[final_ids[i]] == min_use) eligible[count++] = final_ids[i];
        size_t index = (size_t)floor(pseudorandom(state, "boss") * count);
        uint16_t chosen = eligible[index < count ? index : count - 1];
        if (state->boss_usage[chosen] < UINT8_MAX) state->boss_usage[chosen]++;
        return chosen;
    }
    static const uint16_t ids[] = {
        BLIND_BL_ARM,   BLIND_BL_CLUB,    BLIND_BL_EYE,     BLIND_BL_FISH,  BLIND_BL_FLINT,
        BLIND_BL_GOAD,  BLIND_BL_HEAD,    BLIND_BL_HOOK,    BLIND_BL_HOUSE, BLIND_BL_MANACLE,
        BLIND_BL_MARK,  BLIND_BL_MOUTH,   BLIND_BL_NEEDLE,  BLIND_BL_OX,    BLIND_BL_PILLAR,
        BLIND_BL_PLANT, BLIND_BL_PSYCHIC, BLIND_BL_SERPENT, BLIND_BL_TOOTH, BLIND_BL_WALL,
        BLIND_BL_WATER, BLIND_BL_WHEEL,   BLIND_BL_WINDOW,
    };
    static const uint8_t minimum[] = {2, 1, 3, 2, 2, 1, 1, 1, 2, 1, 2, 2, 2, 6, 1, 4, 1, 5, 3, 2, 2, 2, 1};
    uint16_t eligible[sizeof(ids) / sizeof(ids[0])];
    size_t count = 0;
    uint8_t min_use = UINT8_MAX;
    for (size_t i = 0; i < sizeof(ids) / sizeof(ids[0]); ++i)
        if (state->ante >= minimum[i] && state->boss_usage[ids[i]] <= min_use) {
            if (state->boss_usage[ids[i]] < min_use) count = 0, min_use = state->boss_usage[ids[i]];
            eligible[count++] = ids[i];
        }
    if (!count) return BLIND_BL_ARM;
    size_t index = (size_t)floor(pseudorandom(state, "boss") * count);
    uint16_t chosen = eligible[index < count ? index : count - 1];
    if (state->boss_usage[chosen] < UINT8_MAX) state->boss_usage[chosen]++;
    return chosen;
}

static void clear_card_debuffs(State *state) {
    Card *zones[] = {state->deck, state->hand, state->discard, state->jokers};
    const uint16_t counts[] = {state->deck_count, state->hand_count,
                               state->discard_count, state->joker_count};
    for (int zone = 0; zone < 4; ++zone)
        for (uint16_t i = 0; i < counts[zone]; ++i)
            zones[zone][i].flags &= (uint8_t)~CARD_DEBUFFED;
    refresh_joker_cache(state);
}

static int blind_debuffs_card(const State *state, const Card *card) {
    if (state->blind_disabled) return 0;
    int stone = card->enhancement == ENHANCEMENT_STONE;
    int wild = card->enhancement == ENHANCEMENT_WILD;
    switch (state->blind_id) {
    case BLIND_BL_CLUB:
        return !stone && (wild || card->suit == CLUBS);
    case BLIND_BL_GOAD:
        return !stone && (wild || card->suit == SPADES);
    case BLIND_BL_HEAD:
        return !stone && (wild || card->suit == HEARTS);
    case BLIND_BL_WINDOW:
        return !stone && (wild || card->suit == DIAMONDS);
    case BLIND_BL_PLANT:
        return card->rank >= 11 && card->rank <= 13;
    case BLIND_BL_PILLAR:
        return card->state[3] != 0;
    case BLIND_BL_FINAL_LEAF:
        return 1;
    default:
        return 0;
    }
}

static void refresh_card_debuff(const State *state, Card *card) {
    card->flags &= (uint8_t)~CARD_DEBUFFED;
    if (blind_debuffs_card(state, card)) card->flags |= CARD_DEBUFFED;
}

static uint8_t most_played_hand(const State *state) {
    uint8_t best = HIGH_CARD;
    uint16_t best_count = 0;
    for (uint8_t hand = 0; hand < HAND_COUNT; ++hand) {
        uint16_t count = state->hand_plays[hand];
        if (count > best_count || (count && count == best_count && hand < best)) {
            best = hand;
            best_count = count;
        }
    }
    return best;
}

void swap_jokers(State *state, uint8_t left, uint8_t right) {
    Card card = state->jokers[left];
    state->jokers[left] = state->jokers[right];
    state->jokers[right] = card;
}

void default_config(Config *config) {
    if (!config) return;
    *config = (Config){
        .deck = 0,
        .stake = 1,
        .win_ante = 8,
        .fast_rng = 0,
        .progress_reward = 0.25f,
        .blind_bonus = 0.5f,
        .ante_bonus = 1.0f,
        .win_bonus = 10.0f,
        .loss_penalty = 0.25f,
    };
}

int init(State *state, const Config *config, uint64_t seed) {
    if (!state || !config) return ERR_ARGUMENT;
    char text[32];
    KeyBuilder key;
    key_begin(&key, text, sizeof(text));
    key_append_u64(&key, seed);
    memset(state, 0, sizeof(*state));
    state->config = *config;
    state->numeric_seed = seed;
    memcpy(state->seed, text, strlen(text) + 1);
    rng_reset(state);
    state->phase = PHASE_BLIND_SELECT;
    state->ante = 1;
    state->blind_on_deck = 0;
    state->dollars = 4;
    state->hand_size = 8;
    state->joker_slots = 5;
    state->consumable_slots = 2;
    state->reroll_base = 5;
    state->reroll_cost = 5;
    state->shop_joker_max = 2;
    state->hands_per_round = 4;
    state->discards_per_round = 3;
    state->joker_rate = 20.0f;
    state->tarot_rate = 4.0f;
    state->planet_rate = 4.0f;
    state->edition_rate = 1.0f;
    state->ancient_suit = UINT8_MAX;
    state->interest_cap = 25;
    state->interest_amount = 1;
    state->rental_rate = 3;
    state->stake_scaling = config->stake >= 6 ? 3 : config->stake >= 3 ? 2 : 1;
    switch (config->deck) {
    case 0:
    case CENTER_B_RED:
        state->discards_per_round++;
        break;
    case CENTER_B_BLUE:
        state->hands_per_round++;
        break;
    case CENTER_B_BLACK:
        if (state->hands_per_round > 1) state->hands_per_round--;
        state->joker_slots++;
        break;
    case CENTER_B_GREEN:
        state->discards_per_round++;
        break;
    case CENTER_B_YELLOW:
        state->dollars += 10;
        break;
    case CENTER_B_GHOST:
        state->spectral_rate = 2.0f;
        break;
    case CENTER_B_NEBULA:
        if (state->consumable_slots) state->consumable_slots--;
        break;
    case CENTER_B_PAINTED:
        state->hand_size += 2;
        if (state->joker_slots) state->joker_slots--;
        break;
    default:
        break;
    }
    if (config->stake >= 5 && state->discards_per_round) state->discards_per_round--;
    state->base_hand_size = state->hand_size;
    state->next_boss_id = choose_boss(state);
    state->next_voucher_id = pick_voucher(state);
    assign_blind_tags(state);
    choose_orbital_hands(state);
    for (size_t i = 0; i < HAND_COUNT; ++i) state->hand_levels[i] = 1;
    build_starting_deck(state);
    shuffle(state, state->deck, state->deck_count, "shuffle");
    if (config->deck == CENTER_B_GHOST) {
        state->consumables[state->consumable_count++] = (Card){.center_id = CENTER_C_HEX, .cost = 4, .sell_cost = 2};
    } else if (config->deck == CENTER_B_MAGIC) {
        state->consumables[state->consumable_count++] = (Card){.center_id = CENTER_C_FOOL};
        state->consumables[state->consumable_count++] = (Card){.center_id = CENTER_C_FOOL};
        state->consumable_slots++;
        mark_center_used(state, CENTER_V_CRYSTAL_BALL);
    } else if (config->deck == CENTER_B_NEBULA) {
        mark_center_used(state, CENTER_V_TELESCOPE);
    } else if (config->deck == CENTER_B_ZODIAC) {
        state->shop_joker_max++;
        state->tarot_rate = 9.6f;
        state->planet_rate = 9.6f;
        mark_center_used(state, CENTER_V_TAROT_MERCHANT);
        mark_center_used(state, CENTER_V_PLANET_MERCHANT);
        mark_center_used(state, CENTER_V_OVERSTOCK_NORM);
    }
    return OK;
}

static int rank_chips(uint8_t rank) {
    return rank == 14 ? 11 : rank >= 10 ? 10 : rank;
}

static int is_face(const Card *card, int pareidolia) {
    return pareidolia || (card->rank >= 11 && card->rank <= 13);
}

static int is_suit_state(const State *state, const Card *card, uint8_t suit) {
    if (card->enhancement == ENHANCEMENT_WILD || card->suit == suit) return 1;
    for (uint8_t j = 0; j < state->joker_count; ++j) {
        if ((state->jokers[j].flags & CARD_DEBUFFED) || state->jokers[j].center_id != CENTER_J_SMEARED) continue;
        int card_red = card->suit == HEARTS || card->suit == DIAMONDS;
        int suit_red = suit == HEARTS || suit == DIAMONDS;
        int card_black = card->suit == CLUBS || card->suit == SPADES;
        int suit_black = suit == CLUBS || suit == SPADES;
        if ((card_red && suit_red) || (card_black && suit_black)) return 1;
    }
    return 0;
}

static int matador_debuff_bonus(const State *state) {
    if (state->blind_disabled) return 0;
    for (uint8_t j = 0; j < state->joker_count; ++j)
        if (!(state->jokers[j].flags & CARD_DEBUFFED) && state->jokers[j].center_id == CENTER_J_MATADOR) return 8;
    return 0;
}

static uint8_t resolved_joker_source(const State *state, uint8_t start) {
    uint8_t index = start;
    for (uint8_t depth = 0; depth <= state->joker_count; ++depth) {
        if (index >= state->joker_count) return UINT8_MAX;
        const Card *joker = &state->jokers[index];
        if (joker->flags & CARD_DEBUFFED) return UINT8_MAX;
        if (joker->center_id == CENTER_J_BLUEPRINT) {
            index = (uint8_t)(index + 1);
            continue;
        }
        if (joker->center_id == CENTER_J_BRAINSTORM) {
            if (index == 0) return UINT8_MAX;
            index = 0;
            continue;
        }
        return index;
    }
    return UINT8_MAX;
}

typedef struct HandEffect {
    uint8_t hand;
    int16_t chips;
    int16_t mult;
    uint16_t xmult;
} HandEffect;

static const HandEffect hand_effects[CENTER_COUNT] = {
    [CENTER_J_JOLLY] = {PAIR + 1, 0, 8, 0},
    [CENTER_J_ZANY] = {THREE_OF_A_KIND + 1, 0, 12, 0},
    [CENTER_J_MAD] = {TWO_PAIR + 1, 0, 10, 0},
    [CENTER_J_CRAZY] = {STRAIGHT + 1, 0, 12, 0},
    [CENTER_J_DROLL] = {FLUSH + 1, 0, 10, 0},
    [CENTER_J_SLY] = {PAIR + 1, 50, 0, 0},
    [CENTER_J_WILY] = {THREE_OF_A_KIND + 1, 100, 0, 0},
    [CENTER_J_CLEVER] = {TWO_PAIR + 1, 80, 0, 0},
    [CENTER_J_DEVIOUS] = {STRAIGHT + 1, 100, 0, 0},
    [CENTER_J_CRAFTY] = {FLUSH + 1, 80, 0, 0},
    [CENTER_J_DUO] = {PAIR + 1, 0, 0, 200},
    [CENTER_J_TRIO] = {THREE_OF_A_KIND + 1, 0, 0, 300},
    [CENTER_J_FAMILY] = {FOUR_OF_A_KIND + 1, 0, 0, 400},
    [CENTER_J_ORDER] = {STRAIGHT + 1, 0, 0, 300},
    [CENTER_J_TRIBE] = {FLUSH + 1, 0, 0, 200},
};

enum CardEffectCondition {
    CARD_EFFECT_SUIT = 1,
    CARD_EFFECT_RANK_SET,
    CARD_EFFECT_FACE,
    CARD_EFFECT_EVEN,
    CARD_EFFECT_ODD,
};

typedef struct CardEffect {
    uint8_t condition;
    uint8_t suit;
    uint16_t rank_mask;
    int16_t chips;
    int16_t mult;
    int16_t dollars;
    uint16_t xmult;
} CardEffect;

static const CardEffect card_effects[CENTER_COUNT] = {
    [CENTER_J_GREEDY_JOKER] = {CARD_EFFECT_SUIT, DIAMONDS, 0, 0, 3, 0, 0},
    [CENTER_J_LUSTY_JOKER] = {CARD_EFFECT_SUIT, HEARTS, 0, 0, 3, 0, 0},
    [CENTER_J_WRATHFUL_JOKER] = {CARD_EFFECT_SUIT, SPADES, 0, 0, 3, 0, 0},
    [CENTER_J_GLUTTENOUS_JOKER] = {CARD_EFFECT_SUIT, CLUBS, 0, 0, 3, 0, 0},
    [CENTER_J_ARROWHEAD] = {CARD_EFFECT_SUIT, SPADES, 0, 50, 0, 0, 0},
    [CENTER_J_ONYX_AGATE] = {CARD_EFFECT_SUIT, CLUBS, 0, 0, 7, 0, 0},
    [CENTER_J_ROUGH_GEM] = {CARD_EFFECT_SUIT, DIAMONDS, 0, 0, 0, 1, 0},
    [CENTER_J_FIBONACCI] = {CARD_EFFECT_RANK_SET, 0, (1u << 2) | (1u << 3) | (1u << 5) | (1u << 8) | (1u << 14), 0, 8, 0, 0},
    [CENTER_J_SCHOLAR] = {CARD_EFFECT_RANK_SET, 0, 1u << 14, 20, 4, 0, 0},
    [CENTER_J_WALKIE_TALKIE] = {CARD_EFFECT_RANK_SET, 0, (1u << 4) | (1u << 10), 10, 4, 0, 0},
    [CENTER_J_EVEN_STEVEN] = {CARD_EFFECT_EVEN, 0, 0, 0, 4, 0, 0},
    [CENTER_J_ODD_TODD] = {CARD_EFFECT_ODD, 0, 0, 31, 0, 0, 0},
    [CENTER_J_SCARY_FACE] = {CARD_EFFECT_FACE, 0, 0, 30, 0, 0, 0},
    [CENTER_J_SMILEY] = {CARD_EFFECT_FACE, 0, 0, 0, 5, 0, 0},
    [CENTER_J_TRIBOULET] = {CARD_EFFECT_RANK_SET, 0, (1u << 12) | (1u << 13), 0, 0, 0, 200},
};

enum MainEffectCondition {
    MAIN_ALWAYS = 1,
    MAIN_MAX3,
    MAIN_LAST_HAND,
    MAIN_REPEAT_HAND,
    MAIN_NO_DISCARDS,
    MAIN_POSITIVE_DOLLARS,
    MAIN_ENHANCED16,
    MAIN_DECK_LOST,
};

enum MainEffectScale {
    MAIN_UNIT,
    MAIN_JOKERS,
    MAIN_DISCARDS,
    MAIN_DECK,
    MAIN_DOLLARS,
    MAIN_DOLLARS_5,
    MAIN_HAND_PLAYS,
    MAIN_TAROTS,
    MAIN_STONES,
    MAIN_STEELS,
    MAIN_SKIPS,
};

typedef struct MainEffect {
    uint8_t condition;
    uint8_t scale;
    int16_t chips;
    int16_t mult;
    uint16_t xmult;
} MainEffect;

static const MainEffect main_effects[CENTER_COUNT] = {
    [CENTER_J_JOKER] = {MAIN_ALWAYS, MAIN_UNIT, 0, 4, 0},
    [CENTER_J_STUNTMAN] = {MAIN_ALWAYS, MAIN_UNIT, 250, 0, 0},
    [CENTER_J_HALF] = {MAIN_MAX3, MAIN_UNIT, 0, 20, 0},
    [CENTER_J_ABSTRACT] = {MAIN_ALWAYS, MAIN_JOKERS, 0, 3, 0},
    [CENTER_J_ACROBAT] = {MAIN_LAST_HAND, MAIN_UNIT, 0, 0, 300},
    [CENTER_J_MYSTIC_SUMMIT] = {MAIN_NO_DISCARDS, MAIN_UNIT, 0, 15, 0},
    [CENTER_J_BANNER] = {MAIN_ALWAYS, MAIN_DISCARDS, 30, 0, 0},
    [CENTER_J_BLUE_JOKER] = {MAIN_ALWAYS, MAIN_DECK, 2, 0, 0},
    [CENTER_J_BULL] = {MAIN_POSITIVE_DOLLARS, MAIN_DOLLARS, 2, 0, 0},
    [CENTER_J_BOOTSTRAPS] = {MAIN_ALWAYS, MAIN_DOLLARS_5, 0, 2, 0},
    [CENTER_J_SUPERNOVA] = {MAIN_ALWAYS, MAIN_HAND_PLAYS, 0, 1, 0},
    [CENTER_J_CARD_SHARP] = {MAIN_REPEAT_HAND, MAIN_UNIT, 0, 0, 300},
    [CENTER_J_FORTUNE_TELLER] = {MAIN_ALWAYS, MAIN_TAROTS, 0, 1, 0},
    [CENTER_J_EROSION] = {MAIN_DECK_LOST, MAIN_UNIT, 0, 4, 0},
    [CENTER_J_STONE] = {MAIN_ALWAYS, MAIN_STONES, 25, 0, 0},
    [CENTER_J_STEEL_JOKER] = {MAIN_ALWAYS, MAIN_STEELS, 0, 0, 120},
    [CENTER_J_DRIVERS_LICENSE] = {MAIN_ENHANCED16, MAIN_UNIT, 0, 0, 300},
    [CENTER_J_THROWBACK] = {MAIN_ALWAYS, MAIN_SKIPS, 0, 0, 125},
    [CENTER_J_GROS_MICHEL] = {MAIN_ALWAYS, MAIN_UNIT, 0, 15, 0},
    [CENTER_J_CAVENDISH] = {MAIN_ALWAYS, MAIN_UNIT, 0, 0, 300},
};

static const uint8_t state_xmult[CENTER_COUNT] = {
    [CENTER_J_LUCKY_CAT] = 1, [CENTER_J_CAMPFIRE] = 1, [CENTER_J_HOLOGRAM] = 1,
    [CENTER_J_CONSTELLATION] = 1, [CENTER_J_GLASS] = 1, [CENTER_J_CAINO] = 1,
    [CENTER_J_MADNESS] = 1, [CENTER_J_HIT_THE_ROAD] = 1, [CENTER_J_YORICK] = 1,
    [CENTER_J_VAMPIRE] = 1,
};

typedef struct ScoreContext {
    int playing_count;
    int stones;
    int steels;
    int enhanced;
    int sell_total;
    uint8_t hand_has[HAND_COUNT];
    uint8_t sources[MAX_JOKERS * 2];
    uint8_t copied[MAX_JOKERS * 2];
    size_t source_count;
    int held_pareidolia;
    int mime_count;
} ScoreContext;

static int joker_repetitions(const State *state, const ScoreContext *context, const Card *card,
                             size_t card_index, uint8_t scoring_mask, int pareidolia) {
    int repetitions = 0;
    for (size_t j = 0; j < context->source_count; ++j) {
        const Card *joker = &state->jokers[context->sources[j]];
        switch (joker->center_id) {
        case CENTER_J_HACK:
            if (card->rank >= 2 && card->rank <= 5) repetitions++;
            break;
        case CENTER_J_SOCK_AND_BUSKIN:
            if (is_face(card, pareidolia)) repetitions++;
            break;
        case CENTER_J_HANGING_CHAD: {
            size_t first = 0;
            while (first < 5 && !(scoring_mask & (1u << first))) first++;
            if (card_index == first) repetitions += 2;
            break;
        }
        case CENTER_J_DUSK:
            if (state->hands_left == 0) repetitions++;
            break;
        case CENTER_J_SELZER:
            repetitions++;
            break;
        default:
            break;
        }
    }
    return repetitions;
}

static void individual_jokers(State *state, const ScoreContext *context, const Card *played, size_t card_index,
                              uint8_t scoring_mask, double *chips,
                              double *mult, int *dollars, int pareidolia) {
    const Card *card = &played[card_index];
    for (size_t source_index = 0; source_index < context->source_count; ++source_index) {
        Card *joker = &state->jokers[context->sources[source_index]];
        int copied = context->copied[source_index];
        const CardEffect effect = card_effects[joker->center_id];
        int trigger = effect.condition == CARD_EFFECT_SUIT
            ? is_suit_state(state, card, effect.suit)
            : effect.condition == CARD_EFFECT_RANK_SET
            ? (card->rank < 16 && (effect.rank_mask & (1u << card->rank)))
            : effect.condition == CARD_EFFECT_FACE
            ? is_face(card, pareidolia)
            : effect.condition == CARD_EFFECT_EVEN
            ? card->rank <= 10 && !(card->rank & 1u)
            : effect.condition == CARD_EFFECT_ODD
            ? (card->rank <= 10 && (card->rank & 1u)) || card->rank == 14
            : 0;
        if (trigger) {
            *chips += effect.chips;
            *mult += effect.mult;
            *dollars += effect.dollars;
            if (effect.xmult) *mult *= effect.xmult / 100.0;
        }
        switch (joker->center_id) {
        case CENTER_J_BLOODSTONE:
            if (is_suit_state(state, card, HEARTS) && pseudorandom(state, "bloodstone") < probability_normal(state, 0.5))
                *mult *= 1.5;
            break;
        case CENTER_J_IDOL:
            if (joker->state[0] && card->rank == (uint8_t)joker->state[0] && is_suit_state(state, card, (uint8_t)joker->state[1]))
                *mult *= 2;
            break;
        case CENTER_J_PHOTOGRAPH: {
            size_t first_face = 0;
            while (first_face < 5 && (!(scoring_mask & (1u << first_face)) || !is_face(&played[first_face], pareidolia))) first_face++;
            if (card_index == first_face) *mult *= 2;
            break;
        }
        case CENTER_J_ANCIENT:
            if (state->ancient_suit < 4 && is_suit_state(state, card, state->ancient_suit)) *mult *= 1.5;
            break;
        case CENTER_J_TICKET:
            if (card->enhancement == ENHANCEMENT_GOLD)
                *dollars += 4;
            else if (is_face(card, pareidolia))
                for (uint8_t k = 0; k < state->joker_count; ++k)
                    if (state->jokers[k].center_id == CENTER_J_MIDAS_MASK && !(state->jokers[k].flags & CARD_DEBUFFED)) *dollars += 4;
            break;
        case CENTER_J_BUSINESS:
            if (is_face(card, pareidolia) && pseudorandom(state, "business") < probability_normal(state, 0.5)) *dollars += 2;
            break;
        case CENTER_J_HIKER:
            ((Card *)card)->perma_bonus += 5;
            break;
        case CENTER_J_WEE:
            if (!copied && card->rank == 2) {
                joker->state[0] += 8;
            }
            break;
        case CENTER_J_8_BALL:
            if (card->rank == 8 && state->consumable_count < state->consumable_slots &&
                pseudorandom(state, "8ball") < probability_normal(state, 0.25)) {
                Card tarot = create_pooled_card(state, SET_TAROT, "8ba", 0);
                state->consumables[state->consumable_count++] = tarot;
                consumable_added(state, &tarot);
            }
            break;
        default:
            break;
        }
    }
}

static void held_card_effects(State *state, const ScoreContext *context, const Card *held, size_t held_count,
                              double *mult, int *dollars) {
    for (size_t i = 0; i < held_count; ++i) {
        if (held[i].flags & CARD_DEBUFFED) continue;
        int repetitions = (held[i].seal == SEAL_RED ? 2 : 1) + context->mime_count;
        for (int repetition = 0; repetition < repetitions; ++repetition) {
            if (held[i].enhancement == ENHANCEMENT_STEEL) *mult *= 1.5;
            for (size_t j = 0; j < context->source_count; ++j) {
                const Card *joker = &state->jokers[context->sources[j]];
                if (joker->center_id == CENTER_J_SHOOT_THE_MOON && held[i].rank == 12) *mult += 13;
                if (joker->center_id == CENTER_J_BARON && held[i].rank == 13) *mult *= 1.5;
                if (joker->center_id == CENTER_J_RESERVED_PARKING && is_face(&held[i], context->held_pareidolia) &&
                    pseudorandom(state, "parking") < probability_normal(state, 0.5))
                    (*dollars)++;
            }
        }
    }
    for (size_t j = 0; j < context->source_count; ++j) {
        const Card *joker = &state->jokers[context->sources[j]];
        if (joker->center_id != CENTER_J_RAISED_FIST || !held_count) continue;
        size_t lowest = held_count;
        for (size_t i = 0; i < held_count; ++i) {
            if (held[i].enhancement == ENHANCEMENT_STONE) continue;
            if (lowest == held_count || held[i].rank <= held[lowest].rank) lowest = i;
        }
        if (lowest < held_count && !(held[lowest].flags & CARD_DEBUFFED)) *mult += 2 * rank_chips(held[lowest].rank);
    }
}

static void joker_on_joker_effects(const State *state, uint8_t target_index, double *mult) {
    if (target_index >= state->joker_count) return;
    const Card *target = &state->jokers[target_index];
    if (target->center_id >= CENTER_COUNT || centers[target->center_id].rarity != 2) return;
    for (uint8_t source = 0; source < state->joker_count; ++source) {
        const Card *joker = &state->jokers[source];
        if (source == target_index || (joker->flags & CARD_DEBUFFED)) continue;
        if (joker->center_id == CENTER_J_BASEBALL) *mult *= 1.5;
    }
}

static void joker_edition_before(const Card *joker, double *chips, double *mult) {
    if (joker->edition == EDITION_FOIL)
        *chips += 50;
    else if (joker->edition == EDITION_HOLO)
        *mult += 10;
}

static void joker_edition_after(const Card *joker, double *mult) {
    if (joker->edition == EDITION_POLYCHROME) *mult *= 1.5;
}

static void main_jokers(State *state, const ScoreContext *context, HandType hand, const Card *played,
                        size_t played_count, uint8_t scoring_mask, const Card *held, size_t held_count,
                        double *chips, double *mult, int *dollars) {
    for (uint8_t j = 0; j < state->joker_count; ++j) {
        Card *physical = &state->jokers[j];
        Card *joker = physical;
        uint8_t source_index = j;
        int bypass = 0;
        if (!(physical->flags & CARD_DEBUFFED) && joker->center_id == CENTER_J_BLUEPRINT) {
            if (j + 1 >= state->joker_count || (state->jokers[j + 1].flags & CARD_DEBUFFED)) {
                bypass = 1;
            } else {
                joker = &state->jokers[j + 1];
                source_index = (uint8_t)(j + 1);
            }
        } else if (!(physical->flags & CARD_DEBUFFED) && joker->center_id == CENTER_J_BRAINSTORM) {
            if (j == 0 || (state->jokers[0].flags & CARD_DEBUFFED)) {
                bypass = 1;
            } else {
                joker = &state->jokers[0];
                source_index = 0;
            }
        }
        if (bypass) {
            joker_edition_before(physical, chips, mult);
            joker_on_joker_effects(state, j, mult);
            joker_edition_after(physical, mult);
            continue;
        }
        if (joker->flags & CARD_DEBUFFED) {
            joker_on_joker_effects(state, j, mult);
            continue;
        }
        int copied = source_index != j;
        joker_edition_before(physical, chips, mult);
        const MainEffect main_effect = main_effects[joker->center_id];
        int trigger = 0;
        double units = 1.0;
        switch (main_effect.condition) {
        case MAIN_ALWAYS: trigger = 1; break;
        case MAIN_MAX3: trigger = played_count <= 3; break;
        case MAIN_LAST_HAND: trigger = state->hands_left == 0; break;
        case MAIN_REPEAT_HAND: trigger = state->hand_plays_round[hand] > 1; break;
        case MAIN_NO_DISCARDS: trigger = state->discards_left == 0; break;
        case MAIN_POSITIVE_DOLLARS: trigger = state->dollars > 0; break;
        case MAIN_ENHANCED16: trigger = context->enhanced >= 16; break;
        case MAIN_DECK_LOST: {
            int starting = state->config.deck == CENTER_B_ABANDONED ? 40 : 52;
            units = starting - context->playing_count;
            trigger = units > 0;
            break;
        }
        default: break;
        }
        if (trigger) {
            switch (main_effect.scale) {
            case MAIN_JOKERS: units = state->joker_count; break;
            case MAIN_DISCARDS: units = state->discards_left; break;
            case MAIN_DECK: units = state->deck_count; break;
            case MAIN_DOLLARS: units = state->dollars; break;
            case MAIN_DOLLARS_5: units = state->dollars / 5; break;
            case MAIN_HAND_PLAYS: units = state->hand_plays[hand]; break;
            case MAIN_TAROTS: units = state->tarots_used; break;
            case MAIN_STONES: units = context->stones; break;
            case MAIN_STEELS: units = context->steels; break;
            case MAIN_SKIPS: units = state->skips; break;
            default: break;
            }
            *chips += main_effect.chips * units;
            *mult += main_effect.mult * units;
            if (main_effect.xmult) *mult *= 1.0 + (main_effect.xmult - 100) / 100.0 * units;
        }
        const HandEffect hand_effect = hand_effects[joker->center_id];
        if (hand_effect.hand && context->hand_has[hand_effect.hand - 1]) {
            *chips += hand_effect.chips;
            *mult += hand_effect.mult;
            if (hand_effect.xmult) *mult *= hand_effect.xmult / 100.0;
        }
        if (state_xmult[joker->center_id] && joker->state[0] > 100)
            *mult *= joker->state[0] / 100.0;
        switch (joker->center_id) {
        case CENTER_J_MISPRINT:
            *mult += floor(pseudorandom(state, "misprint") * 24);
            break;
        case CENTER_J_BLACKBOARD: {
            int all_black = held_count > 0;
            for (size_t i = 0; i < held_count; ++i)
                if (!(is_suit_state(state, &held[i], CLUBS) || is_suit_state(state, &held[i], SPADES))) all_black = 0;
            if (all_black) *mult *= 3;
            break;
        }
        case CENTER_J_STENCIL: {
            int stencils = 0;
            for (uint8_t k = 0; k < state->joker_count; ++k)
                if (state->jokers[k].center_id == CENTER_J_STENCIL && !(state->jokers[k].flags & CARD_DEBUFFED)) stencils++;
            int factor = state->joker_slots - state->joker_count + stencils;
            if (factor > 1) *mult *= factor;
            break;
        }
        case CENTER_J_FLOWER_POT: {
            uint8_t suits = 0;
            for (size_t k = 0; k < played_count; ++k)
                if ((scoring_mask & (1u << k)) && played[k].enhancement != ENHANCEMENT_WILD && played[k].suit < 4)
                    suits |= (uint8_t)(1u << played[k].suit);
            for (size_t k = 0; k < played_count; ++k)
                if ((scoring_mask & (1u << k)) && played[k].enhancement == ENHANCEMENT_WILD)
                    for (uint8_t suit = 0; suit < 4; ++suit)
                        if (!(suits & (1u << suit))) {
                            suits |= (uint8_t)(1u << suit);
                            break;
                        }
            if (suits == 0x0f) *mult *= 3;
            break;
        }
        case CENTER_J_SEEING_DOUBLE: {
            uint8_t suits = 0;
            for (size_t k = 0; k < played_count; ++k)
                if (scoring_mask & (1u << k)) {
                    if (played[k].enhancement == ENHANCEMENT_WILD)
                        suits = 0x0f;
                    else if (played[k].suit < 4)
                        suits |= (uint8_t)(1u << played[k].suit);
                }
            if ((suits & (1u << CLUBS)) && (suits & ~(1u << CLUBS))) *mult *= 2;
            break;
        }
        case CENTER_J_LOYALTY_CARD: {
            int every = joker->state[0] > 0 ? joker->state[0] : 5;
            int created = joker->state[2] ? joker->state[1] : 0;
            int played = (int)state->run_hands_played - created;
            if (every > 0 && played > 0 && played % (every + 1) == every) *mult *= 4;
            break;
        }
        case CENTER_J_GREEN_JOKER:
            if (!copied) joker->state[0]++;
            *mult += joker->state[0];
            break;
        case CENTER_J_RIDE_THE_BUS: {
            if (!copied) {
                int face = 0;
                for (size_t k = 0; k < played_count; ++k)
                    if (scoring_mask & (1u << k)) face |= is_face(&played[k], 0);
                joker->state[0] = face ? 0 : joker->state[0] + 1;
            }
            *mult += joker->state[0];
            break;
        }
        case CENTER_J_TROUSERS:
            if (!copied && context->hand_has[TWO_PAIR]) joker->state[0] += 2;
            *mult += joker->state[0];
            break;
        case CENTER_J_SQUARE:
            if (!copied && played_count == 4) joker->state[0] += 4;
            *chips += joker->state[0];
            break;
        case CENTER_J_RUNNER:
            if (!copied && context->hand_has[STRAIGHT]) joker->state[0] += 15;
            *chips += joker->state[0];
            break;
        case CENTER_J_SWASHBUCKLER: {
            int sell_total = context->sell_total;
            if (!(state->jokers[source_index].flags & CARD_DEBUFFED)) sell_total -= state->jokers[source_index].sell_cost;
            *mult += sell_total;
            break;
        }
        case CENTER_J_WEE:
            *chips += joker->state[0];
            break;
        case CENTER_J_POPCORN: {
            int current = joker->state[0] > 0 ? joker->state[0] : 20;
            *mult += current;
            break;
        }
        case CENTER_J_FLASH:
        case CENTER_J_RED_CARD:
            *mult += joker->state[0];
            break;
        case CENTER_J_RAMEN: {
            if (joker->state[1]) break;
            int x = joker->state[0] > 100 ? joker->state[0] : 200;
            *mult *= x / 100.0;
            joker->state[0] = x;
            break;
        }
        case CENTER_J_ICE_CREAM: {
            if (joker->state[1]) break;
            int current = joker->state[0] > 0 ? joker->state[0] : 100;
            *chips += current;
            if (current <= 5) {
                joker->state[0] = 0;
                joker->state[1] = 1;
            } else
                joker->state[0] = current - 5;
            break;
        }
        case CENTER_J_CASTLE:
            *chips += joker->state[0];
            break;
        case CENTER_J_CEREMONIAL:
            *mult += joker->state[0];
            break;
        case CENTER_J_TODO_LIST:
            if (joker->state[1] == (int32_t)hand) *dollars += 4;
            break;
        case CENTER_J_OBELISK: {
            if (!copied) {
                int reset = 1;
                for (uint8_t h = 0; h < HAND_COUNT; ++h)
                    if (h != hand && state->hand_plays[h] >= state->hand_plays[hand]) {
                        reset = 0;
                        break;
                    }
                if (reset)
                    joker->state[0] = 100;
                else
                    joker->state[0] = (joker->state[0] > 100 ? joker->state[0] : 100) + 20;
            }
            if (joker->state[0] > 100) {
                int32_t scaled = joker->state[0];
                double x_mult;
                if ((scaled - 100) % 20) {
                    x_mult = scaled / 100.0;
                } else {
                    volatile double accumulated = 1.0;
                    for (int32_t increment = 0; increment < (scaled - 100) / 20; ++increment) accumulated += 0.2;
                    x_mult = accumulated;
                }
                *mult *= x_mult;
            }
            break;
        }
        default:
            break;
        }
        joker_on_joker_effects(state, j, mult);
        joker_edition_after(physical, mult);
    }
}

static int score_hand(State *state, const Card *played, size_t played_count, const Card *held, size_t held_count,
                      ScoreResult *out) {
    if (!state || !played || !out || played_count < 1 || played_count > 5) return ERR_ARGUMENT;
    memset(out, 0, sizeof(*out));
    refresh_joker_cache(state);
    uint8_t scoring = 0;
    int four_fingers = joker_active(state, CENTER_J_FOUR_FINGERS);
    int shortcut = joker_active(state, CENTER_J_SHORTCUT);
    int smeared = joker_active(state, CENTER_J_SMEARED);
    HandType hand = classify_hand_rules(played, played_count, &scoring, four_fingers, shortcut, smeared);
    state->hand_plays[hand]++;
    state->hand_plays_round[hand]++;
    int pareidolia = joker_active(state, CENTER_J_PAREIDOLIA);
    if (joker_active(state, CENTER_J_SPLASH))
        scoring = (uint8_t)((1u << played_count) - 1u);
    for (size_t i = 0; i < played_count; ++i)
        if (played[i].enhancement == ENHANCEMENT_STONE) scoring |= (uint8_t)(1u << i);
    int debuffed_card = 0;
    for (size_t i = 0; i < played_count; ++i)
        if ((scoring & (1u << i)) && (played[i].flags & CARD_DEBUFFED)) debuffed_card = 1;
    for (uint8_t j = 0; j < state->joker_count; ++j) {
        Card *joker = &state->jokers[j];
        if (joker->flags & CARD_DEBUFFED) continue;
        if (joker->center_id == CENTER_J_MIDAS_MASK) {
            for (size_t k = 0; k < played_count; ++k)
                if ((scoring & (1u << k)) && !(played[k].flags & CARD_DEBUFFED) && is_face(&played[k], pareidolia))
                    ((Card *)&played[k])->enhancement = ENHANCEMENT_GOLD;
        } else if (joker->center_id == CENTER_J_VAMPIRE) {
            int enhanced = 0;
            for (size_t k = 0; k < played_count; ++k)
                if ((scoring & (1u << k)) && !(played[k].flags & CARD_DEBUFFED) && played[k].enhancement) {
                    enhanced++;
                    ((Card *)&played[k])->enhancement = ENHANCEMENT_NONE;
                }
            if (enhanced) joker->state[0] = (joker->state[0] > 100 ? joker->state[0] : 100) + enhanced * 10;
        }
    }
    ScoreContext context = {0};
    uint8_t played_in_state[MAX_SELECTION] = {0};
    const Card *zones[] = {state->deck, state->hand, state->discard};
    const size_t zone_counts[] = {state->deck_count, state->hand_count, state->discard_count};
    for (size_t zone = 0; zone < sizeof(zones) / sizeof(zones[0]); ++zone) {
        for (size_t i = 0; i < zone_counts[zone]; ++i) {
            const Card *card = &zones[zone][i];
            context.playing_count++;
            context.stones += card->enhancement == ENHANCEMENT_STONE;
            context.steels += card->enhancement == ENHANCEMENT_STEEL;
            context.enhanced += card->enhancement != ENHANCEMENT_NONE;
            for (size_t k = 0; k < played_count; ++k)
                if (played[k].sort_id && played[k].sort_id == card->sort_id)
                    played_in_state[k] = 1;
        }
    }
    for (size_t i = 0; i < played_count; ++i) {
        if (played_in_state[i]) continue;
        context.stones += played[i].enhancement == ENHANCEMENT_STONE;
        context.steels += played[i].enhancement == ENHANCEMENT_STEEL;
        context.enhanced += played[i].enhancement != ENHANCEMENT_NONE;
    }
    for (uint8_t j = 0; j < state->joker_count; ++j) {
        Card *joker = &state->jokers[j];
        if (!(joker->flags & CARD_DEBUFFED)) context.sell_total += joker->sell_cost;
        uint8_t source = resolved_joker_source(state, j);
        if (source == UINT8_MAX) continue;
        context.sources[context.source_count] = source;
        context.copied[context.source_count++] = source != j;
        if (state->jokers[source].center_id == CENTER_J_PAREIDOLIA) context.held_pareidolia = 1;
        if (state->jokers[source].center_id == CENTER_J_MIME) context.mime_count++;
    }
    uint8_t ranks[15] = {0};
    for (size_t i = 0; i < played_count; ++i)
        if (played[i].enhancement != ENHANCEMENT_STONE && played[i].rank >= 2 && played[i].rank <= 14)
            ranks[played[i].rank]++;
    int groups_of_two = 0;
    int has_three = 0;
    int has_four = 0;
    for (uint8_t rank = 2; rank <= 14; ++rank) {
        groups_of_two += ranks[rank] >= 2;
        has_three |= ranks[rank] >= 3;
        has_four |= ranks[rank] >= 4;
    }
    context.hand_has[PAIR] = groups_of_two > 0;
    context.hand_has[TWO_PAIR] = groups_of_two > 1;
    context.hand_has[THREE_OF_A_KIND] = has_three;
    context.hand_has[FOUR_OF_A_KIND] = has_four;
    context.hand_has[STRAIGHT] = hand == STRAIGHT || hand == STRAIGHT_FLUSH;
    context.hand_has[FLUSH] = hand == FLUSH || hand == FLUSH_HOUSE ||
                              hand == FLUSH_FIVE || hand == STRAIGHT_FLUSH;
    for (uint8_t j = 0; j < state->joker_count; ++j)
        if (!(state->jokers[j].flags & CARD_DEBUFFED) && state->jokers[j].center_id == CENTER_J_SPACE &&
            pseudorandom(state, "space") < probability_normal(state, 0.25) && state->hand_levels[hand] < 255)
            state->hand_levels[hand]++;
    if (!state->blind_disabled) {
        if (state->blind_id == BLIND_BL_PSYCHIC && played_count < 5) {
            out->hand_type = hand;
            out->scoring_mask = scoring;
            out->dollars = matador_debuff_bonus(state);
            return OK;
        }
        if (state->blind_id == BLIND_BL_EYE && (state->blind_hands_mask & (1u << hand))) {
            out->hand_type = hand;
            out->scoring_mask = scoring;
            out->dollars = matador_debuff_bonus(state);
            return OK;
        }
        if (state->blind_id == BLIND_BL_MOUTH) {
            if (state->blind_only_hand == UINT8_MAX)
                state->blind_only_hand = (uint8_t)hand;
            else if (state->blind_only_hand != (uint8_t)hand) {
                out->hand_type = hand;
                out->scoring_mask = scoring;
                out->dollars = matador_debuff_bonus(state);
                return OK;
            }
        }
        state->blind_hands_mask |= (uint16_t)(1u << hand);
        if (state->blind_id == BLIND_BL_ARM && state->hand_levels[hand] > 1) state->hand_levels[hand]--;
    }
    uint8_t level = state->hand_levels[hand] ? state->hand_levels[hand] : 1;
    double chips = base_chips[hand] + level_chips[hand] * (level - 1);
    double mult = base_mult[hand] + level_mult[hand] * (level - 1);
    int dollars = 0;
    if (!state->blind_disabled && state->blind_id == BLIND_BL_FLINT) {
        chips = floor(chips * 0.5 + 0.5);
        mult = fmax(floor(mult * 0.5 + 0.5), 1.0);
    }
    for (size_t i = 0; i < played_count; ++i) {
        if (!(scoring & (1u << i)) || (played[i].flags & CARD_DEBUFFED)) continue;
        int repetitions = (played[i].seal == SEAL_RED ? 2 : 1) +
                          joker_repetitions(state, &context, &played[i], i, scoring, pareidolia);
        for (int repetition = 0; repetition < repetitions; ++repetition) {
            const Card *card = &played[i];
            chips += card->enhancement == ENHANCEMENT_STONE ? 50 : rank_chips(card->rank);
            chips += card->perma_bonus;
            if (card->enhancement == ENHANCEMENT_BONUS) chips += 30;
            if (card->enhancement == ENHANCEMENT_MULT) mult += 4;
            if (card->enhancement == ENHANCEMENT_GLASS) mult *= 2;
            if (card->enhancement == ENHANCEMENT_LUCKY) {
                int lucky_trigger = 0;
                if (pseudorandom(state, "lucky_mult") < probability_normal(state, 0.2)) {
                    mult += 20;
                    lucky_trigger = 1;
                }
                if (pseudorandom(state, "lucky_money") < probability_normal(state, 1.0 / 15.0)) {
                    dollars += 20;
                    lucky_trigger = 1;
                }
                if (lucky_trigger)
                    for (uint8_t j = 0; j < state->joker_count; ++j)
                        if (!(state->jokers[j].flags & CARD_DEBUFFED) && state->jokers[j].center_id == CENTER_J_LUCKY_CAT)
                            state->jokers[j].state[0] = (state->jokers[j].state[0] > 100 ? state->jokers[j].state[0] : 100) + 25;
            }
            if (card->seal == SEAL_GOLD) dollars += 3;
            if (card->edition == EDITION_FOIL)
                chips += 50.0;
            else if (card->edition == EDITION_HOLO)
                mult += 10.0;
            else if (card->edition == EDITION_POLYCHROME)
                mult *= 1.5;
            individual_jokers(state, &context, played, i, scoring, &chips, &mult, &dollars, pareidolia);
        }
    }
    held_card_effects(state, &context, held, held_count, &mult, &dollars);
    main_jokers(state, &context, hand, played, played_count, scoring, held, held_count, &chips, &mult, &dollars);
    if (debuffed_card) dollars += matador_debuff_bonus(state);
    if (center_used(state, CENTER_V_OBSERVATORY)) {
        for (uint8_t i = 0; i < state->consumable_count; ++i)
            if (planet_hand(state->consumables[i].center_id) == hand) mult *= 1.5;
    }
    if (state->config.deck == CENTER_B_PLASMA) {
        double balanced = floor((chips + mult) / 2.0);
        chips = balanced;
        mult = balanced;
    }
    out->hand_type = hand;
    out->scoring_mask = scoring;
    out->chips = chips;
    out->mult = mult;
    out->total = floor(chips * mult);
    out->dollars = dollars;
    for (size_t i = 0; i < played_count; ++i) {
        if ((scoring & (1u << i)) && !(played[i].flags & CARD_DEBUFFED) && played[i].enhancement == ENHANCEMENT_GLASS &&
            pseudorandom(state, "glass") < probability_normal(state, 0.25))
            out->destroyed_mask |= (uint8_t)(1u << i);
    }
    if (!state->blind_disabled && state->blind_id == BLIND_BL_TOOTH) out->dollars -= (int32_t)played_count;
    return OK;
}

static int consumable_no_target_legal(const State *state, uint16_t center_id) {
    switch (center_id) {
    case CENTER_C_FAMILIAR: case CENTER_C_GRIM: case CENTER_C_INCANTATION:
    case CENTER_C_IMMOLATE: case CENTER_C_SIGIL: case CENTER_C_OUIJA:
        return state->hand_count > 1;
    case CENTER_C_ANKH:
        return state->joker_count > 0 && state->joker_count < state->joker_slots;
    case CENTER_C_WRAITH: case CENTER_C_SOUL: case CENTER_C_JUDGEMENT:
        return state->joker_count < state->joker_slots;
    case CENTER_C_ECTOPLASM: case CENTER_C_HEX:
        for (uint8_t i = 0; i < state->joker_count; ++i)
            if (state->jokers[i].edition == EDITION_NONE) return 1;
        return 0;
    case CENTER_C_WHEEL_OF_FORTUNE:
        for (uint8_t i = 0; i < state->joker_count; ++i)
            if (!(state->jokers[i].flags & CARD_DEBUFFED) &&
                state->jokers[i].edition == EDITION_NONE) return 1;
        return 0;
    case CENTER_C_FOOL:
        return state->last_tarot_planet != 0 && state->last_tarot_planet != CENTER_C_FOOL;
    default:
        return 1;
    }
}

int action_is_legal_masks(const LegalMasks *masks, const Action *action) {
    assert(masks && action);
    if (action->selection_count > MAX_SELECTION) return 0;
    if (action->type >= ACTION_TYPE_COUNT || !masks->action_type[action->type] ||
        action->primary >= 64 || !(masks->primary[action->type] & (UINT64_C(1) << action->primary)))
        return 0;
    const ObservedSelection *selection = cached_selection(masks, action->type, action->primary);
    if (!selection || !selection->valid) return action->selection_count == 0;
    if (action->selection_count < selection->minimum ||
        action->selection_count > selection->maximum)
        return 0;
    uint64_t selected = 0;
    for (uint8_t i = 0; i < action->selection_count; ++i) {
        uint8_t index = action->selection[i];
        if (index >= MAX_HAND || (i && action->selection[i - 1] >= index)) return 0;
        selected |= UINT64_C(1) << index;
    }
    return selected && !(selected & ~selection->allowed_hand) &&
           (selected & selection->required_hand) == selection->required_hand;
}

int action_is_legal(const State *state, const Action *action) {
    assert(state && action);
    LegalMasks masks;
    return legal_masks(state, &masks) == OK && action_is_legal_masks(&masks, action);
}

static uint64_t hand_mask(const State *state) {
    return state->hand_count == 64 ? UINT64_MAX : (UINT64_C(1) << state->hand_count) - 1;
}

static uint64_t consumable_allowed_mask(const State *state, uint16_t center_id) {
    uint64_t allowed = hand_mask(state);
    if (center_id == CENTER_C_AURA)
        for (uint8_t i = 0; i < state->hand_count; ++i)
            if (state->hand[i].edition != EDITION_NONE) allowed &= ~(UINT64_C(1) << i);
    return allowed;
}

static void masks_add_discrete(LegalMasks *masks, Action action) {
    int primary = action_has_primary(action.type) ? action.primary : 0;
    assert(action.type < ACTION_TYPE_COUNT && primary < 64);
    masks->action_type[action.type] = 1;
    masks->primary[action.type] |= UINT64_C(1) << primary;
    if (action.type == ACTION_SWAP_HAND_LEFT && action.primary < OBS_MAX_HAND)
        masks->hand_reorder_destination[action.primary] |= UINT64_C(1) << (action.primary - 1);
    else if (action.type == ACTION_SWAP_HAND_RIGHT && action.primary < OBS_MAX_HAND)
        masks->hand_reorder_destination[action.primary] |= UINT64_C(1) << (action.primary + 1);
    else if (action.type == ACTION_SWAP_JOKERS_LEFT && action.primary < OBS_MAX_JOKERS)
        masks->joker_reorder_destination[action.primary] |= UINT64_C(1) << (action.primary - 1);
    else if (action.type == ACTION_SWAP_JOKERS_RIGHT && action.primary < OBS_MAX_JOKERS)
        masks->joker_reorder_destination[action.primary] |= UINT64_C(1) << (action.primary + 1);
}

static void legal_add_selection(LegalMasks *masks, uint8_t type, uint8_t primary, uint8_t minimum,
    uint8_t maximum, uint64_t allowed, uint64_t required) {
    uint8_t allowed_count = (uint8_t)__builtin_popcountll(allowed);
    if (maximum > allowed_count) maximum = allowed_count;
    if (minimum > maximum || __builtin_popcountll(required) > maximum || (required & allowed) != required)
        return;
    int slot = action_has_primary(type) ? primary : 0;
    assert(type < ACTION_TYPE_COUNT && slot < 64);
    masks->action_type[type] = 1;
    masks->primary[type] |= UINT64_C(1) << slot;
    ObservedSelection *selection =
        (ObservedSelection *)cached_selection(masks, type, (uint8_t)slot);
    if (!selection) return;
    *selection = (ObservedSelection){
        .allowed_hand = allowed,
        .required_hand = required,
        .minimum = minimum,
        .maximum = maximum,
        .valid = 1,
    };
}

/* Consumable legality: targeted effect needs a hand selection, otherwise the
   action is discrete when the effect has no target requirement. */
static void legal_add_consumable(LegalMasks *masks, const State *state, uint8_t type,
                                 uint8_t primary, const Card *card) {
    const CenterDefinition *definition = &centers[card->center_id];
    if (definition->target_max)
        legal_add_selection(masks, type, primary, definition->target_min,
            (uint8_t)definition->target_max, consumable_allowed_mask(state, card->center_id), 0);
    else if (consumable_no_target_legal(state, card->center_id))
        masks_add_discrete(masks, (Action){.type = type, .primary = primary});
}

static int legal_masks(const State *state, LegalMasks *out) {
    if (!state || !out) return ERR_ARGUMENT;
    LegalMasks *masks = out;
    *out = (LegalMasks){0};
#define ADD(type_value, primary_value) \
    masks_add_discrete(masks, (Action){.type = (type_value), .primary = (primary_value)})
    if (state->phase == PHASE_BLIND_SELECT) {
        ADD(ACTION_SELECT_BLIND, 0);
        if (state->blind_on_deck < 2) ADD(ACTION_SKIP_BLIND, 0);
        int can_reroll = center_used(state, CENTER_V_RETCON) ||
            (center_used(state, CENTER_V_DIRECTORS_CUT) && !state->boss_rerolled);
        if (can_reroll && can_afford(state, 10)) ADD(ACTION_REROLL_BOSS, 0);
    } else if (state->phase == PHASE_SELECTING_HAND) {
        uint64_t allowed = hand_mask(state), forced = 0;
        uint16_t discard_space = (uint16_t)(MAX_DECK - state->discard_count);
        uint8_t maximum = discard_space > MAX_SELECTION ? MAX_SELECTION : (uint8_t)discard_space;
        for (uint8_t i = 0; i < state->hand_count && i < MAX_HAND; ++i)
            if (state->hand[i].flags & CARD_FORCED) {
                forced = UINT64_C(1) << i;
                break;
            }
        if (state->hands_left)
            legal_add_selection(masks, ACTION_PLAY_HAND, 0, 1, maximum, allowed, forced);
        if (state->discards_left)
            legal_add_selection(masks, ACTION_DISCARD, 0, 1, maximum, allowed, 0);
        for (uint8_t i = 0; i < state->hand_count && i < MAX_HAND; ++i) {
            if (i) ADD(ACTION_SWAP_HAND_LEFT, i);
            if (i + 1 < state->hand_count) ADD(ACTION_SWAP_HAND_RIGHT, i);
        }
        ADD(ACTION_SORT_HAND_RANK, 0);
        ADD(ACTION_SORT_HAND_SUIT, 0);
    } else if (state->phase == PHASE_ROUND_EVAL) {
        ADD(ACTION_CASH_OUT, 0);
    } else if (state->phase == PHASE_SHOP) {
        for (uint8_t i = 0; i < state->shop_main_count && i < OBS_MAX_SHOP_MAIN; ++i) {
            const Card *card = &state->shop_main[i];
            uint8_t set = card_set(card);
            int affordable = can_afford(state, card->cost);
            if (affordable && can_own(state, set, card->edition) &&
                (set == SET_DEFAULT || (set >= SET_ENHANCED && set <= SET_SPECTRAL)))
                ADD(ACTION_BUY_CARD, i);
            if (set >= SET_TAROT && set <= SET_SPECTRAL && affordable)
                legal_add_consumable(masks, state, ACTION_BUY_AND_USE, i, card);
        }
        for (uint8_t i = 0; i < state->shop_voucher_count && i < OBS_MAX_SHOP_VOUCHERS; ++i)
            if (can_afford(state, state->shop_vouchers[i].cost)) ADD(ACTION_REDEEM_VOUCHER, i);
        for (uint8_t i = 0; i < state->shop_booster_count && i < OBS_MAX_SHOP_BOOSTERS; ++i)
            if (can_afford(state, state->shop_boosters[i].cost)) ADD(ACTION_OPEN_BOOSTER, i);
        if (state->free_rerolls || can_afford(state, state->reroll_cost)) ADD(ACTION_REROLL, 0);
        ADD(ACTION_NEXT_ROUND, 0);
    } else if (state->phase == PHASE_PACK_OPENING) {
        for (uint8_t i = 0; i < state->pack_count && i < MAX_PACK_CARDS; ++i) {
            const Card *card = &state->pack_cards[i];
            uint8_t set = card_set(card);
            if (set >= SET_TAROT && set <= SET_SPECTRAL) {
                legal_add_consumable(masks, state, ACTION_PICK_PACK_CARD, i, card);
                continue;
            }
            int has_space = set != SET_JOKER || can_own(state, SET_JOKER, card->edition);
            if ((set == SET_PLAYING || set == SET_ENHANCED) &&
                (state->deck_count >= MAX_DECK || !can_add_playing_cards(state, 1)))
                has_space = 0;
            if (has_space) ADD(ACTION_PICK_PACK_CARD, i);
        }
        ADD(ACTION_SKIP_PACK, 0);
    }
    if (state->phase <= PHASE_PACK_OPENING) {
        if (state->phase != PHASE_ROUND_EVAL)
            for (uint8_t i = 0; i < state->joker_count && i < MAX_JOKERS; ++i) {
                if (i) ADD(ACTION_SWAP_JOKERS_LEFT, i);
                if (i + 1 < state->joker_count) ADD(ACTION_SWAP_JOKERS_RIGHT, i);
                if (!(state->jokers[i].flags & CARD_ETERNAL)) ADD(ACTION_SELL_JOKER, i);
            }
        for (uint8_t i = 0; i < state->consumable_count && i < MAX_CONSUMABLES; ++i) {
            const Card *card = &state->consumables[i];
            if (!(card->flags & CARD_ETERNAL)) ADD(ACTION_SELL_CONSUMABLE, i);
            legal_add_consumable(masks, state, ACTION_USE_CONSUMABLE, i, card);
        }
    }
#undef ADD
    return OK;
}

enum { PUBLIC_CARD_PLAYED_THIS_ANTE = 1u << 5 };

static const float small_integer_log2p1[] = {
    0.0f, 1.0f, 1.5849625f, 2.0f, 2.3219280f, 2.5849625f,
    2.8073549f, 3.0f, 3.1699250f, 3.3219280f, 3.4594316f,
    3.5849626f, 3.7004397f, 3.8073549f, 3.9068906f, 4.0f,
    4.0874629f,
};

static inline float observation_signed_log2(double value) {
    if (value == 0.0) return 0.0f;
    if (value >= -16.0 && value <= 16.0 && value == (double)(int)value) {
        int integer = (int)value;
        return integer < 0 ? -small_integer_log2p1[-integer] : small_integer_log2p1[integer];
    }
    if (isnan(value)) return 0.0f;
    if (isinf(value)) return signbit(value) ? -1024.0f : 1024.0f;
    return (float)copysign(log2(1.0 + fabs(value)), value);
}

typedef struct ObservationVariantBuild {
    uint64_t key;
    uint8_t rank;
    uint8_t suit;
    uint8_t enhancement;
    uint8_t edition;
    uint8_t seal;
    uint8_t flags;
    int16_t perma_bonus;
    uint16_t owned_count;
    uint16_t draw_count;
    uint16_t hand_count;
    uint16_t discard_count;
} ObservationVariantBuild;

static uint8_t public_playing_flags(const Card *card) {
    uint8_t flags = card->flags & (CARD_DEBUFFED | CARD_ETERNAL | CARD_PERISHABLE | CARD_RENTAL |
                                   CARD_FORCED);
    if (card->state[3]) flags |= PUBLIC_CARD_PLAYED_THIS_ANTE;
    return flags;
}

static uint64_t observation_variant_key_parts(uint8_t rank, uint8_t suit, uint8_t enhancement,
                                               uint8_t edition, uint8_t seal, int16_t perma_bonus,
                                               uint8_t flags) {
    uint64_t key = rank;
    key = (key << 2) | suit;
    key = (key << 4) | enhancement;
    key = (key << 3) | edition;
    key = (key << 3) | seal;
    key = (key << 16) | (uint16_t)((int32_t)perma_bonus - INT16_MIN);
    return (key << 7) | flags;
}

static uint64_t observation_variant_build_key(const ObservationVariantBuild *variant) {
    return observation_variant_key_parts(variant->rank, variant->suit, variant->enhancement,
                                         variant->edition, variant->seal, variant->perma_bonus,
                                         variant->flags);
}

static uint16_t find_observation_variant(const ObservationVariantBuild *variants, uint16_t count, const Card *card) {
    uint64_t key = observation_variant_key_parts(card->rank, card->suit, card->enhancement,
                                                 card->edition, card->seal, card->perma_bonus,
                                                 public_playing_flags(card));
    uint16_t low = 0, high = count;
    while (low < high) {
        uint16_t middle = (uint16_t)(low + (high - low) / 2);
        if (variants[middle].key < key)
            low = (uint16_t)(middle + 1);
        else
            high = middle;
    }
    if (low < count && variants[low].key == key) return low;
    return UINT16_MAX;
}

/* Physical ceiling of playing cards across draw/hand/discard zones. */
#define MAX_OBSERVED_CARDS \
    (OBS_MAX_PLAYING_CARDS + OBS_MAX_HAND + OBS_MAX_PLAYING_CARDS)

/* Sort the exceptional variant records by their packed key. Insertion sort
   covers the typical case (a handful of enhanced/editioned cards); the radix
   sort (with its per-pass histogram) only pays off past that point. */
static void sort_observation_variants(ObservationVariantBuild *variants, uint16_t count) {
    if (count < 2) return;
    if (count < 32) {
        for (uint16_t i = 1; i < count; ++i) {
            ObservationVariantBuild value = variants[i];
            uint64_t key = value.key;
            uint16_t j = i;
            while (j > 0 && variants[j - 1].key > key) {
                variants[j] = variants[j - 1];
                --j;
            }
            variants[j] = value;
        }
        return;
    }
    ObservationVariantBuild scratch[MAX_OBSERVED_CARDS];
    radix_sort(variants, scratch, count, sizeof(*variants),
               offsetof(ObservationVariantBuild, key), sizeof(uint64_t));
}

static void deck_summary_add(DeckSummary *summary, const Card *card) {
    int stone = card->enhancement == ENHANCEMENT_STONE;
    if (!stone) {
        if (card->rank >= 2 && card->rank <= 14) {
            uint8_t rank = (uint8_t)(card->rank - 2);
            summary->rank[rank]++;
            if (card->suit < 4) summary->rank_suit[card->suit][rank]++;
        }
        if (card->suit < 4) summary->suit[card->suit]++;
        summary->face_count += card->rank >= 11 && card->rank <= 13;
        summary->numbered_count += card->rank >= 2 && card->rank <= 10;
        summary->ace_count += card->rank == 14;
    }
    if (card->enhancement <= ENHANCEMENT_LUCKY) summary->enhancement[card->enhancement]++;
    if (card->edition <= EDITION_NEGATIVE) summary->edition[card->edition]++;
    if (card->seal <= SEAL_PURPLE) summary->seal[card->seal]++;
    summary->stone_count += stone;
    summary->wild_count += card->enhancement == ENHANCEMENT_WILD;
    summary->steel_count += card->enhancement == ENHANCEMENT_STEEL;
    summary->gold_count += card->enhancement == ENHANCEMENT_GOLD;
    summary->glass_count += card->enhancement == ENHANCEMENT_GLASS;
    summary->enhanced_count += card->enhancement != ENHANCEMENT_NONE;
    summary->unmodified_count += card->enhancement == ENHANCEMENT_NONE && card->edition == EDITION_NONE &&
                           card->seal == SEAL_NONE && card->perma_bonus == 0;
    summary->total_count++;
}

typedef struct PublicSnapshot {
    ObservationVariantBuild variants[OBS_MAX_PLAYING_VARIANTS];
    uint16_t hand_variant[OBS_MAX_HAND];
    DeckSummary owned_deck;
    DeckSummary draw_pile;
    uint16_t variant_count;
} PublicSnapshot;

typedef struct ObservationZoneCounts {
    uint16_t owned;
    uint16_t draw;
    uint16_t hand;
    uint16_t discard;
} ObservationZoneCounts;

static inline int simple_variant_slot(const Card *card) {
    uint8_t flags = public_playing_flags(card);
    if (card->rank < 2 || card->rank > 14 || card->suit >= 4 ||
        card->enhancement != ENHANCEMENT_NONE || card->edition != EDITION_NONE ||
        card->seal != SEAL_NONE || card->perma_bonus != 0 ||
        (flags != 0 && flags != PUBLIC_CARD_PLAYED_THIS_ANTE))
        return -1;
    return (((card->rank - 2) * 4 + card->suit) * 2) +
           (flags == PUBLIC_CARD_PLAYED_THIS_ANTE);
}

static inline void snapshot_card_add(DeckSummary *summary,
                                     ObservationZoneCounts counts[13 * 4 * 2],
                                     ObservationVariantBuild *exceptional,
                                     uint16_t *exceptional_count,
                                     const Card *card, uint8_t zone, int *simple) {
    int slot = simple_variant_slot(card);
    if (slot < 0) {
        *simple = 0;
        deck_summary_add(summary, card);
        exceptional[(*exceptional_count)++] = (ObservationVariantBuild){
            .rank = card->rank,
            .suit = card->suit,
            .enhancement = card->enhancement,
            .edition = card->edition,
            .seal = card->seal,
            .flags = public_playing_flags(card),
            .perma_bonus = card->perma_bonus,
            .owned_count = 1,
            .draw_count = zone == 0,
            .hand_count = zone == 1,
            .discard_count = zone > 1,
        };
        return;
    }
    deck_summary_add(summary, card);
    ObservationZoneCounts *entry = &counts[slot];
    entry->owned++;
    if (zone == 0)
        entry->draw++;
    else if (zone == 1)
        entry->hand++;
    else
        entry->discard++;
}

static void public_snapshot_from_state(const State *state, PublicSnapshot *snapshot) {
    snapshot->owned_deck = (DeckSummary){0};
    snapshot->draw_pile = (DeckSummary){0};
    ObservationZoneCounts simple_counts[13 * 4 * 2] = {0};
    ObservationVariantBuild exceptional[MAX_OBSERVED_CARDS];
    uint16_t exceptional_count = 0;
    int simple = 1;
    const Card *zones[] = {state->deck, state->hand, state->discard};
    const uint16_t zone_counts[] = {state->deck_count, state->hand_count, state->discard_count};
    for (int zone = 0; zone < 3; ++zone) {
        DeckSummary *summary = zone ? &snapshot->owned_deck : &snapshot->draw_pile;
        for (uint16_t i = 0; i < zone_counts[zone]; ++i)
            snapshot_card_add(summary, simple_counts, exceptional,
                              &exceptional_count, &zones[zone][i], zone, &simple);
        if (!zone) snapshot->owned_deck = snapshot->draw_pile;
    }
    if (simple) {
        uint16_t slot_to_variant[13 * 4 * 2];
        uint16_t count = 0;
        for (uint16_t slot = 0; slot < 13 * 4 * 2; ++slot) {
            const ObservationZoneCounts *counts = &simple_counts[slot];
            if (!counts->owned) continue;
            uint16_t card = (uint16_t)(slot / 2);
            ObservationVariantBuild *variant = &snapshot->variants[count];
            *variant = (ObservationVariantBuild){
                .rank = (uint8_t)(card / 4 + 2),
                .suit = (uint8_t)(card % 4),
                .flags = (slot & 1) ? PUBLIC_CARD_PLAYED_THIS_ANTE : 0,
                .owned_count = counts->owned,
                .draw_count = counts->draw,
                .hand_count = counts->hand,
                .discard_count = counts->discard,
            };
            variant->key = observation_variant_build_key(variant);
            slot_to_variant[slot] = count++;
        }
        snapshot->variant_count = count;
        for (uint8_t i = 0; i < state->hand_count; ++i)
            snapshot->hand_variant[i] = slot_to_variant[simple_variant_slot(&state->hand[i])];
        return;
    }

    /* One-pass aggregation: ordinary cards were tallied into simple_counts;
       exceptional cards accumulated in exceptional[]. Sort only the
       exceptional records, then merge with the ordinary slots in canonical
       key order. Ordinary slot order equals key order because rank, suit,
       and played-flag are the leading key bits; a card cannot be both
       ordinary and exceptional, so keys never collide across the streams. */
    for (uint16_t i = 0; i < exceptional_count; ++i)
        exceptional[i].key = observation_variant_build_key(&exceptional[i]);
    sort_observation_variants(exceptional, exceptional_count);

    uint16_t slot_to_variant[13 * 4 * 2];
    uint16_t merged = 0;
    uint16_t slot = 0;
    uint16_t e = 0;
    while (slot < 13 * 4 * 2 || e < exceptional_count) {
        ObservationVariantBuild ordinary = {0};
        int have_ordinary = 0;
        while (slot < 13 * 4 * 2 && !simple_counts[slot].owned) ++slot;
        if (slot < 13 * 4 * 2) {
            const ObservationZoneCounts *counts = &simple_counts[slot];
            uint16_t card = (uint16_t)(slot / 2);
            ordinary = (ObservationVariantBuild){
                .rank = (uint8_t)(card / 4 + 2),
                .suit = (uint8_t)(card % 4),
                .flags = (slot & 1) ? PUBLIC_CARD_PLAYED_THIS_ANTE : 0,
                .owned_count = counts->owned,
                .draw_count = counts->draw,
                .hand_count = counts->hand,
                .discard_count = counts->discard,
            };
            ordinary.key = observation_variant_build_key(&ordinary);
            have_ordinary = 1;
        }
        if (have_ordinary && (e >= exceptional_count || ordinary.key <= exceptional[e].key)) {
            snapshot->variants[merged] = ordinary;
            slot_to_variant[slot] = merged++;
            ++slot;
            continue;
        }
        if (!have_ordinary && e >= exceptional_count) break;
        ObservationVariantBuild *dst = &snapshot->variants[merged];
        *dst = exceptional[e];
        while (++e < exceptional_count && exceptional[e].key == dst->key) {
            dst->owned_count += exceptional[e].owned_count;
            dst->draw_count += exceptional[e].draw_count;
            dst->hand_count += exceptional[e].hand_count;
            dst->discard_count += exceptional[e].discard_count;
        }
        ++merged;
    }
    snapshot->variant_count = merged;
    for (uint8_t i = 0; i < state->hand_count; ++i) {
        int slot_index = simple_variant_slot(&state->hand[i]);
        snapshot->hand_variant[i] = slot_index >= 0
            ? slot_to_variant[slot_index]
            : find_observation_variant(snapshot->variants, merged, &state->hand[i]);
    }
}

typedef struct ObservationPokerHandBuild {
    uint8_t visible;
    uint32_t level;
    float chips;
    float mult;
    uint32_t total_plays;
    uint32_t round_plays;
} ObservationPokerHandBuild;

static ObservationPokerHandBuild observation_poker_hand(const State *state, uint8_t hand) {
    static const float base_chips_log[HAND_COUNT] = {7.33091688f, 7.13955116f, 6.9188633f, 6.65821171f,
                                                     5.9307375f, 5.35755205f, 5.16992521f, 4.95419645f,
                                                     4.95419645f, 4.3923173f, 3.45943165f, 2.58496261f};
    static const float base_mult_log[HAND_COUNT] = {4.0874629f, 3.90689063f, 3.70043969f, 3.16992497f,
                                                    3.0f, 2.32192802f, 2.32192802f, 2.32192802f,
                                                    2.0f, 1.58496249f, 1.58496249f, 1.0f};
    uint32_t level = state->hand_levels[hand] ? state->hand_levels[hand] : 1;
    return (ObservationPokerHandBuild){
        .visible = (uint8_t)(hand >= STRAIGHT_FLUSH || state->hand_plays[hand] != 0),
        .level = level,
        .chips = level == 1 ? base_chips_log[hand] : observation_signed_log2(base_chips[hand] + (double)level_chips[hand] * (level - 1)),
        .mult = level == 1 ? base_mult_log[hand] : observation_signed_log2(base_mult[hand] + (double)level_mult[hand] * (level - 1)),
        .total_plays = state->hand_plays[hand],
        .round_plays = state->hand_plays_round[hand],
    };
}

static int16_t quantize_q8_8(float value) {
    if (!(value == value)) return 0;
    if (value > 127.99609375f) value = 127.99609375f;
    if (value < -128.0f) value = -128.0f;
    float scaled = value * 256.0f;
    return (int16_t)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
}

/* quantize_q8_8(observation_signed_log2(v)) precomputed for integer v in 0..256
   using the exact float arithmetic of the two functions above. */
static const int16_t q8_8_integer_lut[257] = {
    0, 256, 406, 512, 594, 662, 719, 768, 812, 850, 886, 918, 947, 975, 1000, 1024,
    1046, 1068, 1087, 1106, 1124, 1142, 1158, 1174, 1189, 1203, 1217, 1231, 1244, 1256, 1268, 1280,
    1291, 1302, 1313, 1324, 1334, 1343, 1353, 1362, 1372, 1380, 1389, 1398, 1406, 1414, 1422, 1430,
    1437, 1445, 1452, 1459, 1466, 1473, 1480, 1487, 1493, 1500, 1506, 1512, 1518, 1524, 1530, 1536,
    1542, 1547, 1553, 1558, 1564, 1569, 1574, 1580, 1585, 1590, 1595, 1599, 1604, 1609, 1614, 1618,
    1623, 1628, 1632, 1636, 1641, 1645, 1649, 1654, 1658, 1662, 1666, 1670, 1674, 1678, 1682, 1686,
    1690, 1693, 1697, 1701, 1705, 1708, 1712, 1715, 1719, 1722, 1726, 1729, 1733, 1736, 1739, 1743,
    1746, 1749, 1752, 1756, 1759, 1762, 1765, 1768, 1771, 1774, 1777, 1780, 1783, 1786, 1789, 1792,
    1795, 1798, 1801, 1803, 1806, 1809, 1812, 1814, 1817, 1820, 1822, 1825, 1828, 1830, 1833, 1836,
    1838, 1841, 1843, 1846, 1848, 1851, 1853, 1855, 1858, 1860, 1863, 1865, 1867, 1870, 1872, 1874,
    1877, 1879, 1881, 1884, 1886, 1888, 1890, 1892, 1895, 1897, 1899, 1901, 1903, 1905, 1908, 1910,
    1912, 1914, 1916, 1918, 1920, 1922, 1924, 1926, 1928, 1930, 1932, 1934, 1936, 1938, 1940, 1942,
    1944, 1946, 1947, 1949, 1951, 1953, 1955, 1957, 1959, 1961, 1962, 1964, 1966, 1968, 1970, 1971,
    1973, 1975, 1977, 1978, 1980, 1982, 1984, 1985, 1987, 1989, 1990, 1992, 1994, 1995, 1997, 1999,
    2000, 2002, 2004, 2005, 2007, 2008, 2010, 2012, 2013, 2015, 2016, 2018, 2020, 2021, 2023, 2024,
    2026, 2027, 2029, 2030, 2032, 2033, 2035, 2036, 2038, 2039, 2041, 2042, 2044, 2045, 2047, 2048,
    2049,
};

static inline int16_t observation_q8_8(double value) {
    if (value == 0.0) return 0;
    if (value >= -256.0 && value <= 256.0 && value == (double)(int)value) {
        int integer = (int)value;
        if (integer < 0) return (int16_t)-q8_8_integer_lut[-integer];
        return (int16_t)q8_8_integer_lut[integer];
    }
    return quantize_q8_8(observation_signed_log2(value));
}

static ObservationCard observation_card(const Card *card, int playing) {
    ObservationCard out = {
        .center_id = card->center_id,
        .rank = card->rank,
        .suit = card->suit,
        .enhancement = card->enhancement,
        .edition = card->edition,
        .seal = card->seal,
        .flags = playing ? public_playing_flags(card) : card->flags,
    };
    out.perma_bonus_q8_8 = observation_q8_8(card->perma_bonus);
    out.cost_q8_8 = observation_q8_8(card->cost);
    out.sell_cost_q8_8 = observation_q8_8(card->sell_cost);
    for (int i = 0; i < 4; ++i) out.state_q8_8[i] = observation_q8_8(card->state[i]);
    return out;
}

int observe(const State *state, Observation *out, LegalMasks *legal) {
    if (!state || !out || !legal) return ERR_ARGUMENT;
    if (!state_layout_valid(state)) return ERR_INVARIANT;
    *out = (Observation){0};
    out->tags.count = 0;

    PublicSnapshot snapshot;
    public_snapshot_from_state(state, &snapshot);
    for (uint16_t i = 0; i < snapshot.variant_count; ++i) {
        const ObservationVariantBuild *variant = &snapshot.variants[i];
        out->variants.values[i] = (ObservationVariant){
            .rank = variant->rank,
            .suit = variant->suit,
            .enhancement = variant->enhancement,
            .edition = variant->edition,
            .seal = variant->seal,
            .flags = variant->flags,
            .perma_bonus_q8_8 = observation_q8_8(variant->perma_bonus),
            .owned_count = variant->owned_count,
            .draw_count = variant->draw_count,
            .hand_count = variant->hand_count,
            .discard_count = variant->discard_count,
        };
    }
    out->variants.count = snapshot.variant_count;
    out->hand.count = state->hand_count;
    for (uint8_t i = 0; i < state->hand_count; ++i) {
        out->hand.values[i] = (ObservationHandCard){
            .variant = snapshot.hand_variant[i],
            .flags = public_playing_flags(&state->hand[i]),
        };
    }
    out->owned_deck = snapshot.owned_deck;
    out->draw_pile = snapshot.draw_pile;

    ObservationCard *values[] = {
        out->jokers.values, out->consumables.values, out->shop.values,
        out->shop_vouchers.values, out->shop_boosters.values, out->pack.values,
    };
    uint16_t *counts_out[] = {
        &out->jokers.count, &out->consumables.count, &out->shop.count,
        &out->shop_vouchers.count, &out->shop_boosters.count, &out->pack.count,
    };
    const Card *cards[] = {
        state->jokers, state->consumables, state->shop_main, state->shop_vouchers,
        state->shop_boosters, state->pack_cards,
    };
    const uint8_t counts[] = {
        state->joker_count, state->consumable_count, state->shop_main_count,
        state->shop_voucher_count, state->shop_booster_count, state->pack_count,
    };
    for (int zone = 0; zone < 6; ++zone) {
        *counts_out[zone] = counts[zone];
        for (uint8_t i = 0; i < counts[zone]; ++i) {
            int playing = 0;
            if (zone == 2 || zone == 5) {
                uint8_t set = card_set(&cards[zone][i]);
                playing = set == SET_DEFAULT || set == SET_ENHANCED;
            }
            values[zone][i] = observation_card(&cards[zone][i], playing);
        }
    }

    if (state->double_tag) {
        uint16_t slot = out->tags.count++;
        out->tags.tag_id[slot] = TAG_TAG_DOUBLE;
        out->tags.flags[slot] = 1;
    }
    for (uint8_t blind = 0; blind < 2; ++blind)
        if (state->blind_tags[blind] != TAG_NONE) {
            uint16_t slot = out->tags.count++;
            out->tags.tag_id[slot] = state->blind_tags[blind];
            out->tags.orbital_hand[slot] = state->orbital_hands[blind];
            out->tags.flags[slot] = (uint8_t)(1u << (blind + 1));
        }

    for (uint8_t hand = 0; hand < HAND_COUNT; ++hand) {
        ObservationPokerHandBuild poker_hand = observation_poker_hand(state, hand);
        out->poker_hands[hand] = (ObservationPokerHand){
            .visible = poker_hand.visible,
            .level = poker_hand.level,
            .chips_q8_8 = quantize_q8_8(poker_hand.chips),
            .mult_q8_8 = quantize_q8_8(poker_hand.mult),
            .total_plays = poker_hand.total_plays,
            .round_plays = poker_hand.round_plays,
        };
    }

    uint16_t next_voucher_id = 0;
    for (uint8_t i = 0; i < state->shop_voucher_count; ++i)
        if (state->shop_vouchers[i].center_id == state->next_voucher_id) {
            next_voucher_id = state->next_voucher_id;
            break;
        }
    double chips_over_blind = state->blind_chips ? state->chips / state->blind_chips : 0.0;
#define OBS_GLOBAL_FIELDS(X) \
    X(blind_id) X(next_boss_id) X(last_tarot_planet) X(phase) X(blind_on_deck) \
    X(blind_disabled) X(hand_sort_suit) X(most_played_hand) X(last_hand_type) \
    X(blind_skipped_mask) X(blind_only_hand) X(boss_rerolled) X(free_rerolls) \
    X(reroll_base) X(reroll_increase) X(discount_percent) X(hands_per_round) \
    X(discards_per_round) X(base_hand_size) X(pack_kind) X(double_tag) \
    X(active_tag) X(tag_hand_bonus) X(tag_force_rarity) X(tag_force_rarity_count) \
    X(tag_force_edition) X(tag_force_edition_count) X(tag_voucher_pending) \
    X(tag_coupon_pending) X(tag_coupon_active) X(tag_investment_pending) \
    X(tag_d_six_pending) X(tag_d_six_active) X(ecto_penalty) X(gros_michel_extinct) \
    X(ante) X(run_hands_played) X(hands_left) X(discards_left) X(hands_played) \
    X(discards_used) X(hand_size) X(joker_slots) X(consumable_slots) X(skips) \
    X(pack_choices) X(unused_discards) X(blind_hands_mask) X(tarots_used) \
    X(planet_usage_mask)
    ObservationGlobals globals = {
        .deck_id = state->config.deck,
        .stake = state->config.stake,
        .next_voucher_id = next_voucher_id,
#define COPY_GLOBAL(field) .field = state->field,
        OBS_GLOBAL_FIELDS(COPY_GLOBAL)
#undef COPY_GLOBAL
        .dollars_q8_8 = observation_q8_8(state->dollars),
        .chips_q8_8 = observation_q8_8(state->chips),
        .blind_chips_q8_8 = observation_q8_8(state->blind_chips),
        .last_hand_score_q8_8 = observation_q8_8(state->last_hand_score),
        .chips_over_blind_q8_8 = observation_q8_8(chips_over_blind),
        .reroll_cost_q8_8 = observation_q8_8(state->reroll_cost),
        .round_earnings_q8_8 = observation_q8_8(state->round_earnings),
        .interest_cap_q8_8 = observation_q8_8(state->interest_cap),
        .interest_amount_q8_8 = observation_q8_8(state->interest_amount),
        .blind_reward_q8_8 = observation_q8_8(state->blind_reward),
        .joker_rate_q8_8 = quantize_q8_8(state->joker_rate),
        .tarot_rate_q8_8 = quantize_q8_8(state->tarot_rate),
        .planet_rate_q8_8 = quantize_q8_8(state->planet_rate),
        .spectral_rate_q8_8 = quantize_q8_8(state->spectral_rate),
        .playing_card_rate_q8_8 = quantize_q8_8(state->playing_card_rate),
        .edition_rate_q8_8 = quantize_q8_8(state->edition_rate),
    };
#undef OBS_GLOBAL_FIELDS
    size_t voucher_count = 0;
    const uint16_t *vouchers = center_pool(SET_VOUCHER, &voucher_count);
    for (size_t i = 0; i < voucher_count; ++i) {
        uint16_t id = vouchers[i];
        if ((state->used_centers[id / 8] >> (id % 8)) & 1u)
            globals.redeemed_vouchers[id / 8] |= (uint8_t)(1u << (id % 8));
    }
    out->globals = globals;
    return legal_masks(state, legal);
}

static void end_round_jokers(State *state);

static int remove_selected(State *state, const Action *action, Card played[MAX_SELECTION]) {
    uint8_t out_count = 0, keep_count = 0;
    Card keep[MAX_HAND];
    for (uint8_t i = 0; i < state->hand_count; ++i) {
        int selected = 0;
        for (uint8_t j = 0; j < action->selection_count; ++j)
            if (action->selection[j] == i) {
                selected = 1;
                break;
            }
        if (selected)
            played[out_count++] = state->hand[i];
        else
            keep[keep_count++] = state->hand[i];
    }
    memcpy(state->hand, keep, sizeof(Card) * keep_count);
    state->hand_count = keep_count;
    return out_count;
}

static void finish_blind(State *state) {
    state->phase = PHASE_ROUND_EVAL;
    if (!state->blind_disabled && state->blind_id == BLIND_BL_MANACLE && state->hand_size < UINT8_MAX) state->hand_size++;
    for (uint8_t i = 0; i < state->joker_count; ++i)
        if (state->jokers[i].flags & CARD_RENTAL) state->dollars -= state->rental_rate;
    state->unused_discards += state->discards_left;
    if (state->blind_on_deck == 2) state->most_played_hand = most_played_hand(state);
    int32_t tag_eval_bonus = 0;
    int32_t gold_hand_bonus = 0;
    int mime_count = 0;
    for (uint8_t j = 0; j < state->joker_count; ++j) {
        uint8_t source = resolved_joker_source(state, j);
        mime_count += source != UINT8_MAX && state->jokers[source].center_id == CENTER_J_MIME;
    }
    for (uint8_t i = 0; i < state->hand_count; ++i) {
        if (state->hand[i].flags & CARD_DEBUFFED) continue;
        uint8_t repetitions = (state->hand[i].seal == SEAL_RED ? 2 : 1) + mime_count;
        if (state->hand[i].enhancement == ENHANCEMENT_GOLD) gold_hand_bonus += 3 * repetitions;
        if (state->hands_played && state->hand[i].seal == SEAL_BLUE)
            for (uint8_t repetition = 0; repetition < repetitions; ++repetition)
                if (state->consumable_count < state->consumable_slots)
                    (void)add_specific_consumable(state, planet_center(state->last_hand_type));
    }
    if (state->tag_investment_pending && state->blind_on_deck == 2) {
        tag_eval_bonus = 25 * state->tag_investment_pending;
        state->tag_investment_pending = 0;
    }
    while (state->discard_count && state->deck_count < MAX_DECK)
        state->deck[state->deck_count++] = state->discard[--state->discard_count];
    while (state->hand_count && state->deck_count < MAX_DECK) state->deck[state->deck_count++] = state->hand[--state->hand_count];
    end_round_jokers(state);
    for (uint16_t i = 0; i < state->deck_count; ++i) state->deck[i].flags &= (uint8_t)~(CARD_DEBUFFED | CARD_FORCED);
    for (uint8_t i = 0; i < state->joker_count; ++i) state->jokers[i].flags &= (uint8_t)~CARD_DEBUFFED;
    if (state->blind_on_deck == 2) {
        if (state->config.deck == CENTER_B_ANAGLYPH) state->double_tag = 1;
        for (uint16_t i = 0; i < state->deck_count; ++i) state->deck[i].state[3] = 0;
        state->ante++;
        state->blind_on_deck = 0;
        state->blind_skipped_mask = 0;
        state->next_voucher_id = pick_voucher(state);
        assign_blind_tags(state);
        choose_orbital_hands(state);
        state->boss_rerolled = 0;
        state->next_boss_id = choose_boss(state);
    } else
        state->blind_on_deck++;
    if (state->tag_d_six_active) {
        state->tag_d_six_active = 0;
        state->reroll_cost = state->free_rerolls ? 0 : state->reroll_base + state->reroll_increase;
    }
    if (state->blind_id != BLIND_BL_SMALL && state->blind_id != BLIND_BL_BIG) {
        for (uint8_t i = 0; i < state->joker_count; ++i)
            if (!(state->jokers[i].flags & CARD_DEBUFFED) && state->jokers[i].center_id == CENTER_J_ROCKET)
                state->jokers[i].state[0] = (state->jokers[i].state[0] > 0 ? state->jokers[i].state[0] : 1) + 2;
    }
    for (uint8_t i = 0; i < state->joker_count; ++i) {
        Card *joker = &state->jokers[i];
        if (joker->flags & CARD_DEBUFFED) continue;
        if (joker->center_id == CENTER_J_EGG)
            joker->sell_cost += 3;
        else if (joker->center_id == CENTER_J_GIFT) {
            for (uint8_t j = 0; j < state->joker_count; ++j) state->jokers[j].sell_cost++;
            for (uint8_t j = 0; j < state->consumable_count; ++j) state->consumables[j].sell_cost++;
        } else if (joker->center_id == CENTER_J_INVISIBLE)
            joker->state[0]++;
        if ((joker->flags & CARD_PERISHABLE) && !(joker->flags & CARD_DEBUFFED)) {
            if (joker->state[3] <= 1) {
                joker->state[3] = 0;
                joker->flags |= CARD_DEBUFFED;
            } else
                joker->state[3]--;
        }
    }
    state->dollars += gold_hand_bonus;
    state->round_earnings = calculate_round_earnings(state) + tag_eval_bonus;
    state->blind_disabled = 1;
}

static double shaped_transition_reward(const State *state,
                                       const Action *action,
                                       double previous_chips,
                                       double previous_blind_chips,
                                       uint8_t previous_ante,
                                       uint8_t previous_phase) {
    double reward = 0.0;
    if (action->type == ACTION_PLAY_HAND &&
        isfinite(previous_chips) && previous_chips >= 0.0 &&
        isfinite(previous_blind_chips) && previous_blind_chips > 0.0 &&
        isfinite(state->last_hand_score) && state->last_hand_score > 0.0) {
        if (state->config.potential_scale) {
            double scale = state->config.potential_scale;
            double normalizer = log1p(scale);
            double previous_ratio = previous_chips / previous_blind_chips;
            double current_ratio = state->chips / previous_blind_chips;
            double previous_potential = log1p(scale * previous_ratio) / normalizer;
            double current_potential = log1p(scale * current_ratio) / normalizer;
            reward += state->config.progress_reward
                * (current_potential - previous_potential);
        } else {
            double chips_remaining = previous_blind_chips - previous_chips;
            if (chips_remaining > 0.0) {
                double credited_chips = state->last_hand_score;
                if (credited_chips > chips_remaining)
                    credited_chips = chips_remaining;
                reward += state->config.progress_reward
                    * credited_chips / previous_blind_chips;
            }
        }
    }
    if (action->type == ACTION_PLAY_HAND &&
        previous_phase == PHASE_SELECTING_HAND &&
        state->phase == PHASE_ROUND_EVAL)
        reward += state->config.blind_bonus;
    if (state->ante > previous_ante)
        reward += (double)(state->ante - previous_ante) * state->config.ante_bonus;
    if (state->terminal)
        reward += state->won
            ? state->config.win_bonus : -state->config.loss_penalty;
    return reward;
}

static void remove_joker_at(State *state, uint8_t index) {
    Card removed = state->jokers[index];
    joker_removed(state, &removed);
    ZONE_REMOVE(state->jokers, state->joker_count, index);
    refresh_joker_cache(state);
}

static void end_round_jokers(State *state) {
    for (uint8_t i = 0; i < state->joker_count; ++i) {
        Card *joker = &state->jokers[i];
        if (joker->flags & CARD_DEBUFFED) continue;
        if (joker->center_id == CENTER_J_POPCORN) {
            int mult = joker->state[0] > 0 ? joker->state[0] : 20;
            joker->state[0] = mult - 4 > 0 ? mult - 4 : 0;
            if (!joker->state[0]) {
                remove_joker_at(state, i--);
                continue;
            }
        } else if (joker->center_id == CENTER_J_CAMPFIRE && state->blind_on_deck == 2) {
            joker->state[0] = 100;
        } else if (joker->center_id == CENTER_J_HIT_THE_ROAD) {
            joker->state[0] = 100;
        } else if (joker->center_id == CENTER_J_TURTLE_BEAN && joker->state[0] > 0) {
            joker->state[0]--;
            if (!joker->state[0]) {
                remove_joker_at(state, i--);
                continue;
            }
        } else if (joker->center_id == CENTER_J_TODO_LIST) {
            joker->state[1] = choose_to_do_hand(state, joker->state[1]);
        } else if (joker->center_id == CENTER_J_GROS_MICHEL || joker->center_id == CENTER_J_CAVENDISH) {
            const double odds = joker->center_id == CENTER_J_GROS_MICHEL ? 6.0 : 1000.0;
            const char *stream = joker->center_id == CENTER_J_GROS_MICHEL ? "gros_michel" : "cavendish";
            if (pseudorandom(state, stream) < probability_normal(state, 1.0 / odds)) {
                if (joker->center_id == CENTER_J_GROS_MICHEL) state->gros_michel_extinct = 1;
                remove_joker_at(state, i--);
                continue;
            }
        }
    }
}

static void start_blind(State *state) {
    state->phase = PHASE_SELECTING_HAND;
    state->hand_size = state->base_hand_size;
    state->round++;
    reset_round_rerolls(state);
    clear_card_debuffs(state);
    state->blind_disabled = 0;
    state->blind_only_hand = UINT8_MAX;
    state->blind_hands_mask = 0;
    if (state->blind_on_deck == 2 && state->next_boss_id == 0) state->next_boss_id = choose_boss(state);
    state->blind_id = state->blind_on_deck == 0   ? BLIND_BL_SMALL
                      : state->blind_on_deck == 1 ? BLIND_BL_BIG
                                                  : state->next_boss_id;
    reset_round_targets(state);
    state->chips = 0;
    state->blind_chips = blind_amount(state->ante, state->stake_scaling)
        * (state->blind_on_deck == 0 ? 1.0 : state->blind_on_deck == 1 ? 1.5 : 2.0);
    if (state->config.deck == CENTER_B_PLASMA) state->blind_chips *= 2;
    if (state->blind_id == BLIND_BL_NEEDLE)
        state->blind_chips /= 2;
    else if (state->blind_id == BLIND_BL_WALL)
        state->blind_chips *= 2;
    else if (state->blind_id == BLIND_BL_FINAL_VESSEL)
        state->blind_chips *= 3;
    state->blind_reward = state->blind_id == BLIND_BL_FINAL_ACORN || state->blind_id == BLIND_BL_FINAL_BELL ||
                          state->blind_id == BLIND_BL_FINAL_HEART || state->blind_id == BLIND_BL_FINAL_LEAF ||
                          state->blind_id == BLIND_BL_FINAL_VESSEL
        ? 8
        : state->blind_on_deck == 0 ? 3 : state->blind_on_deck == 1 ? 4 : 5;
    if (state->config.stake >= 2 && state->blind_on_deck == 0) state->blind_reward = 0;
    state->hands_left = state->hands_per_round;
    state->discards_left = state->discards_per_round;
    if (state->tag_hand_bonus) {
        state->hand_size += state->tag_hand_bonus;
        state->tag_hand_bonus = 0;
    }
    for (uint8_t j = 0; j < state->joker_count; ++j) {
        Card *joker = &state->jokers[j];
        if (joker->flags & CARD_DEBUFFED) continue;
        if (joker->center_id == CENTER_J_JUGGLER)
            state->hand_size++;
        else if (joker->center_id == CENTER_J_TROUBADOUR) {
            state->hand_size += 2;
            if (state->hands_left > 1) state->hands_left--;
        } else if (joker->center_id == CENTER_J_MERRY_ANDY) {
            if (state->hand_size > 1) state->hand_size--;
            state->discards_left += 3;
        } else if (joker->center_id == CENTER_J_DRUNKARD) {
            state->discards_left++;
        } else if (joker->center_id == CENTER_J_TURTLE_BEAN) {
            if (!joker->state[0]) joker->state[0] = 5;
            state->hand_size += joker->state[0];
        }
    }
    for (uint8_t j = 0; j < state->joker_count; ++j) {
        Card *joker = &state->jokers[j];
        if (joker->flags & CARD_DEBUFFED) continue;
        if (joker->center_id == CENTER_J_BURGLAR) {
            state->hands_left += 3;
            state->discards_left = 0;
        } else if (joker->center_id == CENTER_J_CHICOT && state->blind_on_deck == 2) {
            state->blind_disabled = 1;
        } else if (joker->center_id == CENTER_J_CEREMONIAL && j + 1 < state->joker_count) {
            Card *target = &state->jokers[j + 1];
            if (!(target->flags & CARD_ETERNAL)) {
                joker->state[0] += target->sell_cost * 2;
                remove_joker_at(state, j + 1);
            }
        }
    }
    if (state->blind_disabled) {
        if (state->blind_id == BLIND_BL_WALL)
            state->blind_chips /= 2;
        else if (state->blind_id == BLIND_BL_FINAL_VESSEL)
            state->blind_chips /= 3;
    }
    if (state->blind_on_deck != 2) {
        int madness = -1;
        for (uint8_t j = 0; j < state->joker_count; ++j)
            if (!(state->jokers[j].flags & CARD_DEBUFFED) && state->jokers[j].center_id == CENTER_J_MADNESS) {
                madness = j;
                state->jokers[j].state[0] = (state->jokers[j].state[0] > 100 ? state->jokers[j].state[0] : 100) + 50;
                break;
            }
        if (madness >= 0) {
            uint8_t eligible[MAX_JOKERS];
            uint8_t count = 0;
            for (uint8_t j = 0; j < state->joker_count; ++j)
                if (j != (uint8_t)madness && !(state->jokers[j].flags & CARD_ETERNAL) && !(state->jokers[j].flags & CARD_DEBUFFED)) eligible[count++] = j;
            if (count) {
                sort_indices(state->jokers, eligible, count);
                size_t pick = (size_t)floor(pseudorandom(state, "madness") * count);
                remove_joker_at(state, eligible[pick]);
            }
        }
    }
    if (!state->blind_disabled && state->blind_id == BLIND_BL_NEEDLE) state->hands_left = 1;
    if (!state->blind_disabled && state->blind_id == BLIND_BL_WATER) state->discards_left = 0;
    if (!state->blind_disabled && state->blind_id == BLIND_BL_MANACLE && state->hand_size > 1) state->hand_size--;
    for (uint8_t j = 0; j < state->joker_count; ++j) {
        Card *joker = &state->jokers[j];
        if (joker->flags & CARD_DEBUFFED) continue;
        if (joker->center_id == CENTER_J_MARBLE && state->deck_count < MAX_DECK &&
            can_add_playing_cards(state, 1)) {
            Card stone = random_playing_card(state, "marb_fr");
            stone.enhancement = ENHANCEMENT_STONE;
            state->deck[state->deck_count++] = stone;
            playing_card_added(state, 1);
        } else if (joker->center_id == CENTER_J_CARTOMANCER) {
            add_pooled_consumable(state, SET_TAROT, "car", 0);
        } else if (joker->center_id == CENTER_J_RIFF_RAFF) {
            (void)add_joker_rarity(state, 1, "rif", 0);
            (void)add_joker_rarity(state, 1, "rif", 0);
        }
    }
    state->hands_played = state->discards_used = 0;
    memset(state->hand_plays_round, 0, sizeof(state->hand_plays_round));
    char round_shuffle[32];
    KeyBuilder round_key;
    key_begin(&round_key, round_shuffle, sizeof(round_shuffle));
    key_append(&round_key, "nr");
    key_append_u64(&round_key, state->ante);
    shuffle(state, state->deck, state->deck_count, round_shuffle);
    draw_to_hand(state);
    for (uint8_t i = 0; i < state->deck_count; ++i)
        if (blind_debuffs_card(state, &state->deck[i])) state->deck[i].flags |= CARD_DEBUFFED;
    for (uint8_t i = 0; i < state->hand_count; ++i)
        if (blind_debuffs_card(state, &state->hand[i])) state->hand[i].flags |= CARD_DEBUFFED;
    apply_drawn_to_hand_boss(state, 1);
    if (joker_active(state, CENTER_J_CERTIFICATE) && state->hand_count < MAX_HAND &&
        can_add_playing_cards(state, 1)) {
        Card certificate = random_playing_card(state, "cert_fr");
        double seal = pseudorandom(state, "certsl");
        certificate.seal = seal > 0.75 ? SEAL_RED : seal > 0.5 ? SEAL_BLUE : seal > 0.25 ? SEAL_GOLD : SEAL_PURPLE;
        if (blind_debuffs_card(state, &certificate)) certificate.flags |= CARD_DEBUFFED;
        state->hand[state->hand_count++] = certificate;
        playing_card_added(state, 1);
        sort_hand_desc(state);
    }
}

static void play_or_discard(State *state, const Action *action) {
    Card cards[MAX_SELECTION] = {0};
    int count = remove_selected(state, action, cards);
        uint16_t discard_start = state->discard_count;
    memcpy(&state->discard[state->discard_count], cards, sizeof(Card) * (size_t)count);
    state->discard_count += (uint16_t)count;
    if (action->type == ACTION_PLAY_HAND) {
        int first_hand = state->hands_played == 0;
        if (first_hand && count == 1 && joker_active(state, CENTER_J_DNA) && state->hand_count < MAX_HAND &&
            can_add_playing_cards(state, 1)) {
            Card copy = cards[0];
            copy.sort_id = ++state->next_sort_id;
            copy.flags &= (uint8_t)~CARD_DEBUFFED;
            state->hand[state->hand_count++] = copy;
            if (blind_debuffs_card(state, &state->hand[state->hand_count - 1]))
                state->hand[state->hand_count - 1].flags |= CARD_DEBUFFED;
            sort_hand_desc(state);
            playing_card_added(state, 1);
        }
        int ox_trigger = 0;
        if (!state->blind_disabled && state->blind_id == BLIND_BL_OX) {
            uint8_t mask = 0;
            HandType candidate = classify_hand(cards, (size_t)count, &mask);
            ox_trigger = candidate == (HandType)state->most_played_hand;
        }
        ScoreResult score;
        apply_hook_discard(state);
        for (uint8_t i = 0; i < state->hand_count; ++i) state->hand[i].flags &= (uint8_t)~CARD_FORCED;
        state->hands_left--;
        score_hand(state, cards, (size_t)count, state->hand, state->hand_count, &score);
        for (uint8_t j = state->joker_count; j-- > 0;) {
            Card *joker = &state->jokers[j];
            if (joker->center_id == CENTER_J_ICE_CREAM && joker->state[1]) {
                remove_joker_at(state, j);
            } else if (!(joker->flags & CARD_DEBUFFED) && joker->center_id == CENTER_J_SELZER) {
                int remaining = joker->state[0] > 0 ? joker->state[0] : 10;
                if (remaining <= 1)
                    remove_joker_at(state, j);
                else
                    joker->state[0] = remaining - 1;
            }
        }
        state->last_hand_score = score.total;
        state->last_hand_type = (uint8_t)score.hand_type;
        uint8_t shattered = 0, shattered_faces = 0, has_ace = 0;
        Card destroyed[MAX_SELECTION] = {0};
        uint8_t destroyed_count = 0;
        for (uint8_t i = 0; i < count; ++i)
            if (score.destroyed_mask & (1u << i)) {
                destroyed[destroyed_count++] = cards[i];
                shattered++;
                if (cards[i].rank >= 11 && cards[i].rank <= 13) shattered_faces++;
            }
        for (uint8_t i = 0; i < count; ++i) has_ace |= cards[i].rank == 14;
        notify_removed_playing_cards(state, destroyed, destroyed_count, 1);
        if (shattered)
            for (uint8_t j = 0; j < state->joker_count; ++j) {
                Card *joker = &state->jokers[j];
                if (joker->flags & CARD_DEBUFFED) continue;
                if (joker->center_id == CENTER_J_CAINO && shattered_faces)
                    joker->state[0] = (joker->state[0] > 100 ? joker->state[0] : 100) + shattered_faces * 100;
                else if (joker->center_id == CENTER_J_GLASS)
                    joker->state[0] = (joker->state[0] > 100 ? joker->state[0] : 100) + shattered * 75;
            }
        uint16_t discard_index = discard_start;
        for (uint8_t i = 0; i < count; ++i) {
            cards[i].state[3] = 1;
            if (score.destroyed_mask & (1u << i)) {
                ZONE_REMOVE(state->discard, state->discard_count, discard_index);
            } else {
                state->discard[discard_index].state[3] = 1;
                state->discard[discard_index].enhancement = cards[i].enhancement;
                state->discard[discard_index].perma_bonus = cards[i].perma_bonus;
                discard_index++;
            }
        }
        if (first_hand && count == 1 && cards[0].rank == 6 && joker_active(state, CENTER_J_SIXTH_SENSE)) {
            (void)remove_discard_sort_id(state, cards[0].sort_id);
            (void)add_pooled_consumable(state, SET_SPECTRAL, "sixth", 0);
        }
        if (score.hand_type == STRAIGHT_FLUSH && joker_active(state, CENTER_J_SEANCE))
            (void)add_pooled_consumable(state, SET_SPECTRAL, "sea", 0);
        if ((score.hand_type == STRAIGHT || score.hand_type == STRAIGHT_FLUSH) && has_ace &&
            joker_active(state, CENTER_J_SUPERPOSITION))
            (void)add_pooled_consumable(state, SET_TAROT, "sup", 0);
        if (state->dollars <= 4 && joker_active(state, CENTER_J_VAGABOND)) (void)add_pooled_consumable(state, SET_TAROT, "vag", 0);
        state->dollars += score.dollars;
        if (ox_trigger) state->dollars = 0;
        state->chips += state->last_hand_score;
        state->hands_played++;
        state->run_hands_played++;
        if (state->chips - state->blind_chips >= 0.0)
            finish_blind(state);
        else if (state->hands_left == 0) {
            int saved = 0;
            for (uint8_t j = 0; j < state->joker_count; ++j)
                if (!(state->jokers[j].flags & CARD_DEBUFFED) && state->jokers[j].center_id == CENTER_J_MR_BONES &&
                    state->chips / state->blind_chips >= 0.25) {
                    remove_joker_at(state, j);
                    saved = 1;
                    break;
                }
            if (saved) {
                state->blind_reward = 0;
                finish_blind(state);
            } else {
                state->terminal = 1;
                state->phase = PHASE_GAME_OVER;
            }
        }
    } else {
        int first_discard = state->discards_used == 0;
        state->discards_left--;
        state->discards_used++;
        apply_discard_effects(state, cards, (uint8_t)count, first_discard, 0);
    }
    if (!state->terminal && state->phase == PHASE_SELECTING_HAND) {
        draw_after_play(state);
        apply_drawn_to_hand_boss(state, action->type == ACTION_PLAY_HAND);
    }
}

int apply_step(State *state, const Action *action, const LegalMasks *masks, StepResult *out) {
    if (!state || !action || !out) return ERR_ARGUMENT;
    *out = (StepResult){0};
    if (!masks && !state_layout_valid(state)) return ERR_INVARIANT;
    if (!masks) refresh_joker_cache(state);
    if (!masks && !action_is_legal(state, action)) return ERR_ACTION;
    double previous_chips = state->chips;
    double previous_blind_chips = state->blind_chips;
    uint8_t previous_ante = state->ante;
    uint8_t previous_phase = state->phase;
    state->actions_taken++;

    switch (action->type) {
    case ACTION_REROLL_BOSS:
        state->boss_rerolled = 1;
        state->dollars -= 10;
        state->next_boss_id = choose_boss(state);
        break;
    case ACTION_SELECT_BLIND:
        start_blind(state);
        break;
    case ACTION_SKIP_BLIND: {
        uint8_t tag = state->blind_on_deck < 2 ? state->blind_tags[state->blind_on_deck] : TAG_NONE;
        state->skips++;
        state->blind_skipped_mask |= (uint8_t)(1u << state->blind_on_deck);
        if (tag != TAG_NONE) {
            apply_skip_tag(state, tag);
            if (tag != TAG_TAG_DOUBLE && state->double_tag) {
                state->double_tag = 0;
                apply_skip_tag(state, tag);
            }
        }
        state->blind_on_deck++;
        break;
    }
    case ACTION_PLAY_HAND:
    case ACTION_DISCARD:
        play_or_discard(state, action);
        break;
    case ACTION_SORT_HAND_RANK:
        state->hand_sort_suit = 0;
        sort_hand_mode(state, 0);
        break;
    case ACTION_SORT_HAND_SUIT:
        state->hand_sort_suit = 1;
        sort_hand_mode(state, 1);
        break;
    case ACTION_SWAP_HAND_LEFT:
    case ACTION_SWAP_HAND_RIGHT: {
        uint8_t other = action->type == ACTION_SWAP_HAND_LEFT ? action->primary - 1 : action->primary + 1;
        Card card = state->hand[action->primary];
        state->hand[action->primary] = state->hand[other];
        state->hand[other] = card;
        break;
    }
    case ACTION_SWAP_JOKERS_LEFT:
    case ACTION_SWAP_JOKERS_RIGHT: {
        uint8_t other = action->type == ACTION_SWAP_JOKERS_LEFT ? action->primary - 1 : action->primary + 1;
        swap_jokers(state, action->primary, other);
        break;
    }
    case ACTION_CASH_OUT: {
        char stream[32];
        key_with_u64(stream, sizeof(stream), "cashout", state->ante);
        shuffle(state, state->deck, state->deck_count, stream);
        state->dollars += state->round_earnings;
        state->round_earnings = 0;
        if (state->ante > state->config.win_ante) {
            state->won = state->terminal = 1;
            state->phase = PHASE_GAME_OVER;
        } else {
            state->phase = PHASE_SHOP;
            populate_shop(state);
        }
        break;
    }
    case ACTION_BUY_CARD:
        buy_shop_card(state, action->primary);
        break;
    case ACTION_BUY_AND_USE: {
        Card card = state->shop_main[action->primary];
        state->dollars -= card.cost;
        memmove(&state->shop_main[action->primary], &state->shop_main[action->primary + 1],
                (state->shop_main_count - action->primary - 1) * sizeof(Card));
        state->shop_main_count--;
        consumable_added(state, &card);
        apply_consumable(state, action, card);
        consumable_removed(state, &card);
        break;
    }
    case ACTION_SELL_JOKER:
        sell_joker(state, action->primary);
        break;
    case ACTION_SELL_CONSUMABLE:
        sell_consumable(state, action->primary);
        break;
    case ACTION_REROLL:
        reroll_shop(state);
        for (uint8_t j = 0; j < state->joker_count; ++j)
            if (!(state->jokers[j].flags & CARD_DEBUFFED) && state->jokers[j].center_id == CENTER_J_FLASH)
                state->jokers[j].state[0] += 2;
        break;
    case ACTION_REDEEM_VOUCHER:
        redeem_voucher(state, action->primary);
        break;
    case ACTION_OPEN_BOOSTER:
        open_booster(state, action->primary);
        break;
    case ACTION_PICK_PACK_CARD: {
        Card card = state->pack_cards[action->primary];
        uint8_t set = card_set(&card);
        if (set == SET_TAROT || set == SET_PLANET || set == SET_SPECTRAL) {
            apply_consumable(state, action, card);
            complete_pack_pick(state, action->primary);
        } else {
            pick_pack_card(state, action->primary);
        }
        break;
    }
    case ACTION_SKIP_PACK:
        skip_pack(state);
        for (uint8_t j = 0; j < state->joker_count; ++j)
            if (!(state->jokers[j].flags & CARD_DEBUFFED) && state->jokers[j].center_id == CENTER_J_RED_CARD)
                state->jokers[j].state[0] += 3;
        break;
    case ACTION_USE_CONSUMABLE:
        use_consumable(state, action);
        break;
    case ACTION_NEXT_ROUND:
        if (joker_active(state, CENTER_J_PERKEO) && state->consumable_count && state->consumable_count < state->consumable_slots &&
            state->consumable_count < MAX_CONSUMABLES) {
            size_t index = (size_t)floor(pseudorandom(state, "perkeo") * state->consumable_count);
            Card copy = state->consumables[index];
            copy.sort_id = ++state->next_sort_id;
            copy.edition = EDITION_NEGATIVE;
            price_card(state, &copy);
            state->consumables[state->consumable_count++] = copy;
            consumable_added(state, &copy);
        }
        state->hand_size = state->base_hand_size;
        state->shop_main_count = 0;
        state->shop_voucher_count = 0;
        state->shop_booster_count = 0;
        state->phase = PHASE_BLIND_SELECT;
        break;
    default:
        break;
    }

    out->terminal = state->terminal;
    out->won = state->won;
    out->ante = state->ante;
    out->sparse_reward = state->won ? 1.0f : 0.0f;
    out->reward = 0.0f;
    if (state->config.shaped_reward) {
        out->reward = (float)shaped_transition_reward(
            state, action, previous_chips, previous_blind_chips,
            previous_ante, previous_phase);
    }
    return OK;
}
