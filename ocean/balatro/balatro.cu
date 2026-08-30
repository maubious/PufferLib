// Balatro GPU encoder: sparse live-slot token pooling.
//
// Decodes live tokens directly from the packed Observation token stream,
// pools tokens permutation-invariantly per section (Hand, Jokers, Consumables,
// Shop, Vouchers, Boosters, Pack), decodes fixed globals and DeckMatrix features,
// and projects the concatenated section vectors once to the policy hidden size.

#include "joker_signatures.h"

static constexpr int BA_CENTER_EMBED = 8;
static constexpr int BA_BLIND_EMBED = 4;
static constexpr int BA_TAG_EMBED = 4;
static constexpr int BA_TOKEN_W = 32;
static constexpr int BA_DIMS_PER_PART = BA_TOKEN_W / 4;    // 8
static constexpr int BA_CARD_IN = 22;                     // center embed (8), rank (1), suit (1), enh, ed, seal, flags, dval, zone (7)
static constexpr int BA_ZONES = 7;                        // hand, jokers, consumables, shop, vouchers, boosters, pack

// Fixed sections layout
static constexpr int BA_OFF_GLOBALS = 0;
static constexpr int BA_GLOBALS_FEATURES = 117;           // 32 ids + 6 phase + 24 hand one-hots + 30 u8 + 2 u32 + 7 u16 stats + 3 i32 + 13 q8_8
static constexpr int BA_DECK_FEATURES = 279;              // 52 slots x 5 + 9 + 9 + 1
static constexpr int BA_OFF_DECK = BA_OFF_GLOBALS + BA_GLOBALS_FEATURES;
static constexpr int BA_OFF_VOUCHER = BA_OFF_DECK + BA_DECK_FEATURES;
static constexpr int BA_VOUCHER_FEATURES = 32;
static constexpr int BA_OFF_COUNTS = BA_OFF_VOUCHER + BA_VOUCHER_FEATURES;
static constexpr int BA_COUNT_FEATURES = 7;               // 7 zone counts
static constexpr int BA_OFF_POKER = BA_OFF_COUNTS + BA_COUNT_FEATURES;
static constexpr int BA_POKER_FEATURES = HAND_COUNT * 6;  // 12 * 6 = 72
static constexpr int BA_OFF_TAGS = BA_OFF_POKER + BA_POKER_FEATURES;
static constexpr int BA_TAG_FEATURES = 1 + 2 * (BA_TAG_EMBED + 2); // 1 + 2 * 6 = 13
static constexpr int BA_FIXED = BA_OFF_TAGS + BA_TAG_FEATURES;     // 520

static constexpr int BA_POOL_SECTIONS = 7;                // hand, jokers, consumables, shop, vouchers, boosters, pack
static constexpr int BA_MAX_OFFSET = BA_FIXED + BA_POOL_SECTIONS * BA_TOKEN_W; // 520 + 224 = 744
static constexpr int BA_POOLED = BA_MAX_OFFSET + BA_POOL_SECTIONS * BA_TOKEN_W; // 744 + 224 = 968
static constexpr int BA_ARGS = BA_POOL_SECTIONS * BA_TOKEN_W;                   // 224

// Per-slot sidecar: each live token's 16-dim embedding exposed per slot for
// the zones the AR heads target by index.
static constexpr int BA_SLOT_HAND = 16;
static constexpr int BA_SLOT_JOKERS = 8;
static constexpr int BA_SLOT_CONSUMABLES = 4;
static constexpr int BA_SLOT_SHOP = 4;
static constexpr int BA_SLOT_VOUCHERS = 2;
static constexpr int BA_SLOT_BOOSTERS = 2;
static constexpr int BA_SLOT_PACK = 5;

static constexpr int BA_SLOT_CAPS[BA_POOL_SECTIONS] = {
    BA_SLOT_HAND, BA_SLOT_JOKERS, BA_SLOT_CONSUMABLES,
    BA_SLOT_SHOP, BA_SLOT_VOUCHERS, BA_SLOT_BOOSTERS, BA_SLOT_PACK};
static constexpr int BA_SLOT_OFFSET[BA_POOL_SECTIONS] = {
    0,
    BA_SLOT_HAND,
    BA_SLOT_HAND + BA_SLOT_JOKERS,
    BA_SLOT_HAND + BA_SLOT_JOKERS + BA_SLOT_CONSUMABLES,
    BA_SLOT_HAND + BA_SLOT_JOKERS + BA_SLOT_CONSUMABLES + BA_SLOT_SHOP,
    BA_SLOT_HAND + BA_SLOT_JOKERS + BA_SLOT_CONSUMABLES + BA_SLOT_SHOP + BA_SLOT_VOUCHERS,
    BA_SLOT_HAND + BA_SLOT_JOKERS + BA_SLOT_CONSUMABLES + BA_SLOT_SHOP + BA_SLOT_VOUCHERS + BA_SLOT_BOOSTERS};

static constexpr int BA_SLOT_W = 16;
static constexpr int BA_SLOT_FEATURES =
    (BA_SLOT_HAND + BA_SLOT_JOKERS + BA_SLOT_CONSUMABLES + BA_SLOT_SHOP
        + BA_SLOT_VOUCHERS + BA_SLOT_BOOSTERS + BA_SLOT_PACK) * BA_SLOT_W; // 41 * 16 = 656

// Per-slot joker signature sidecar: typed effect vectors for center-bearing zones (Jokers..Pack)
static constexpr int SIG_W = 16;
static constexpr int SIG_ZONES = 6;
static constexpr int SIG_ZONE_CAPS[SIG_ZONES] = {
    BA_SLOT_JOKERS, BA_SLOT_CONSUMABLES,
    BA_SLOT_SHOP, BA_SLOT_VOUCHERS, BA_SLOT_BOOSTERS, BA_SLOT_PACK};
static constexpr int SIG_ZONE_OFFSET[SIG_ZONES] = {
    0,
    BA_SLOT_JOKERS,
    BA_SLOT_JOKERS + BA_SLOT_CONSUMABLES,
    BA_SLOT_JOKERS + BA_SLOT_CONSUMABLES + BA_SLOT_SHOP,
    BA_SLOT_JOKERS + BA_SLOT_CONSUMABLES + BA_SLOT_SHOP + BA_SLOT_VOUCHERS,
    BA_SLOT_JOKERS + BA_SLOT_CONSUMABLES + BA_SLOT_SHOP + BA_SLOT_VOUCHERS + BA_SLOT_BOOSTERS};
static constexpr int SIG_SLOTS =
    BA_SLOT_JOKERS + BA_SLOT_CONSUMABLES + BA_SLOT_SHOP
    + BA_SLOT_VOUCHERS + BA_SLOT_BOOSTERS + BA_SLOT_PACK; // 25
static constexpr int SIG_OFFSET = BA_POOLED + BA_SLOT_FEATURES; // 968 + 656 = 1624
static constexpr int BA_TOTAL = SIG_OFFSET + SIG_SLOTS * SIG_W; // 1624 + 400 = 2024

static constexpr int BA_BLOCK_AGENTS = 4;
static constexpr int BA_CELLS_C = BA_TOKEN_W * BA_CARD_IN; // 704
static constexpr int BA_TOKEN_CELLS = BA_CELLS_C + BA_TOKEN_W; // 736

static_assert(sizeof(Observation) == 1725,
    "Balatro encoder must be updated for the Observation layout");
static_assert(BA_FIXED == 520 && BA_POOLED == 968 && BA_TOTAL == 2024,
    "Balatro encoder pooled layout mismatch");

// Signature feature vector: SIG_W normalized floats from a table row.
__device__ __forceinline__ float sig_value(const JokerSignature* sig, int d) {
    switch (d) {
    case 0: return sig->trigger / 6.0f;
    case 1: return sig->condition / 10.0f;
    case 2: return sig->cond_value / 14.0f;
    case 3: return (sig->rank_mask & 0xFF) / 255.0f;
    case 4: return ((sig->rank_mask >> 8) & 0xFF) / 255.0f;
    case 5: return sig->chips / 500.0f;
    case 6: return sig->mult / 64.0f;
    case 7: return sig->xmult_pct / 400.0f;
    case 8: return sig->dollars / 16.0f;
    case 9: return sig->scale / 16.0f;
    case 10: return sig->growth / 32.0f;
    case 11: return sig->chance_pct / 100.0f;
    case 12: return sig->retriggers / 4.0f;
    case 13: return sig->levels / 8.0f;
    case 14: return sig->rule / 5.0f;
    default: return (float)sig->opaque;
    }
}

static int ba_use_max_pool = 0;
extern "C" void ba_set_use_max_pool(int use) {
    ba_use_max_pool = use ? 1 : 0;
}

struct BalatroEncoderWeights {
    Prec center_embed, blind_embed, tag_embed; // (CENTER,8) (BLIND,4) (TAG,4)
    Prec c_w, c_b;  // token MLP (32, 22) + bias (32)
    Prec proj_w;    // (hidden, BA_TOTAL)
    int obs_size, hidden;
};

struct BalatroEncoderActivations {
    Prec pooled, out, d_pooled, proj_wgrad;
    Int argmax;  // max-pool argmax per (batch, section, token dim)
    Float center_wgrad_f, blind_wgrad_f, tag_wgrad_f;
    Prec center_wgrad, blind_wgrad, tag_wgrad;
    Prec c_wgrad, c_bgrad;
    Float token_partials;
    int token_partial_rows;
    const unsigned char* obs_data;
    int obs_batch;
};

__device__ __forceinline__ int ba_byte(
        const unsigned char* obs, int64_t base, int offset) {
    return (int)obs[base + offset];
}

__device__ __forceinline__ int ba_u16(
        const unsigned char* obs, int64_t base, int offset) {
    return ba_byte(obs, base, offset)
        | (ba_byte(obs, base, offset + 1) << 8);
}

__device__ __forceinline__ int ba_i16(
        const unsigned char* obs, int64_t base, int offset) {
    int value = ba_u16(obs, base, offset);
    return value >= 32768 ? value - 65536 : value;
}

__device__ __forceinline__ uint32_t ba_u32(
        const unsigned char* obs, int64_t base, int offset) {
    return (uint32_t)ba_byte(obs, base, offset)
        | ((uint32_t)ba_byte(obs, base, offset + 1) << 8)
        | ((uint32_t)ba_byte(obs, base, offset + 2) << 16)
        | ((uint32_t)ba_byte(obs, base, offset + 3) << 24);
}

__device__ __forceinline__ float ba_gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

__device__ __forceinline__ float ba_gelu_deriv(float x) {
    return 0.5f * (1.0f + erff(x * 0.7071067811865475f))
        + 0.3989422804014327f * x * expf(-0.5f * x * x);
}

__device__ __forceinline__ void ba_live_counts(
        const unsigned char* obs, int64_t in, int counts[BA_POOL_SECTIONS]) {
    counts[0] = ba_byte(obs, in, offsetof(Observation, hand_count));
    counts[1] = ba_byte(obs, in, offsetof(Observation, joker_count));
    counts[2] = ba_byte(obs, in, offsetof(Observation, consumable_count));
    counts[3] = ba_byte(obs, in, offsetof(Observation, shop_count));
    counts[4] = ba_byte(obs, in, offsetof(Observation, voucher_count));
    counts[5] = ba_byte(obs, in, offsetof(Observation, booster_count));
    counts[6] = ba_byte(obs, in, offsetof(Observation, pack_count));
}

__device__ __forceinline__ int ba_slot_section(
        const int counts[BA_POOL_SECTIONS], int slot, int* local) {
    int cursor = 0;
    for (int s = 0; s < BA_POOL_SECTIONS; ++s) {
        if (slot < cursor + counts[s]) {
            *local = slot - cursor;
            return s;
        }
        cursor += counts[s];
    }
    return -1;
}

__device__ __forceinline__ void ba_token_input(
        const unsigned char* obs, int64_t in, int token_idx,
        const precision_t* center_embed, float* x, int* center_out, int* local_out) {
    int base = offsetof(Observation, tokens) + token_idx * sizeof(CardToken);
    int id = ba_u16(obs, in, base + 0);
    int zone = ba_byte(obs, in, base + 2);
    int enh = ba_byte(obs, in, base + 3);
    int ed = ba_byte(obs, in, base + 4);
    int seal = ba_byte(obs, in, base + 5);
    int flags = ba_byte(obs, in, base + 6);
    int dval = (int8_t)ba_byte(obs, in, base + 7);

    int is_playing = (zone == ZONE_HAND) || (id > 300);
    int center = -1;
    float rank = 0.0f, suit = 0.0f;
    if (is_playing) {
        rank = (float)(id >> 8) / 14.0f;
        suit = (float)(id & 0xFF) / 3.0f;
    } else {
        center = id;
    }
    for (int k = 0; k < BA_CENTER_EMBED; ++k) {
        x[k] = (center >= 0 && center < CENTER_COUNT)
            ? to_float(center_embed[center * BA_CENTER_EMBED + k]) : 0.0f;
    }
    x[8] = rank;
    x[9] = suit;
    x[10] = (float)enh / 8.0f;
    x[11] = (float)ed / 4.0f;
    x[12] = (float)seal / 4.0f;
    x[13] = (float)flags / 255.0f;
    x[14] = (float)dval / 128.0f;
    for (int k = 0; k < BA_ZONES; ++k) x[15 + k] = (k == zone) ? 1.0f : 0.0f;

    *center_out = center;
}

__global__ void __launch_bounds__(256, 4) ba_encode_kernel(
        precision_t* __restrict__ pooled,
        const unsigned char* __restrict__ obs,
        const precision_t* __restrict__ center_embed,
        const precision_t* __restrict__ blind_embed,
        const precision_t* __restrict__ tag_embed,
        const precision_t* __restrict__ c_w, const precision_t* __restrict__ c_b,
        int* __restrict__ argmax_out, int use_max,
        int B, int obs_size) {
    __shared__ float s_tok[64][BA_TOKEN_W];
    __shared__ float s_pool[BA_POOL_SECTIONS * BA_TOKEN_W];
    __shared__ float s_max[BA_POOL_SECTIONS * BA_TOKEN_W];
    __shared__ int s_argmax[BA_POOL_SECTIONS * BA_TOKEN_W];
    __shared__ float s_x[64][BA_CARD_IN];
    __shared__ int s_section[64];
    __shared__ int s_local[64];
    __shared__ int s_center[64];

    int b = blockIdx.x;
    int64_t in = (int64_t)b * obs_size;
    int base = offsetof(Observation, globals);
    int deck = ba_u16(obs, in, base + offsetof(ObservationGlobals, deck_id));
    int blind = ba_u16(obs, in, base + offsetof(ObservationGlobals, blind_id));
    int boss = ba_u16(obs, in, base + offsetof(ObservationGlobals, next_boss_id));
    int voucher = ba_u16(obs, in, base + offsetof(ObservationGlobals, next_voucher_id));
    int tarot = ba_u16(obs, in, base + offsetof(ObservationGlobals, last_tarot_planet));

    static constexpr int scalar_offset[30] = {
        offsetof(ObservationGlobals, stake), offsetof(ObservationGlobals, blind_on_deck),
        offsetof(ObservationGlobals, blind_disabled), offsetof(ObservationGlobals, hand_sort_suit),
        offsetof(ObservationGlobals, blind_skipped_mask), offsetof(ObservationGlobals, blind_only_hand),
        offsetof(ObservationGlobals, boss_rerolled), offsetof(ObservationGlobals, free_rerolls),
        offsetof(ObservationGlobals, reroll_base), offsetof(ObservationGlobals, reroll_increase),
        offsetof(ObservationGlobals, discount_percent), offsetof(ObservationGlobals, hands_per_round),
        offsetof(ObservationGlobals, discards_per_round), offsetof(ObservationGlobals, base_hand_size),
        offsetof(ObservationGlobals, pack_kind), offsetof(ObservationGlobals, double_tag),
        offsetof(ObservationGlobals, active_tag), offsetof(ObservationGlobals, tag_hand_bonus),
        offsetof(ObservationGlobals, tag_force_rarity), offsetof(ObservationGlobals, tag_force_rarity_count),
        offsetof(ObservationGlobals, tag_force_edition), offsetof(ObservationGlobals, tag_force_edition_count),
        offsetof(ObservationGlobals, tag_voucher_pending), offsetof(ObservationGlobals, tag_coupon_pending),
        offsetof(ObservationGlobals, tag_coupon_active), offsetof(ObservationGlobals, tag_investment_pending),
        offsetof(ObservationGlobals, tag_d_six_pending), offsetof(ObservationGlobals, tag_d_six_active),
        offsetof(ObservationGlobals, ecto_penalty), offsetof(ObservationGlobals, gros_michel_extinct),
    };
    static constexpr float scalar_scale[30] = {
        1.0f / 8.0f,   1.0f / 2.0f,  -1.0f,         -1.0f,         1.0f / 7.0f,
        1.0f / 16.0f,  -1.0f,        1.0f / 16.0f,  1.0f / 16.0f,  1.0f / 16.0f,
        1.0f / 100.0f, 1.0f / 16.0f, 1.0f / 16.0f,  1.0f / 16.0f,  1.0f / 16.0f,
        -1.0f,        1.0f / TAG_COUNT, 1.0f / 16.0f, 1.0f / 4.0f, 1.0f / 16.0f,
        1.0f / 4.0f,   1.0f / 16.0f, -1.0f,         -1.0f,         -1.0f,
        -1.0f,        -1.0f,        -1.0f,          1.0f / 16.0f, -1.0f,
    };

    for (int feature = threadIdx.x; feature < BA_FIXED; feature += blockDim.x) {
        float value = 0.0f;
        if (feature < BA_OFF_DECK) {
            int local = feature - BA_OFF_GLOBALS;
            if (local < 32) {
                int id = -1, d = 0;
                const precision_t* table = center_embed;
                if (local < 8) { id = deck; d = local; }
                else if (local < 12) { id = blind; d = local - 8; table = blind_embed; }
                else if (local < 16) { id = boss; d = local - 12; table = blind_embed; }
                else if (local < 24) { id = voucher; d = local - 16; }
                else { id = tarot; d = local - 24; }
                int limit = table == blind_embed ? BLIND_COUNT : CENTER_COUNT;
                int stored_id = id > 0 && id < limit ? id : -1;
                if (stored_id >= 0)
                    value = to_float(table[stored_id * (table == blind_embed ? 4 : 8) + d]);
            } else {
                local -= 32;
                int phase = ba_byte(obs, in, base + offsetof(ObservationGlobals, phase));
                int most = ba_byte(obs, in, base + offsetof(ObservationGlobals, most_played_hand));
                int last = ba_byte(obs, in, base + offsetof(ObservationGlobals, last_hand_type));
                if (local < 6) value = local == phase;
                else if ((local -= 6) < HAND_COUNT) value = local == most;
                else if ((local -= HAND_COUNT) < HAND_COUNT) value = local == last;
                else if ((local -= HAND_COUNT) < 30) {
                    int raw = ba_byte(obs, in, base + scalar_offset[local]);
                    value = scalar_scale[local] < 0.0f ? (float)(raw != 0) : (float)raw * scalar_scale[local];
                } else if ((local -= 30) < 2) {
                    int off = local == 0 ? offsetof(ObservationGlobals, ante)
                        : offsetof(ObservationGlobals, run_hands_played);
                    value = (float)ba_byte(obs, in, base + off) / (local == 0 ? 16.0f : 256.0f);
                } else if ((local -= 2) < 7) {
                    static constexpr int u16_stat_offsets[7] = {
                        offsetof(ObservationGlobals, hands_left), offsetof(ObservationGlobals, discards_left),
                        offsetof(ObservationGlobals, hands_played), offsetof(ObservationGlobals, discards_used),
                        offsetof(ObservationGlobals, blind_hands_mask), offsetof(ObservationGlobals, tarots_used),
                        offsetof(ObservationGlobals, planet_usage_mask),
                    };
                    value = (float)ba_u16(obs, in, base + u16_stat_offsets[local]) / 64.0f;
                } else if ((local -= 7) < 3) {
                    static constexpr int i32_stat_offsets[3] = {
                        offsetof(ObservationGlobals, dollars), offsetof(ObservationGlobals, reroll_cost),
                        offsetof(ObservationGlobals, round_earnings),
                    };
                    value = (float)(int32_t)ba_u32(obs, in, base + i32_stat_offsets[local]) / 64.0f;
                } else {
                    local -= 3;
                    value = (float)ba_i16(obs, in,
                        base + offsetof(ObservationGlobals, chips_q8_8) + 2 * local) / (256.0f * 16.0f);
                }
            }
        } else if (feature < BA_OFF_VOUCHER) {
            int local = feature - BA_OFF_DECK;
            int deck_base = offsetof(Observation, deck_matrix);
            if (local < 260) {
                int slot = local / 5;
                int f = local % 5;
                int slot_off = deck_base + slot * sizeof(DeckSlot);
                if (f == 0) value = (float)ba_byte(obs, in, slot_off + 0) / 52.0f;
                else if (f == 1) value = (float)ba_byte(obs, in, slot_off + 1) / 16.0f;
                else if (f == 2) value = (float)ba_byte(obs, in, slot_off + 2) / 52.0f;
                else if (f == 3) value = (float)ba_u16(obs, in, slot_off + 4) / 512.0f;
                else value = (float)ba_u16(obs, in, slot_off + 6) / 2048.0f;
            } else if ((local -= 260) < 9) {
                value = (float)ba_u16(obs, in, deck_base + offsetof(DeckMatrix, draw_enhancements) + 2 * local) / 52.0f;
            } else if ((local -= 9) < 9) {
                value = (float)ba_u16(obs, in, deck_base + offsetof(DeckMatrix, discard_enhancements) + 2 * local) / 52.0f;
            } else {
                value = (float)ba_u16(obs, in, deck_base + offsetof(DeckMatrix, total_deck_size)) / 256.0f;
            }
        } else if (feature < BA_OFF_COUNTS) {
            int bit = feature - BA_OFF_VOUCHER;
            uint32_t mask_lo = ba_u32(obs, in, base + offsetof(ObservationGlobals, redeemed_vouchers_mask));
            uint32_t mask_hi = ba_u32(obs, in, base + offsetof(ObservationGlobals, redeemed_vouchers_mask) + 4);
            uint32_t mask = bit < 32 ? mask_lo : mask_hi;
            value = (float)((mask >> (bit & 31)) & 1u);
        } else if (feature < BA_OFF_POKER) {
            int i = feature - BA_OFF_COUNTS;
            static constexpr int zone_offsets[7] = {
                offsetof(Observation, hand_count), offsetof(Observation, joker_count),
                offsetof(Observation, consumable_count), offsetof(Observation, shop_count),
                offsetof(Observation, voucher_count), offsetof(Observation, booster_count),
                offsetof(Observation, pack_count),
            };
            static constexpr float zone_scales[7] = {
                1.0f / 16.0f, 1.0f / 8.0f, 1.0f / 4.0f, 1.0f / 4.0f,
                1.0f / 2.0f,  1.0f / 2.0f, 1.0f / 5.0f,
            };
            value = (float)ba_byte(obs, in, zone_offsets[i]) * zone_scales[i];
        } else if (feature < BA_OFF_TAGS) {
            int local = feature - BA_OFF_POKER;
            int rec = local / 6;
            int f = local % 6;
            int record = offsetof(Observation, poker_hands) + rec * sizeof(PokerHandStat);
            if (f == 0) value = (float)ba_byte(obs, in, record + 1); // visible
            else if (f == 1) value = (float)ba_byte(obs, in, record + 0) / 16.0f; // level
            else if (f == 2) value = (float)ba_i16(obs, in, record + 4) / (256.0f * 128.0f); // chips_q8_8
            else if (f == 3) value = (float)ba_i16(obs, in, record + 6) / (256.0f * 128.0f); // mult_q8_8
            else if (f == 4) value = (float)ba_byte(obs, in, record + 3) / 256.0f; // total_plays
            else value = (float)ba_byte(obs, in, record + 2) / 64.0f; // round_plays
        } else {
            int local = feature - BA_OFF_TAGS;
            if (local == 0) {
                value = (float)ba_byte(obs, in, base + offsetof(ObservationGlobals, double_tag));
            } else {
                int blind_idx = (local - 1) / (BA_TAG_EMBED + 2);
                int f = (local - 1) % (BA_TAG_EMBED + 2);
                int tag_id = ba_byte(obs, in, base + offsetof(ObservationGlobals, blind_tags) + blind_idx);
                if (tag_id == 0 || tag_id >= TAG_COUNT) {
                    value = 0.0f;
                } else if (f < BA_TAG_EMBED) {
                    value = to_float(tag_embed[tag_id * BA_TAG_EMBED + f]);
                } else if (f == BA_TAG_EMBED) {
                    int orb = ba_byte(obs, in, base + offsetof(ObservationGlobals, orbital_hands) + blind_idx);
                    value = (float)orb / (float)HAND_COUNT;
                } else {
                    value = 1.0f;
                }
            }
        }
        pooled[b * BA_TOTAL + feature] = from_float(value);
    }

    for (int i = threadIdx.x; i < BA_POOL_SECTIONS * BA_TOKEN_W; i += blockDim.x) {
        s_pool[i] = 0.0f;
        if (use_max) {
            s_max[i] = 0.0f;
            s_argmax[i] = 0;
        }
    }
    __syncthreads();

    int counts[BA_POOL_SECTIONS];
    ba_live_counts(obs, in, counts);
    int total = ba_byte(obs, in, offsetof(Observation, total_tokens));
    if (total > MAX_OBS_TOKENS) total = MAX_OBS_TOKENS;

    for (int chunk_base = 0; chunk_base < total; chunk_base += 64) {
        int n = total - chunk_base < 64 ? total - chunk_base : 64;
        __syncthreads();
        for (int slot = threadIdx.x >> 2; slot < n; slot += blockDim.x >> 2) {
            int part = threadIdx.x & 3;
            int local;
            int section = ba_slot_section(counts, chunk_base + slot, &local);
            int center;
            ba_token_input(obs, in, chunk_base + slot, center_embed,
                s_x[slot], &center, &local);
            if (part == 0) {
                s_section[slot] = section;
                s_local[slot] = local;
                s_center[slot] = center;
            }
        }
        __syncthreads();
        for (int slot = threadIdx.x >> 2; slot < n; slot += blockDim.x >> 2) {
            int part = threadIdx.x & 3;
            const precision_t* w = c_w;
            const precision_t* bias = c_b;
            float acc[BA_DIMS_PER_PART];
            for (int d = 0; d < BA_DIMS_PER_PART; ++d)
                acc[d] = to_float(bias[part * BA_DIMS_PER_PART + d]);
            using Pair = short __attribute__((ext_vector_type(2)));
            int k = 0;
            for (; k + 1 < BA_CARD_IN; k += 2) {
                Pair xv = {
                    __builtin_bit_cast(short, from_float(s_x[slot][k])),
                    __builtin_bit_cast(short, from_float(s_x[slot][k + 1])),
                };
                for (int d = 0; d < BA_DIMS_PER_PART; ++d) {
                    int dim = part * BA_DIMS_PER_PART + d;
                    Pair wv = {
                        __builtin_bit_cast(short, w[dim * BA_CARD_IN + k]),
                        __builtin_bit_cast(short, w[dim * BA_CARD_IN + k + 1]),
                    };
                    acc[d] = __builtin_amdgcn_fdot2_f32_bf16(xv, wv, acc[d], false);
                }
            }
            if (k < BA_CARD_IN) {
                float xk = to_float(from_float(s_x[slot][k]));
                for (int d = 0; d < BA_DIMS_PER_PART; ++d) {
                    int dim = part * BA_DIMS_PER_PART + d;
                    acc[d] += xk * to_float(w[dim * BA_CARD_IN + k]);
                }
            }
            for (int d = 0; d < BA_DIMS_PER_PART; ++d)
                s_tok[slot][part * BA_DIMS_PER_PART + d] = ba_gelu(acc[d]);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < n * BA_SLOT_W; i += blockDim.x) {
            int slot = i >> 4;
            int d = i & (BA_SLOT_W - 1);
            int sec = s_section[slot];
            int local = s_local[slot];
            if (sec >= 0 && sec < BA_POOL_SECTIONS && local < BA_SLOT_CAPS[sec]) {
                pooled[b * BA_TOTAL + BA_POOLED
                    + (BA_SLOT_OFFSET[sec] + local) * BA_SLOT_W + d]
                    = from_float(s_tok[slot][d]);
            }
        }
        for (int i = threadIdx.x; i < n * SIG_W; i += blockDim.x) {
            int slot = i / SIG_W;
            int d = i % SIG_W;
            int sec = s_section[slot];
            if (sec >= 1 && sec <= 6) {
                int zone = sec - 1;
                int local = s_local[slot];
                if (local < SIG_ZONE_CAPS[zone]) {
                    int center = s_center[slot];
                    float v = (center >= 0 && center < CENTER_COUNT)
                        ? sig_value(&JOKER_SIGNATURES[center], d) : 0.0f;
                    pooled[b * BA_TOTAL + SIG_OFFSET
                        + (SIG_ZONE_OFFSET[zone] + local) * SIG_W + d]
                        = from_float(v);
                }
            }
        }
        for (int c = threadIdx.x; c < BA_POOL_SECTIONS * BA_TOKEN_W; c += blockDim.x) {
            int sec = c / BA_TOKEN_W;
            int d = c % BA_TOKEN_W;
            int start = 0;
            for (int s = 0; s < sec; ++s) start += counts[s];
            int end = start + counts[sec];
            if (end > chunk_base && start < chunk_base + n) {
                int lo = start > chunk_base ? start - chunk_base : 0;
                int hi = end < chunk_base + n ? end - chunk_base : n;
                float sum = 0.0f;
                float mx = 0.0f;
                int arg = 0;
                if (use_max) {
                    mx = s_max[c];
                    arg = s_argmax[c];
                }
                for (int slot = lo; slot < hi; ++slot) {
                    float v = s_tok[slot][d];
                    sum += v;
                    if (use_max && v > mx) {
                        mx = v;
                        arg = chunk_base + slot - start;
                    }
                }
                s_pool[c] += sum;
                if (use_max) {
                    s_max[c] = mx;
                    s_argmax[c] = arg;
                }
            }
        }
    }

    for (int i = threadIdx.x; i < SIG_SLOTS * SIG_W; i += blockDim.x) {
        int f = i / SIG_W;
        int z = 0, acc = 0;
        for (; z < SIG_ZONES; ++z) {
            if (f < acc + SIG_ZONE_CAPS[z]) break;
            acc += SIG_ZONE_CAPS[z];
        }
        if (z >= SIG_ZONES) continue;
        if (f - acc >= counts[z + 1])
            pooled[b * BA_TOTAL + SIG_OFFSET + i] = from_float(0.0f);
    }
    for (int i = threadIdx.x; i < BA_SLOT_FEATURES; i += blockDim.x) {
        int f = i / BA_SLOT_W;
        int d = i % BA_SLOT_W;
        int z = 0, acc = 0;
        for (; z < BA_POOL_SECTIONS; ++z) {
            if (f < acc + BA_SLOT_CAPS[z]) break;
            acc += BA_SLOT_CAPS[z];
        }
        if (z >= BA_POOL_SECTIONS) continue;
        int local = f - acc;
        if (local >= counts[z]) {
            pooled[b * BA_TOTAL + BA_POOLED + i] = from_float(0.0f);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < BA_POOL_SECTIONS * BA_TOKEN_W; i += blockDim.x)
        pooled[b * BA_TOTAL + BA_FIXED + i] = from_float(s_pool[i]);
    for (int i = threadIdx.x; i < BA_POOL_SECTIONS * BA_TOKEN_W; i += blockDim.x) {
        int64_t off = (int64_t)b * BA_TOTAL + BA_MAX_OFFSET + i;
        pooled[off] = use_max ? from_float(s_max[i]) : from_float(0.0f);
        if (use_max) {
            argmax_out[(int64_t)b * BA_ARGS + i] = s_argmax[i];
        }
    }
}

__global__ void __launch_bounds__(256, 4) ba_token_backward_kernel(
        float* __restrict__ center_wgrad, float* __restrict__ blind_wgrad,
        float* __restrict__ tag_wgrad,
        float* __restrict__ partials,
        const precision_t* __restrict__ d_pooled,
        const unsigned char* __restrict__ obs,
        const precision_t* __restrict__ center_embed,
        const precision_t* __restrict__ c_w, const precision_t* __restrict__ c_b,
        const int* __restrict__ argmax, int use_max,
        int B, int obs_size) {
    static constexpr int CELLS_C = BA_CELLS_C;
    static constexpr int CELLS = BA_TOKEN_CELLS;
    __shared__ float s_x[64][BA_CARD_IN];
    __shared__ float s_dh[64][BA_TOKEN_W];
    __shared__ int s_section[64];
    __shared__ int s_center[64];
    __shared__ int s_local[64];
    __shared__ int s_argmax[BA_POOL_SECTIONS * BA_TOKEN_W];

    int b0 = blockIdx.x * BA_BLOCK_AGENTS;
    float acc[((CELLS + 255) / 256)] = {0};
    const int d = threadIdx.x & (BA_TOKEN_W - 1);
    const int k0 = threadIdx.x >> 5;

    for (int g = 0; g < BA_BLOCK_AGENTS; ++g) {
        int b = b0 + g;
        if (b >= B) break;
        int64_t in = (int64_t)b * obs_size;
        const precision_t* dp = d_pooled + (int64_t)b * BA_TOTAL;
        if (threadIdx.x == 0) {
            int base = offsetof(Observation, globals);
            int deck = ba_u16(obs, in, base + offsetof(ObservationGlobals, deck_id));
            int blind = ba_u16(obs, in, base + offsetof(ObservationGlobals, blind_id));
            int boss = ba_u16(obs, in, base + offsetof(ObservationGlobals, next_boss_id));
            int voucher = ba_u16(obs, in, base + offsetof(ObservationGlobals, next_voucher_id));
            int tarot = ba_u16(obs, in, base + offsetof(ObservationGlobals, last_tarot_planet));
            for (int k = 0; k < BA_CENTER_EMBED; ++k) {
                if (deck > 0 && deck < CENTER_COUNT)
                    atomicAdd(&center_wgrad[deck * BA_CENTER_EMBED + k], to_float(dp[k]));
                if (voucher > 0 && voucher < CENTER_COUNT)
                    atomicAdd(&center_wgrad[voucher * BA_CENTER_EMBED + k], to_float(dp[16 + k]));
                if (tarot > 0 && tarot < CENTER_COUNT)
                    atomicAdd(&center_wgrad[tarot * BA_CENTER_EMBED + k], to_float(dp[24 + k]));
            }
            for (int k = 0; k < BA_BLIND_EMBED; ++k) {
                if (blind > 0 && blind < BLIND_COUNT)
                    atomicAdd(&blind_wgrad[blind * BA_BLIND_EMBED + k], to_float(dp[8 + k]));
                if (boss > 0 && boss < BLIND_COUNT)
                    atomicAdd(&blind_wgrad[boss * BA_BLIND_EMBED + k], to_float(dp[12 + k]));
            }
            for (int s = 0; s < 2; ++s) {
                int tag_id = ba_byte(obs, in, base + offsetof(ObservationGlobals, blind_tags) + s);
                if (tag_id > 0 && tag_id < TAG_COUNT) {
                    for (int k = 0; k < BA_TAG_EMBED; ++k)
                        atomicAdd(&tag_wgrad[tag_id * BA_TAG_EMBED + k],
                            to_float(dp[BA_OFF_TAGS + 1 + s * (BA_TAG_EMBED + 2) + k]));
                }
            }
        }
        int counts[BA_POOL_SECTIONS];
        ba_live_counts(obs, in, counts);
        int total = ba_byte(obs, in, offsetof(Observation, total_tokens));
        if (total > MAX_OBS_TOKENS) total = MAX_OBS_TOKENS;
        if (use_max) {
            for (int i = threadIdx.x; i < BA_POOL_SECTIONS * BA_TOKEN_W; i += blockDim.x)
                s_argmax[i] = argmax[(int64_t)b * BA_ARGS + i];
        }
        for (int base_idx = 0; base_idx < total; base_idx += 64) {
            int n = total - base_idx < 64 ? total - base_idx : 64;
            for (int slot = threadIdx.x >> 2; slot < n; slot += blockDim.x >> 2) {
                int part = threadIdx.x & 3;
                int local;
                int section = ba_slot_section(counts, base_idx + slot, &local);
                int center;
                ba_token_input(obs, in, base_idx + slot, center_embed,
                    s_x[slot], &center, &local);
                if (part == 0) {
                    s_section[slot] = section;
                    s_center[slot] = center;
                    s_local[slot] = local;
                }
            }
            __syncthreads();
            for (int slot = threadIdx.x >> 2; slot < n; slot += blockDim.x >> 2) {
                int part = threadIdx.x & 3;
                int section = s_section[slot];
                int center = s_center[slot];
                const precision_t* w = c_w;
                const precision_t* bias = c_b;
                float accv[BA_DIMS_PER_PART];
                for (int d = 0; d < BA_DIMS_PER_PART; ++d)
                    accv[d] = to_float(bias[part * BA_DIMS_PER_PART + d]);
                using Pair = short __attribute__((ext_vector_type(2)));
                int k = 0;
                for (; k + 1 < BA_CARD_IN; k += 2) {
                    Pair xv = {
                        __builtin_bit_cast(short, from_float(s_x[slot][k])),
                        __builtin_bit_cast(short, from_float(s_x[slot][k + 1])),
                    };
                    for (int d = 0; d < BA_DIMS_PER_PART; ++d) {
                        int dim = part * BA_DIMS_PER_PART + d;
                        Pair wv = {
                            __builtin_bit_cast(short, w[dim * BA_CARD_IN + k]),
                            __builtin_bit_cast(short, w[dim * BA_CARD_IN + k + 1]),
                        };
                        accv[d] = __builtin_amdgcn_fdot2_f32_bf16(xv, wv, accv[d], false);
                    }
                }
                if (k < BA_CARD_IN) {
                    float xk = to_float(from_float(s_x[slot][k]));
                    for (int d = 0; d < BA_DIMS_PER_PART; ++d) {
                        int dim = part * BA_DIMS_PER_PART + d;
                        accv[d] += xk * to_float(w[dim * BA_CARD_IN + k]);
                    }
                }
                for (int d = 0; d < BA_DIMS_PER_PART; ++d) {
                    int dim = part * BA_DIMS_PER_PART + d;
                    float dh = to_float(dp[BA_FIXED + section * BA_TOKEN_W + dim]);
                    int local = s_local[slot];
                    if (dim < BA_SLOT_W && local < BA_SLOT_CAPS[section])
                        dh += to_float(dp[BA_POOLED
                            + (BA_SLOT_OFFSET[section] + local) * BA_SLOT_W + dim]);
                    if (use_max && local == s_argmax[section * BA_TOKEN_W + dim])
                        dh += to_float(dp[BA_MAX_OFFSET + section * BA_TOKEN_W + dim]);
                    s_dh[slot][dim] = dh * ba_gelu_deriv(accv[d]);
                }
                if (center >= 0 && center < CENTER_COUNT) {
                    float dxe[BA_CENTER_EMBED] = {0};
                    for (int d = 0; d < BA_DIMS_PER_PART; ++d) {
                        int dim = part * BA_DIMS_PER_PART + d;
                        float dh = s_dh[slot][dim];
                        for (int k = 0; k < BA_CENTER_EMBED; ++k)
                            dxe[k] += to_float(w[dim * BA_CARD_IN + k]) * dh;
                    }
                    for (int k = 0; k < BA_CENTER_EMBED; ++k)
                        atomicAdd(&center_wgrad[center * BA_CENTER_EMBED + k], dxe[k]);
                }
            }
            __syncthreads();
            float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int slot = 0; slot < n; ++slot) {
                float dh = s_dh[slot][d];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int k = k0 + (j << 3);
                    if (k < BA_CARD_IN)
                        sums[j] += to_float(from_float(s_x[slot][k])) * dh;
                }
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int k = k0 + (j << 3);
                if (k < BA_CARD_IN)
                    acc[(k * BA_TOKEN_W + d) >> 8] += sums[j];
            }
            if (threadIdx.x < BA_TOKEN_W) {
                float bsum = 0.0f;
                for (int slot = 0; slot < n; ++slot)
                    bsum += s_dh[slot][threadIdx.x];
                acc[(CELLS_C + threadIdx.x) >> 8] += bsum;
            }
            __syncthreads();
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int k = k0 + (j << 3);
        if (k < BA_CARD_IN) {
            int cell = k * BA_TOKEN_W + d;
            partials[blockIdx.x * BA_TOKEN_CELLS + cell] = acc[cell >> 8];
        }
    }
    if (threadIdx.x < BA_TOKEN_W) {
        int cell = CELLS_C + threadIdx.x;
        partials[blockIdx.x * BA_TOKEN_CELLS + cell] = acc[cell >> 8];
    }
}

__global__ void ba_float_to_precision_kernel(
        precision_t* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = from_float(src[idx]);
}

__global__ void ba_grad_finalize_kernel(
        precision_t* __restrict__ c_wgrad, precision_t* __restrict__ c_bgrad,
        const float* __restrict__ partials, int num_blocks) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= BA_TOKEN_CELLS) return;
    float sum = 0.0f;
    const float* p = partials + cell;
    int b = 0;
    for (; b + 7 < num_blocks; b += 8, p += 8 * BA_TOKEN_CELLS) {
        float s0 = p[0];
        float s1 = p[BA_TOKEN_CELLS];
        float s2 = p[2 * BA_TOKEN_CELLS];
        float s3 = p[3 * BA_TOKEN_CELLS];
        float s4 = p[4 * BA_TOKEN_CELLS];
        float s5 = p[5 * BA_TOKEN_CELLS];
        float s6 = p[6 * BA_TOKEN_CELLS];
        float s7 = p[7 * BA_TOKEN_CELLS];
        sum += (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
    }
    for (; b < num_blocks; ++b, p += BA_TOKEN_CELLS) sum += *p;
    if (cell < BA_CELLS_C) {
        c_wgrad[(cell & 31) * BA_CARD_IN + (cell >> 5)] = from_float(sum);
    } else {
        int d = cell - BA_CELLS_C;
        c_bgrad[d] = from_float(sum);
    }
}

static BalatroEncoderWeights* ba_encoder_create(int obs_size, int hidden) {
    assert(obs_size == (int)sizeof(Observation)
        && "Balatro encoder expects packed Observation bytes");
    BalatroEncoderWeights* ew =
        (BalatroEncoderWeights*)calloc(1, sizeof(BalatroEncoderWeights));
    ew->obs_size = obs_size;
    ew->hidden = hidden;
    return ew;
}

static Prec ba_encoder_forward(
        void* w, void* activations, PolicyObs input, cudaStream_t stream) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    BalatroEncoderActivations* a = (BalatroEncoderActivations*)activations;
    int B = input.shape[0];
    a->obs_data = input.data;
    a->obs_batch = B;
    ba_encode_kernel<<<B, BLOCK_SIZE, 0, stream>>>(
        a->pooled.data, input.data,
        ew->center_embed.data, ew->blind_embed.data,
        ew->tag_embed.data,
        ew->c_w.data, ew->c_b.data, a->argmax.data, ba_use_max_pool,
        B, ew->obs_size);
    puf_mm(&a->pooled, &ew->proj_w, &a->out, stream);
    return a->out;
}

static void ba_encoder_backward(
        void* w, void* activations, Prec grad, cudaStream_t stream) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    BalatroEncoderActivations* a = (BalatroEncoderActivations*)activations;
    int B = grad.shape[0];
    puf_mm_tn(&grad, &a->pooled, &a->proj_wgrad, stream);
    puf_mm_nn(&grad, &ew->proj_w, &a->d_pooled, stream);
    int embed_n = CENTER_COUNT * BA_CENTER_EMBED;
    int blind_n = BLIND_COUNT * BA_BLIND_EMBED;
    int tag_n = TAG_COUNT * BA_TAG_EMBED;
    cudaMemsetAsync(a->center_wgrad_f.data, 0, embed_n * sizeof(float), stream);
    cudaMemsetAsync(a->blind_wgrad_f.data, 0, blind_n * sizeof(float), stream);
    cudaMemsetAsync(a->tag_wgrad_f.data, 0, tag_n * sizeof(float), stream);
    int blocks = (B + BA_BLOCK_AGENTS - 1) / BA_BLOCK_AGENTS;
    assert(blocks <= a->token_partial_rows && "backward batch exceeds partials");
    ba_token_backward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        a->center_wgrad_f.data, a->blind_wgrad_f.data, a->tag_wgrad_f.data,
        a->token_partials.data,
        a->d_pooled.data, a->obs_data, ew->center_embed.data,
        ew->c_w.data, ew->c_b.data, a->argmax.data, ba_use_max_pool,
        B, ew->obs_size);
    ba_grad_finalize_kernel<<<grid_size(BA_TOKEN_CELLS), BLOCK_SIZE, 0, stream>>>(
        a->c_wgrad.data, a->c_bgrad.data, a->token_partials.data, blocks);
    ba_float_to_precision_kernel<<<grid_size(embed_n), BLOCK_SIZE, 0, stream>>>(
        a->center_wgrad.data, a->center_wgrad_f.data, embed_n);
    ba_float_to_precision_kernel<<<grid_size(blind_n), BLOCK_SIZE, 0, stream>>>(
        a->blind_wgrad.data, a->blind_wgrad_f.data, blind_n);
    ba_float_to_precision_kernel<<<grid_size(tag_n), BLOCK_SIZE, 0, stream>>>(
        a->tag_wgrad.data, a->tag_wgrad_f.data, tag_n);
}

static void ba_encoder_init_weights(
        void* w, uint64_t* seed, cudaStream_t stream) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    puf_normal_init(&ew->center_embed, 0.25f, (*seed)++, stream);
    puf_normal_init(&ew->blind_embed, 0.25f, (*seed)++, stream);
    puf_normal_init(&ew->tag_embed, 0.25f, (*seed)++, stream);
    Prec c_w = {.data = ew->c_w.data, .shape = {BA_TOKEN_W, BA_CARD_IN}};
    puf_kaiming_init(&c_w, sqrtf(2.0f), (*seed)++, stream);
    Prec projection = {
        .data = ew->proj_w.data,
        .shape = {ew->hidden, BA_TOTAL},
    };
    puf_kaiming_init(&projection, sqrtf(2.0f), (*seed)++, stream);
    cudaMemsetAsync(ew->c_b.data, 0, BA_TOKEN_W * sizeof(precision_t), stream);
}

static void ba_encoder_reg_params(void* w, Allocator* alloc) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    ew->center_embed = {.shape = {CENTER_COUNT, BA_CENTER_EMBED}};
    ew->blind_embed = {.shape = {BLIND_COUNT, BA_BLIND_EMBED}};
    ew->tag_embed = {.shape = {TAG_COUNT, BA_TAG_EMBED}};
    ew->c_w = {.shape = {BA_TOKEN_W, BA_CARD_IN}};
    ew->c_b = {.shape = {BA_TOKEN_W}};
    ew->proj_w = {.shape = {ew->hidden, BA_TOTAL}};
    alloc_register(alloc, &ew->center_embed);
    alloc_register(alloc, &ew->blind_embed);
    alloc_register(alloc, &ew->tag_embed);
    alloc_register(alloc, &ew->c_w);
    alloc_register(alloc, &ew->c_b);
    alloc_register(alloc, &ew->proj_w);
}

static void ba_encoder_reg_train(
        void* w, void* activations, Allocator* acts,
        Allocator* grads, int B_TT) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    BalatroEncoderActivations* a = (BalatroEncoderActivations*)activations;
    *a = {};
    a->pooled = {.shape = {B_TT, BA_TOTAL}};
    a->out = {.shape = {B_TT, ew->hidden}};
    a->d_pooled = {.shape = {B_TT, BA_TOTAL}};
    a->argmax = {.shape = {B_TT, BA_ARGS}};
    a->proj_wgrad = {.shape = {ew->hidden, BA_TOTAL}};
    a->center_wgrad_f = {.shape = {CENTER_COUNT, BA_CENTER_EMBED}};
    a->blind_wgrad_f = {.shape = {BLIND_COUNT, BA_BLIND_EMBED}};
    a->tag_wgrad_f = {.shape = {TAG_COUNT, BA_TAG_EMBED}};
    a->c_wgrad = {.shape = {BA_CARD_IN, BA_TOKEN_W}};
    a->c_bgrad = {.shape = {BA_TOKEN_W}};
    a->center_wgrad = {.shape = {CENTER_COUNT, BA_CENTER_EMBED}};
    a->blind_wgrad = {.shape = {BLIND_COUNT, BA_BLIND_EMBED}};
    a->tag_wgrad = {.shape = {TAG_COUNT, BA_TAG_EMBED}};
    a->token_partial_rows = (B_TT + BA_BLOCK_AGENTS - 1) / BA_BLOCK_AGENTS;
    a->token_partials = {.shape = {a->token_partial_rows, BA_TOKEN_CELLS}};
    alloc_register(acts, &a->pooled);
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->d_pooled);
    alloc_register(acts, &a->argmax);
    alloc_register(acts, &a->center_wgrad_f);
    alloc_register(acts, &a->blind_wgrad_f);
    alloc_register(acts, &a->tag_wgrad_f);
    alloc_register(acts, &a->token_partials);
    alloc_register(grads, &a->center_wgrad);
    alloc_register(grads, &a->blind_wgrad);
    alloc_register(grads, &a->tag_wgrad);
    alloc_register(grads, &a->c_wgrad);
    alloc_register(grads, &a->c_bgrad);
    alloc_register(grads, &a->proj_wgrad);
}

static void ba_encoder_reg_rollout(
        void* w, void* activations, Allocator* alloc, int B) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    BalatroEncoderActivations* a = (BalatroEncoderActivations*)activations;
    *a = {};
    a->pooled = {.shape = {B, BA_TOTAL}};
    a->out = {.shape = {B, ew->hidden}};
    a->argmax = {.shape = {B, BA_ARGS}};
    alloc_register(alloc, &a->pooled);
    alloc_register(alloc, &a->out);
    alloc_register(alloc, &a->argmax);
}

static void* ba_encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    return ba_encoder_create(e->in_dim, e->out_dim);
}

static void create_balatro_encoder(Encoder* enc) {
    *enc = Encoder{
        .forward = ba_encoder_forward,
        .backward = ba_encoder_backward,
        .init_weights = ba_encoder_init_weights,
        .reg_params = ba_encoder_reg_params,
        .reg_train = ba_encoder_reg_train,
        .reg_rollout = ba_encoder_reg_rollout,
        .create_weights = ba_encoder_create_weights,
        .in_dim = enc->in_dim,
        .out_dim = enc->out_dim,
        .activation_size = sizeof(BalatroEncoderActivations),
    };
}
