// Balatro GPU encoder: sparse live-slot token pooling.
//
// Decodes only live records from the packed Observation (no fixed-capacity
// slot scans), resolves hand slots into their referenced variant attributes,
// pools variant/hand/card tokens permutation-invariantly per section, and
// projects the concatenated section vectors once to the policy hidden size.
// The 300-bit redeemed-voucher expansion is replaced by a 32-bit mask decode
// (the voucher vocabulary is the contiguous center range 268..299), and the
// redundant deck summaries are not decoded at all.

static constexpr int BA_CENTER_EMBED = 8;
static constexpr int BA_BLIND_EMBED = 4;
static constexpr int BA_TAG_EMBED = 4;
static constexpr int BA_TOKEN_W = 32;
static constexpr int BA_DIMS_PER_PART = BA_TOKEN_W / 4;    // pooled width per token section
static constexpr int BA_VARIANT_IN = 11; // rank, suit, enh, edition, seal, flags, perma, 4 counts
static constexpr int BA_HAND_IN = 9;     // variant attrs (7), position, flags
static constexpr int BA_CARD_IN = 28;    // center embed (8), attrs (6), q8_8 fields (7), pos, zone (6)
static constexpr int BA_ZONES = 6;       // jokers, consumables, shop, vouchers, boosters, pack

// Pooled concatenation layout (fixed sections + 8 pooled sections).
static constexpr int BA_OFF_GLOBALS = 0;
static constexpr int BA_GLOBALS_FEATURES = 123; // 32 ids + 6 phase + 24 hand one-hots + 30 u8 + 2 u32 + 13 u16 + 16 q8_8
static constexpr int BA_OFF_VOUCHER = BA_OFF_GLOBALS + BA_GLOBALS_FEATURES;
static constexpr int BA_VOUCHER_FEATURES = 32;
static constexpr int BA_OFF_COUNTS = BA_OFF_VOUCHER + BA_VOUCHER_FEATURES;
static constexpr int BA_COUNT_FEATURES = 7; // hand + 6 card zones
static constexpr int BA_OFF_POKER = BA_OFF_COUNTS + BA_COUNT_FEATURES;
static constexpr int BA_POKER_FEATURES = HAND_COUNT * 6;
static constexpr int BA_OFF_TAGS = BA_OFF_POKER + BA_POKER_FEATURES;
static constexpr int BA_TAG_FEATURES = 1 + OBS_MAX_TAGS * (BA_TAG_EMBED + 2);
static constexpr int BA_FIXED = BA_OFF_TAGS + BA_TAG_FEATURES;
static constexpr int BA_POOL_SECTIONS = 2 + BA_ZONES; // variants, hand, 6 card zones
static constexpr int BA_POOLED = BA_FIXED + BA_POOL_SECTIONS * BA_TOKEN_W;

// Per-slot sidecar: each live token's 32-dim embedding exposed per slot for
// the zones the AR heads target by index (hand, jokers, consumables, shop,
// vouchers, boosters, pack). The pooled sections stay permutation-invariant;
// the sidecar restores 1-to-1 slot identity so sell/use/play/reorder heads
// can ground their targets. Caps sized to the game's realistic slot maxima
// (hand 8 + growth, jokers 5 + negative-edition slot, consumables 2, shop 4,
// vouchers 2, boosters 2, pack 5); live slots beyond a cap are zero-padded
// (the pooled stats still cover them).
// Caps cover the game's growth mechanics: hand size can reach ~16 (base 8 +
// Juggler +1, Troubadour +2, Turtle Bean +5), joker slots grow past 5 with
// negative-edition jokers (+1 each), consumables likewise.
static constexpr int BA_SLOT_HAND = 16;
static constexpr int BA_SLOT_JOKERS = 8;
static constexpr int BA_SLOT_CONSUMABLES = 4;
static constexpr int BA_SLOT_SHOP = 4;
static constexpr int BA_SLOT_VOUCHERS = 2;
static constexpr int BA_SLOT_BOOSTERS = 2;
static constexpr int BA_SLOT_PACK = 5;
// Section -> sidecar slot cap / cumulative slot offset. Section 0 (variants)
// has no sidecar; hand is section 1, then the six card zones.
static constexpr int BA_SLOT_CAPS[BA_POOL_SECTIONS] = {
    0, BA_SLOT_HAND, BA_SLOT_JOKERS, BA_SLOT_CONSUMABLES,
    BA_SLOT_SHOP, BA_SLOT_VOUCHERS, BA_SLOT_BOOSTERS, BA_SLOT_PACK};
static constexpr int BA_SLOT_OFFSET[BA_POOL_SECTIONS] = {
    0, 0, BA_SLOT_HAND,
    BA_SLOT_HAND + BA_SLOT_JOKERS,
    BA_SLOT_HAND + BA_SLOT_JOKERS + BA_SLOT_CONSUMABLES,
    BA_SLOT_HAND + BA_SLOT_JOKERS + BA_SLOT_CONSUMABLES + BA_SLOT_SHOP,
    BA_SLOT_HAND + BA_SLOT_JOKERS + BA_SLOT_CONSUMABLES + BA_SLOT_SHOP
        + BA_SLOT_VOUCHERS,
    BA_SLOT_HAND + BA_SLOT_JOKERS + BA_SLOT_CONSUMABLES + BA_SLOT_SHOP
        + BA_SLOT_VOUCHERS + BA_SLOT_BOOSTERS};
// Sidecar width: 16 of the token MLP's 32 outputs. The pool still sums all
// 32, but only dims 0-15 get per-slot gradient pressure, so the shared MLP
// learns to put per-slot discriminative signal there. Halves the sidecar's
// write volume and projection cost vs a full 32-dim slot.
static constexpr int BA_SLOT_W = 16;
static constexpr int BA_SLOT_FEATURES =
    (BA_SLOT_HAND + BA_SLOT_JOKERS + BA_SLOT_CONSUMABLES + BA_SLOT_SHOP
        + BA_SLOT_VOUCHERS + BA_SLOT_BOOSTERS + BA_SLOT_PACK) * BA_SLOT_W;
// Full encoder output: pooled sections + per-slot sidecar.
static constexpr int BA_TOTAL = BA_POOLED + BA_SLOT_FEATURES;

static constexpr int BA_BLOCK_AGENTS = 4; // agents per backward-kernel block
// Token-MLP wgrad cells: 3 sections of (in_dim x BA_TOKEN_W) plus 3 biases.
static constexpr int BA_CELLS_W = BA_TOKEN_W * BA_VARIANT_IN; // 352
static constexpr int BA_CELLS_H = BA_TOKEN_W * BA_HAND_IN;    // 288
static constexpr int BA_CELLS_C = BA_TOKEN_W * BA_CARD_IN;    // 896
static constexpr int BA_TOKEN_CELLS =
    BA_CELLS_W + BA_CELLS_H + BA_CELLS_C + 3 * BA_TOKEN_W;    // 1632

static_assert(sizeof(Observation) == 8120,
    "Balatro encoder must be updated for the Observation layout");
static_assert(BA_FIXED == 253 && BA_POOLED == 509 && BA_TOTAL == 1165,
    "Balatro encoder pooled layout mismatch");

struct BalatroEncoderWeights {
    Prec center_embed, blind_embed, tag_embed; // (CENTER,8) (BLIND,4) (TAG,4)
    Prec v_w, v_b;  // variant token MLP (64,11) + bias (64)
    Prec h_w, h_b;  // hand token MLP (64,9) + bias
    Prec c_w, c_b;  // card token MLP (64,28) + bias
    Prec proj_w;    // (hidden, BA_TOTAL)
    int obs_size, hidden;
};

struct BalatroEncoderActivations {
    Prec pooled, out, d_pooled, proj_wgrad;
    Float center_wgrad_f, blind_wgrad_f, tag_wgrad_f;
    Prec v_wgrad, v_bgrad, h_wgrad, h_bgrad, c_wgrad, c_bgrad;
    Prec center_wgrad, blind_wgrad, tag_wgrad;
    // Block-private token-MLP wgrad partials: (ceil(B_TT / BA_BLOCK_AGENTS),
    // BA_TOKEN_CELLS). Every block writes every cell; ba_grad_finalize_kernel
    // reduces the rows to the bf16 wgrads without atomics or memset.
    Float token_partials;
    int token_partial_rows;
    const unsigned char* obs_data; // forward stashes the packed-obs tensor for backward
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

// Per-zone obs offsets and capacities for the six card zones.
struct BaZoneInfo {
    int obs_offset;
    int capacity;
};

__device__ __forceinline__ BaZoneInfo ba_card_zone(int zone) {
    switch (zone) {
    case 0: return {offsetof(Observation, jokers), OBS_MAX_JOKERS};
    case 1: return {offsetof(Observation, consumables), OBS_MAX_CONSUMABLES};
    case 2: return {offsetof(Observation, shop), OBS_MAX_SHOP_MAIN};
    case 3: return {offsetof(Observation, shop_vouchers), OBS_MAX_SHOP_VOUCHERS};
    case 4: return {offsetof(Observation, shop_boosters), OBS_MAX_SHOP_BOOSTERS};
    default: return {offsetof(Observation, pack), OBS_MAX_PACK_CARDS};
    }
}

// Live counts per section, clamped to the observation's physical capacities
// (a corrupted count must never drive reads past the packed arrays).
__device__ __forceinline__ void ba_live_counts(
        const unsigned char* obs, int64_t in, int counts[8]) {
    counts[0] = ba_u16(obs, in, offsetof(Observation, variants));
    if (counts[0] > OBS_MAX_PLAYING_VARIANTS) counts[0] = OBS_MAX_PLAYING_VARIANTS;
    counts[1] = ba_u16(obs, in, offsetof(Observation, hand));
    if (counts[1] > OBS_MAX_HAND) counts[1] = OBS_MAX_HAND;
    for (int z = 0; z < BA_ZONES; ++z) {
        BaZoneInfo info = ba_card_zone(z);
        counts[2 + z] = ba_u16(obs, in, info.obs_offset);
        if (counts[2 + z] > info.capacity) counts[2 + z] = info.capacity;
    }
}

// Map a flattened live-slot index to (section, local). Returns -1 past total.
__device__ __forceinline__ int ba_slot_section(
        const int counts[8], int slot, int* local) {
    int cursor = 0;
    for (int s = 0; s < 8; ++s) {
        if (slot < cursor + counts[s]) {
            *local = slot - cursor;
            return s;
        }
        cursor += counts[s];
    }
    return -1;
}

// Token input features for a (section, local) slot. Writes into x (BA_CARD_IN
// floats); center_out receives the card center id (or -1); in_dim the section
// input width. Mirror of the obs decode; also used by the backward kernel.
__device__ __forceinline__ void ba_token_input(
        const unsigned char* obs, int64_t in, int section, int local,
        const precision_t* center_embed, float* x, int* center_out, int* in_dim) {
    if (section == 0) {
        int rec = offsetof(Observation, variants)
            + offsetof(ObservationVariants, values) + local * sizeof(ObservationVariant);
        x[0] = (float)ba_byte(obs, in, rec + 0) / 14.0f;
        x[1] = (float)ba_byte(obs, in, rec + 1) / 3.0f;
        x[2] = (float)ba_byte(obs, in, rec + 2) / 8.0f;
        x[3] = (float)ba_byte(obs, in, rec + 3) / 4.0f;
        x[4] = (float)ba_byte(obs, in, rec + 4) / 4.0f;
        x[5] = (float)ba_byte(obs, in, rec + 5) / 255.0f;
        x[6] = (float)ba_i16(obs, in, rec + 6) / 32768.0f;
        x[7] = (float)ba_u16(obs, in, rec + 8) / 256.0f;
        x[8] = (float)ba_u16(obs, in, rec + 10) / 256.0f;
        x[9] = (float)ba_u16(obs, in, rec + 12) / 256.0f;
        x[10] = (float)ba_u16(obs, in, rec + 14) / 256.0f;
        *center_out = -1;
        *in_dim = BA_VARIANT_IN;
        return;
    }
    if (section == 1) {
        int hb = offsetof(Observation, hand);
        int v = ba_u16(obs, in, hb + 2 + local * sizeof(ObservationHandCard));
        int variants_count = ba_u16(obs, in, offsetof(Observation, variants));
        float vrank = 0.0f, vsuit = 0.0f, venh = 0.0f, ved = 0.0f, vseal = 0.0f;
        float vflags = 0.0f, vperma = 0.0f;
        if (v < variants_count && v < OBS_MAX_PLAYING_VARIANTS) {
            int rec = offsetof(Observation, variants)
                + offsetof(ObservationVariants, values) + v * sizeof(ObservationVariant);
            vrank = (float)ba_byte(obs, in, rec + 0) / 14.0f;
            vsuit = (float)ba_byte(obs, in, rec + 1) / 3.0f;
            venh = (float)ba_byte(obs, in, rec + 2) / 8.0f;
            ved = (float)ba_byte(obs, in, rec + 3) / 4.0f;
            vseal = (float)ba_byte(obs, in, rec + 4) / 4.0f;
            vflags = (float)ba_byte(obs, in, rec + 5) / 255.0f;
            vperma = (float)ba_i16(obs, in, rec + 6) / 32768.0f;
        }
        x[0] = vrank;
        x[1] = vsuit;
        x[2] = venh;
        x[3] = ved;
        x[4] = vseal;
        x[5] = vflags;
        x[6] = vperma;
        x[7] = (float)local / OBS_MAX_HAND;
        x[8] = (float)ba_byte(obs, in, hb + 2 + local * sizeof(ObservationHandCard) + 2) / 255.0f;
        *center_out = -1;
        *in_dim = BA_HAND_IN;
        return;
    }
    int zone = section - 2;
    BaZoneInfo info = ba_card_zone(zone);
    int rec = info.obs_offset + 2 + local * sizeof(ObservationCard);
    int center = ba_u16(obs, in, rec);
    for (int k = 0; k < BA_CENTER_EMBED; ++k)
        x[k] = (center >= 0 && center < CENTER_COUNT)
            ? to_float(center_embed[center * BA_CENTER_EMBED + k]) : 0.0f;
    x[8] = (float)ba_byte(obs, in, rec + 2) / 14.0f;
    x[9] = (float)ba_byte(obs, in, rec + 3) / 3.0f;
    x[10] = (float)ba_byte(obs, in, rec + 4) / 8.0f;
    x[11] = (float)ba_byte(obs, in, rec + 5) / 4.0f;
    x[12] = (float)ba_byte(obs, in, rec + 6) / 4.0f;
    x[13] = (float)ba_byte(obs, in, rec + 7) / 255.0f;
    x[14] = (float)ba_i16(obs, in, rec + 8) / 32768.0f;
    x[15] = (float)ba_i16(obs, in, rec + 10) / 32768.0f;
    x[16] = (float)ba_i16(obs, in, rec + 12) / 32768.0f;
    for (int k = 0; k < 4; ++k)
        x[17 + k] = (float)ba_i16(obs, in, rec + 14 + 2 * k) / 32768.0f;
    x[21] = (float)local / info.capacity;
    for (int k = 0; k < BA_ZONES; ++k) x[22 + k] = (k == zone) ? 1.0f : 0.0f;
    *center_out = center;
    *in_dim = BA_CARD_IN;
}

// Fused forward encoder: one kernel writes the whole pooled row per agent —
// the fixed features (globals, voucher mask, zone counts, poker, tags) and the
// token-pooled sections (two-pass, no atomics). Single launch per forward.
__global__ void ba_encode_kernel(
        precision_t* __restrict__ pooled,
        const unsigned char* __restrict__ obs,
        const precision_t* __restrict__ center_embed,
        const precision_t* __restrict__ blind_embed,
        const precision_t* __restrict__ tag_embed,
        const precision_t* __restrict__ v_w, const precision_t* __restrict__ v_b,
        const precision_t* __restrict__ h_w, const precision_t* __restrict__ h_b,
        const precision_t* __restrict__ c_w, const precision_t* __restrict__ c_b,
        int B, int obs_size) {
    __shared__ float s_tok[64][BA_TOKEN_W];
    __shared__ float s_pool[BA_POOL_SECTIONS * BA_TOKEN_W];
    __shared__ float s_x[64][BA_CARD_IN];
    __shared__ int s_section[64];
    __shared__ int s_local[64];
    int b = blockIdx.x;
    int64_t in = (int64_t)b * obs_size;
    int base = offsetof(Observation, globals);
    int deck = ba_u16(obs, in, base + offsetof(ObservationGlobals, deck_id));
    int blind = ba_u16(obs, in, base + offsetof(ObservationGlobals, blind_id));
    int boss = ba_u16(obs, in, base + offsetof(ObservationGlobals, next_boss_id));
    int voucher = ba_u16(obs, in, base + offsetof(ObservationGlobals, next_voucher_id));
    int tarot = ba_u16(obs, in, base + offsetof(ObservationGlobals, last_tarot_planet));
    // struct-order scalar u8 fields (phase/most/last excluded: one-hotted)
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
        if (feature < BA_OFF_VOUCHER) {
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
                    value = (float)ba_u32(obs, in, base + off) / (local == 0 ? 16.0f : 256.0f);
                } else if ((local -= 2) < 13) {
                    value = (float)ba_u16(obs, in,
                        base + offsetof(ObservationGlobals, hands_left) + 2 * local) / 64.0f;
                } else {
                    local -= 13;
                    value = (float)ba_i16(obs, in,
                        base + offsetof(ObservationGlobals, dollars_q8_8) + 2 * local) / (256.0f * 16.0f);
                }
            }
        } else if (feature < BA_OFF_COUNTS) {
            // 32-bit redeemed-voucher mask: bit i = center CENTER_V_ANTIMATTER + i
            int bit = feature - BA_OFF_VOUCHER;
            value = (float)((ba_u32(obs, in, base + offsetof(ObservationGlobals, redeemed_vouchers)) >> bit) & 1u);
        } else if (feature < BA_OFF_POKER) {
            static constexpr int zone_offsets[BA_COUNT_FEATURES] = {
                offsetof(Observation, hand), offsetof(Observation, jokers),
                offsetof(Observation, consumables), offsetof(Observation, shop),
                offsetof(Observation, shop_vouchers), offsetof(Observation, shop_boosters),
                offsetof(Observation, pack),
            };
            static constexpr int zone_caps[BA_COUNT_FEATURES] = {
                OBS_MAX_HAND, OBS_MAX_JOKERS, OBS_MAX_CONSUMABLES, OBS_MAX_SHOP_MAIN,
                OBS_MAX_SHOP_VOUCHERS, OBS_MAX_SHOP_BOOSTERS, OBS_MAX_PACK_CARDS,
            };
            int i = feature - BA_OFF_COUNTS;
            value = (float)ba_u16(obs, in, zone_offsets[i]) / zone_caps[i];
        } else if (feature < BA_OFF_TAGS) {
            int local = feature - BA_OFF_POKER;
            int rec = local / 6;
            int f = local % 6;
            int record = offsetof(Observation, poker_hands) + rec * sizeof(ObservationPokerHand);
            if (f == 0) value = (float)ba_byte(obs, in, record);
            else if (f == 1) value = (float)ba_u32(obs, in, record + 1) / 16.0f;
            else if (f == 2) value = (float)ba_i16(obs, in, record + 5) / (256.0f * 128.0f);
            else if (f == 3) value = (float)ba_i16(obs, in, record + 7) / (256.0f * 128.0f);
            else if (f == 4) value = (float)ba_u32(obs, in, record + 9) / 256.0f;
            else value = (float)ba_u32(obs, in, record + 13) / 64.0f;
        } else {
            int local = feature - BA_OFF_TAGS;
            int base_tag = offsetof(Observation, tags);
            if (local == 0) {
                value = (float)ba_u16(obs, in, base_tag) / OBS_MAX_TAGS;
            } else {
                int rec = (local - 1) / (BA_TAG_EMBED + 2);
                int f = (local - 1) % (BA_TAG_EMBED + 2);
                int id = ba_byte(obs, in, base_tag + 2 + rec);
                int active = rec < ba_u16(obs, in, base_tag);
                if (!active) {
                    value = 0.0f;
                } else if (f < BA_TAG_EMBED) {
                    if ((unsigned)id < TAG_COUNT)
                        value = to_float(tag_embed[id * BA_TAG_EMBED + f]);
                } else {
                    int field = f - BA_TAG_EMBED + 1;
                    int offset = base_tag + 2 + field * OBS_MAX_TAGS + rec;
                    value = (float)ba_byte(obs, in, offset)
                        / (field == 1 ? (float)HAND_COUNT : 255.0f);
                }
            }
        }
        pooled[b * BA_TOTAL + feature] = from_float(value);
    }
    for (int i = threadIdx.x; i < BA_POOL_SECTIONS * BA_TOKEN_W; i += blockDim.x)
        s_pool[i] = 0.0f;
    __syncthreads();
    int counts[8];
    ba_live_counts(obs, in, counts);
    int total = 0;
    for (int s = 0; s < 8; ++s) total += counts[s];
    for (int chunk_base = 0; chunk_base < total; chunk_base += 64) {
        int n = total - chunk_base < 64 ? total - chunk_base : 64;
        __syncthreads();
        for (int slot = threadIdx.x >> 2; slot < n; slot += blockDim.x >> 2) {
            // All 4 threads of a slot decode in parallel; part 0 writes the
            // section, each part stores a disjoint k-range of s_x.
            int part = threadIdx.x & 3;
            int local;
            int section = ba_slot_section(counts, chunk_base + slot, &local);
            float x[BA_CARD_IN];
            int center, in_dim;
            ba_token_input(obs, in, section, local, center_embed, x, &center, &in_dim);
            if (part == 0) {
                s_section[slot] = section;
                s_local[slot] = local;
            }
            int per = (in_dim + 3) >> 2;
            int k = part * per;
            int k_end = k + per < in_dim ? k + per : in_dim;
            for (; k < k_end; ++k) s_x[slot][k] = x[k];
        }
        __syncthreads();
        for (int slot = threadIdx.x >> 2; slot < n; slot += blockDim.x >> 2) {
            int part = threadIdx.x & 3;
            int section = s_section[slot];
            int in_dim = section == 0 ? BA_VARIANT_IN
                : section == 1 ? BA_HAND_IN : BA_CARD_IN;
            const precision_t* w = section <= 1 ? (section == 0 ? v_w : h_w) : c_w;
            const precision_t* bias = section <= 1 ? (section == 0 ? v_b : h_b) : c_b;
            for (int d = 0; d < BA_DIMS_PER_PART; ++d) {
                int dim = part * BA_DIMS_PER_PART + d;
                float acc = to_float(bias[dim]);
                for (int k = 0; k < in_dim; ++k)
                    acc += s_x[slot][k] * to_float(w[dim * in_dim + k]);
                s_tok[slot][dim] = ba_gelu(acc);
            }
        }
        __syncthreads();
        // Per-slot sidecar: expose each live slot's first BA_SLOT_W embedding
        // dims so the AR heads can ground index targeting. Slots beyond a
        // zone cap stay zero.
        for (int i = threadIdx.x; i < n * BA_SLOT_W; i += blockDim.x) {
            int slot = i >> 4;
            int d = i & (BA_SLOT_W - 1);
            int sec = s_section[slot];
            int local = s_local[slot];
            if (local >= BA_SLOT_CAPS[sec]) continue;
            pooled[b * BA_TOTAL + BA_POOLED
                + (BA_SLOT_OFFSET[sec] + local) * BA_SLOT_W + d]
                = from_float(s_tok[slot][d]);
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
                for (int slot = lo; slot < hi; ++slot) sum += s_tok[slot][d];
                s_pool[c] += sum;
            }
        }
    }
    // Tail zeroing: the sidecar pass overwrote the live slots; only the
    // [count, cap) tail of each zone can hold stale embeddings from a
    // previous step, so clear just that (typically a couple of slots).
    for (int i = threadIdx.x; i < BA_SLOT_FEATURES; i += blockDim.x) {
        int f = i / BA_SLOT_W;
        int d = i % BA_SLOT_W;
        int z = 1, acc = 0;
        for (; z < BA_POOL_SECTIONS; ++z) {
            if (f < acc + BA_SLOT_CAPS[z]) break;
            acc += BA_SLOT_CAPS[z];
        }
        if (z >= BA_POOL_SECTIONS) continue;
        int local = f - acc;
        if (local >= counts[z])
            pooled[b * BA_TOTAL + BA_POOLED + i] = from_float(0.0f);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < BA_POOL_SECTIONS * BA_TOKEN_W; i += blockDim.x)
        pooled[b * BA_TOTAL + BA_FIXED + i] = from_float(s_pool[i]);
}

// Backward through the token MLPs and the fixed-section embeddings.
// Recomputes each slot's token input from the packed obs; no saved per-slot
// activations. MLP wgrads are computed without atomics: pass 1 decodes each
// slot's x and dh into shared tiles; pass 2 sums the outer product per cell
// in registers; one flush per block. Center-embed gradients stay on global
// atomics (few card slots per agent, measured free).
__global__ void ba_token_backward_kernel(
        float* __restrict__ center_wgrad, float* __restrict__ blind_wgrad,
        float* __restrict__ tag_wgrad,
        float* __restrict__ partials,
        const precision_t* __restrict__ d_pooled,
        const unsigned char* __restrict__ obs,
        const precision_t* __restrict__ center_embed,
        const precision_t* __restrict__ v_w, const precision_t* __restrict__ v_b,
        const precision_t* __restrict__ h_w, const precision_t* __restrict__ h_b,
        const precision_t* __restrict__ c_w, const precision_t* __restrict__ c_b,
        int B, int obs_size) {
    static constexpr int CELLS_W = BA_CELLS_W;
    static constexpr int CELLS_H = BA_CELLS_H;
    static constexpr int CELLS_C = BA_CELLS_C;
    static constexpr int CELLS_BIAS = 3 * BA_TOKEN_W;            // 96
    static constexpr int CELLS = BA_TOKEN_CELLS;
    __shared__ float s_x[64][BA_CARD_IN];
    __shared__ float s_dh[64][BA_TOKEN_W];
    __shared__ int s_section[64];
    __shared__ int s_center[64];
    __shared__ int s_local[64];
    int b0 = blockIdx.x * BA_BLOCK_AGENTS;
    // Per-thread cell accumulators, accumulated across all agents in the
    // block; flushed to global once per block instead of once per agent.
    float acc[((CELLS + 255) / 256)] = {0};
    // Fixed per-thread cell coordinates (k-major layout): d is the pooled
    // dim, k0 the base input index; k walks k0, k0+8, k0+16, k0+24.
    const int d = threadIdx.x & (BA_TOKEN_W - 1);
    const int k0 = threadIdx.x >> 5;
    for (int g = 0; g < BA_BLOCK_AGENTS; ++g) {
        int b = b0 + g;
        if (b >= B) break;
        int64_t in = (int64_t)b * obs_size;
        const precision_t* dp = d_pooled + (int64_t)b * BA_TOTAL;
        if (threadIdx.x == 0) {
            // Fixed-section embedding gradients (globals ids + tag ids).
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
            int tag_base = offsetof(Observation, tags);
            int tag_count = ba_u16(obs, in, tag_base);
            for (int s = 0; s < OBS_MAX_TAGS; ++s) {
                int id = ba_byte(obs, in, tag_base + 2 + s);
                if (s < tag_count && id > 0 && id < TAG_COUNT) {
                    for (int k = 0; k < BA_TAG_EMBED; ++k)
                        atomicAdd(&tag_wgrad[id * BA_TAG_EMBED + k],
                            to_float(dp[BA_OFF_TAGS + 1 + s * (BA_TAG_EMBED + 2) + k]));
                }
            }
        }
        int counts[8];
        ba_live_counts(obs, in, counts);
        int total = 0;
        for (int s = 0; s < 8; ++s) total += counts[s];
        for (int base = 0; base < total; base += 64) {
            int n = total - base < 64 ? total - base : 64;
            for (int slot = threadIdx.x >> 2; slot < n; slot += blockDim.x >> 2) {
                // All 4 threads of a slot decode in parallel: the obs/embed load
                // chains overlap (same addresses broadcast in the warp) and each
                // part stores a disjoint k-range of s_x, so the decode is latency
                // -bound instead of serialized on one thread per slot.
                int part = threadIdx.x & 3;
                int local;
                int section = ba_slot_section(counts, base + slot, &local);
                float x[BA_CARD_IN];
                int center, in_dim;
                ba_token_input(obs, in, section, local, center_embed, x, &center, &in_dim);
                if (part == 0) {
                    s_section[slot] = section;
                    s_center[slot] = center;
                    s_local[slot] = local;
                }
                int per = (in_dim + 3) >> 2;
                int k = part * per;
                int k_end = k + per < in_dim ? k + per : in_dim;
                for (; k < k_end; ++k) s_x[slot][k] = x[k];
            }
            __syncthreads();
            // Pass 1: run the token MLP with one decode per slot (4 threads/slot).
            for (int slot = threadIdx.x >> 2; slot < n; slot += blockDim.x >> 2) {
                int part = threadIdx.x & 3;
                int section = s_section[slot];
                int center = s_center[slot];
                int in_dim = section == 0 ? BA_VARIANT_IN
                    : section == 1 ? BA_HAND_IN : BA_CARD_IN;
                const precision_t* w = section <= 1 ? (section == 0 ? v_w : h_w) : c_w;
                const precision_t* bias = section <= 1 ? (section == 0 ? v_b : h_b) : c_b;
                for (int d = 0; d < BA_DIMS_PER_PART; ++d) {
                    int dim = part * BA_DIMS_PER_PART + d;
                    float accv = to_float(bias[dim]);
                    for (int k = 0; k < in_dim; ++k)
                        accv += s_x[slot][k] * to_float(w[dim * in_dim + k]);
                    // Gradient of the slot's embedding: the pooled-section term
                    // plus, for sidecar zones, the per-slot term.
                    float dh = to_float(dp[BA_FIXED + section * BA_TOKEN_W + dim]);
                    int local = s_local[slot];
                    if (dim < BA_SLOT_W && local < BA_SLOT_CAPS[section])
                        dh += to_float(dp[BA_POOLED
                            + (BA_SLOT_OFFSET[section] + local) * BA_SLOT_W + dim]);
                    s_dh[slot][dim] = dh * ba_gelu_deriv(accv);
                }
                if (section >= 2 && center >= 0 && center < CENTER_COUNT) {
                    float dxe[BA_CENTER_EMBED] = {0};
                    for (int d = 0; d < BA_DIMS_PER_PART; ++d) {
                        int dim = part * BA_DIMS_PER_PART + d;
                        float dh = s_dh[slot][dim];
                        for (int k = 0; k < BA_CENTER_EMBED; ++k)
                            dxe[k] += to_float(w[dim * in_dim + k]) * dh;
                    }
                    for (int k = 0; k < BA_CENTER_EMBED; ++k)
                        atomicAdd(&center_wgrad[center * BA_CENTER_EMBED + k], dxe[k]);
                }
            }
            __syncthreads();
            // Pass 2: sum only the active section cells for this chunk.
            // k-major cell layout per section (cell = cell_begin + k*32 + d):
            // d is fixed per thread and k walks k0, k0+8, ... so one s_dh load
            // per slot feeds every k-step and no div/mod is required.
            int chunk_end = base + n;
            int section_start = 0;
            int cell_begin = 0;
            for (int sec = 0; sec < 3; ++sec) {
                int section_end = sec == 0 ? counts[0]
                    : sec == 1 ? counts[0] + counts[1] : total;
                int cell_end = sec == 0 ? CELLS_W
                    : sec == 1 ? CELLS_W + CELLS_H : CELLS_W + CELLS_H + CELLS_C;
                if (section_end > base && section_start < chunk_end) {
                    int lo = section_start > base ? section_start - base : 0;
                    int hi = section_end < chunk_end ? section_end - base : n;
                    int in_dim = sec == 0 ? BA_VARIANT_IN
                        : sec == 1 ? BA_HAND_IN : BA_CARD_IN;
                    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (int slot = lo; slot < hi; ++slot) {
                        float dh = s_dh[slot][d];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            int k = k0 + (j << 3);
                            if (k < in_dim) sums[j] += s_x[slot][k] * dh;
                        }
                    }
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int k = k0 + (j << 3);
                        if (k < in_dim)
                            acc[(cell_begin + (k << 5) + d) >> 8] += sums[j];
                    }
                    if (threadIdx.x < BA_TOKEN_W) {
                        // Bias cell for dim threadIdx.x (threads 0..31).
                        int bias_begin = CELLS_W + CELLS_H + CELLS_C + sec * BA_TOKEN_W;
                        float bsum = 0.0f;
                        for (int slot = lo; slot < hi; ++slot)
                            bsum += s_dh[slot][threadIdx.x];
                        acc[(bias_begin + threadIdx.x) >> 8] += bsum;
                    }
                }
                section_start = section_end;
                cell_begin = cell_end;
            }
            __syncthreads();
        }
    }
    // Flush the block-accumulated cells to per-block partials (a warp covers
    // 32 consecutive cells of one k-step, so each store is a full 128-byte
    // segment). ba_grad_finalize_kernel reduces the rows to the bf16 wgrads
    // without atomics or memset.
    for (int sec = 0; sec < 3; ++sec) {
        int in_dim = sec == 0 ? BA_VARIANT_IN : sec == 1 ? BA_HAND_IN : BA_CARD_IN;
        int cell_begin = sec == 0 ? 0 : sec == 1 ? CELLS_W : CELLS_W + CELLS_H;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int k = k0 + (j << 3);
            if (k < in_dim) {
                int cell = cell_begin + (k << 5) + d;
                partials[blockIdx.x * BA_TOKEN_CELLS + cell] = acc[cell >> 8];
            }
        }
    }
    if (threadIdx.x < BA_TOKEN_W) {
        #pragma unroll
        for (int sec = 0; sec < 3; ++sec) {
            int cell = CELLS_W + CELLS_H + CELLS_C + sec * BA_TOKEN_W + threadIdx.x;
            partials[blockIdx.x * BA_TOKEN_CELLS + cell] = acc[cell >> 8];
        }
    }
}

__global__ void ba_float_to_precision_kernel(
        precision_t* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = from_float(src[idx]);
}

// Reduce the per-block token-MLP wgrad partials to the final bf16 gradients.
// One thread per cell sums the num_blocks rows (coalesced: a warp reads 32
// consecutive cells of one block row), then converts in place. Replaces the
// per-backward 6 memsets + 6 converts + the atomic flush of the token kernel.
__global__ void ba_grad_finalize_kernel(
        precision_t* __restrict__ v_wgrad, precision_t* __restrict__ v_bgrad,
        precision_t* __restrict__ h_wgrad, precision_t* __restrict__ h_bgrad,
        precision_t* __restrict__ c_wgrad, precision_t* __restrict__ c_bgrad,
        const float* __restrict__ partials, int num_blocks) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= BA_TOKEN_CELLS) return;
    // 8 independent accumulator chains keep a full row of loads in flight;
    // a dependent per-row add would stall on DRAM latency at this occupancy.
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
    // k-major cell layout: k = cell >> 5, d = cell & 31.
    if (cell < BA_CELLS_W) {
        v_wgrad[(cell & 31) * BA_VARIANT_IN + (cell >> 5)] = from_float(sum);
    } else if (cell < BA_CELLS_W + BA_CELLS_H) {
        int local = cell - BA_CELLS_W;
        h_wgrad[(local & 31) * BA_HAND_IN + (local >> 5)] = from_float(sum);
    } else if (cell < BA_CELLS_W + BA_CELLS_H + BA_CELLS_C) {
        int local = cell - BA_CELLS_W - BA_CELLS_H;
        c_wgrad[(local & 31) * BA_CARD_IN + (local >> 5)] = from_float(sum);
    } else {
        int local = cell - BA_CELLS_W - BA_CELLS_H - BA_CELLS_C;
        int sec = local >> 5;
        int d = local & 31;
        precision_t* bg = sec == 0 ? v_bgrad : sec == 1 ? h_bgrad : c_bgrad;
        bg[d] = from_float(sum);
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
        a->pooled.data, input.data, ew->center_embed.data, ew->blind_embed.data,
        ew->tag_embed.data, ew->v_w.data, ew->v_b.data, ew->h_w.data, ew->h_b.data,
        ew->c_w.data, ew->c_b.data, B, ew->obs_size);
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
        ew->v_w.data, ew->v_b.data, ew->h_w.data, ew->h_b.data,
        ew->c_w.data, ew->c_b.data, B, ew->obs_size);
    ba_grad_finalize_kernel<<<grid_size(BA_TOKEN_CELLS), BLOCK_SIZE, 0, stream>>>(
        a->v_wgrad.data, a->v_bgrad.data, a->h_wgrad.data, a->h_bgrad.data,
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
    Prec v_w = {.data = ew->v_w.data, .shape = {BA_TOKEN_W, BA_VARIANT_IN}};
    puf_kaiming_init(&v_w, sqrtf(2.0f), (*seed)++, stream);
    Prec h_w = {.data = ew->h_w.data, .shape = {BA_TOKEN_W, BA_HAND_IN}};
    puf_kaiming_init(&h_w, sqrtf(2.0f), (*seed)++, stream);
    Prec c_w = {.data = ew->c_w.data, .shape = {BA_TOKEN_W, BA_CARD_IN}};
    puf_kaiming_init(&c_w, sqrtf(2.0f), (*seed)++, stream);
    Prec projection = {
        .data = ew->proj_w.data,
        .shape = {ew->hidden, BA_TOTAL},
    };
    puf_kaiming_init(&projection, sqrtf(2.0f), (*seed)++, stream);
    cudaMemsetAsync(ew->v_b.data, 0, BA_TOKEN_W * sizeof(precision_t), stream);
    cudaMemsetAsync(ew->h_b.data, 0, BA_TOKEN_W * sizeof(precision_t), stream);
    cudaMemsetAsync(ew->c_b.data, 0, BA_TOKEN_W * sizeof(precision_t), stream);
}

static void ba_encoder_reg_params(void* w, Allocator* alloc) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    ew->center_embed = {.shape = {CENTER_COUNT, BA_CENTER_EMBED}};
    ew->blind_embed = {.shape = {BLIND_COUNT, BA_BLIND_EMBED}};
    ew->tag_embed = {.shape = {TAG_COUNT, BA_TAG_EMBED}};
    ew->v_w = {.shape = {BA_TOKEN_W, BA_VARIANT_IN}};
    ew->v_b = {.shape = {BA_TOKEN_W}};
    ew->h_w = {.shape = {BA_TOKEN_W, BA_HAND_IN}};
    ew->h_b = {.shape = {BA_TOKEN_W}};
    ew->c_w = {.shape = {BA_TOKEN_W, BA_CARD_IN}};
    ew->c_b = {.shape = {BA_TOKEN_W}};
    ew->proj_w = {.shape = {ew->hidden, BA_TOTAL}};
    alloc_register(alloc, &ew->center_embed);
    alloc_register(alloc, &ew->blind_embed);
    alloc_register(alloc, &ew->tag_embed);
    alloc_register(alloc, &ew->v_w);
    alloc_register(alloc, &ew->v_b);
    alloc_register(alloc, &ew->h_w);
    alloc_register(alloc, &ew->h_b);
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
    a->proj_wgrad = {.shape = {ew->hidden, BA_TOTAL}};
    a->center_wgrad_f = {.shape = {CENTER_COUNT, BA_CENTER_EMBED}};
    a->blind_wgrad_f = {.shape = {BLIND_COUNT, BA_BLIND_EMBED}};
    a->tag_wgrad_f = {.shape = {TAG_COUNT, BA_TAG_EMBED}};
    a->v_wgrad = {.shape = {BA_VARIANT_IN, BA_TOKEN_W}};
    a->v_bgrad = {.shape = {BA_TOKEN_W}};
    a->h_wgrad = {.shape = {BA_HAND_IN, BA_TOKEN_W}};
    a->h_bgrad = {.shape = {BA_TOKEN_W}};
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
    alloc_register(acts, &a->center_wgrad_f);
    alloc_register(acts, &a->blind_wgrad_f);
    alloc_register(acts, &a->tag_wgrad_f);
    alloc_register(acts, &a->token_partials);
    // Gradients must be registered in exactly the params order (reg_params):
    // the optimizer reads each param's gradient at the same flat offset.
    alloc_register(grads, &a->center_wgrad);
    alloc_register(grads, &a->blind_wgrad);
    alloc_register(grads, &a->tag_wgrad);
    alloc_register(grads, &a->v_wgrad);
    alloc_register(grads, &a->v_bgrad);
    alloc_register(grads, &a->h_wgrad);
    alloc_register(grads, &a->h_bgrad);
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
    alloc_register(alloc, &a->pooled);
    alloc_register(alloc, &a->out);
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
