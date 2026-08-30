// Balatro GPU encoder: typed live-token embeddings with pooled state and
// pointer-decoder keys.  The fixed observation fields retain their compact
// byte decoding, while every live item gets an exact-id embedding and a
// shared nonlinear representation.
//
// Decodes live tokens directly from the packed Observation token stream,
// pools tokens permutation-invariantly per section (Hand, Jokers, Consumables,
// Shop, Vouchers, Boosters, Pack), keeps every live key for indexed actions,
// decodes fixed globals and DeckMatrix features, and projects the concatenated
// section vectors directly to the policy hidden size.

#include "joker_signatures.h"
#include "consumable_signatures.h"

static constexpr int BA_RAW_DIM = 28;
static constexpr int BA_KEY_DIM = 32;
static constexpr int BA_ZONES = 7;

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
static constexpr int BA_TAG_FEATURES = 13;                // 1 double tag + 2 * 6 tag features
static constexpr int BA_OFF_BLIND_SIG = BA_OFF_TAGS + BA_TAG_FEATURES;
static constexpr int BA_BLIND_SIG_FEATURES = 32;          // 16 for current blind + 16 for upcoming boss
static constexpr int BA_FIXED = BA_OFF_BLIND_SIG + BA_BLIND_SIG_FEATURES; // 552

// Permutation-invariant set pooling (Mean + Max across 7 zones)
static constexpr int BA_POOL_SECTIONS = 7;                // hand, jokers, consumables, shop, vouchers, boosters, pack
static constexpr int BA_POOLED_FEATURES = BA_POOL_SECTIONS * BA_KEY_DIM * 2;
static constexpr int BA_OFF_POOLED = BA_FIXED;            // 552
static constexpr int BA_TOTAL = BA_OFF_POOLED + BA_POOLED_FEATURES; // 1000
static constexpr int BA_RANK_EMBED_ROWS = 15;
static constexpr int BA_SUIT_EMBED_ROWS = 4;
static constexpr int BA_ZONE_EMBED_ROWS = BA_ZONES;
static constexpr int BA_PRIMARY_QUERY_COUNT = POLICY_PRIMARY_HEADS;
static constexpr int BA_CARD_QUERY_COUNT = 5;
static constexpr int BA_QUERY_COUNT = BA_PRIMARY_QUERY_COUNT + BA_CARD_QUERY_COUNT;
static constexpr int BA_QUERY_DIM = BA_QUERY_COUNT * BA_KEY_DIM;
static constexpr int BA_DEC_ROWS = 23 + POLICY_PRIMARY_HEAD_SIZE + 6 + 5 * 64 + 1;
static constexpr int BA_PRIMARY_OFFSET = 23;
static constexpr int BA_COUNT_OFFSET = BA_PRIMARY_OFFSET + POLICY_PRIMARY_HEAD_SIZE;
static constexpr int BA_CARD_OFFSET = BA_COUNT_OFFSET + 6;
static constexpr int BA_VALUE_OFFSET = BA_CARD_OFFSET + 5 * 64;
static constexpr int BA_LINEAR_ROWS = 30;
static constexpr int BA_LINEAR_PAD = 32;
static constexpr int BA_FUSED_ROWS = BA_LINEAR_PAD + BA_QUERY_DIM;
static constexpr float BA_QUERY_SCALE = 0.1767766952966369f;
static_assert(BA_LINEAR_ROWS <= BA_LINEAR_PAD, "linear decoder rows exceed padding");

static_assert(sizeof(Observation) == 3055,
    "Balatro encoder must be updated for the Observation layout");
static_assert(BA_FIXED == 552 && BA_TOTAL == 1000,
    "Balatro encoder feature layout mismatch");

struct BlindSignature {
    uint8_t debuff_suit;            // 0=Spades, 1=Hearts, 2=Clubs, 3=Diamonds, 255=None
    uint8_t debuff_face;            // The Plant (1)
    uint8_t debuff_played;          // The Pillar (1)
    uint8_t debuff_all_jokers;      // Final Leaf / Amber Acorn (1)
    uint8_t min_play_size;          // The Psychic (5), 1 otherwise
    uint8_t max_hands;              // The Needle (1), 0=unlimited
    uint8_t max_discards;           // The Water (0), 255=unlimited
    uint8_t single_hand_type;       // The Mouth (1)
    uint8_t no_repeat_hands;        // The Eye (1)
    uint8_t base_score_halved;      // The Flint (1)
    uint8_t chips_mult;             // The Wall (2), Violet Vessel (3), 1 otherwise
    uint8_t discard_random_on_play; // The Hook (1)
    uint8_t money_lost_on_play;     // The Tooth ($1/card)
    uint8_t set_money_zero_on_most; // The Ox (1)
    uint8_t cards_drawn_facedown;   // The Mark, The Wheel, The House (1)
    uint8_t hand_size_reduced;      // The Manacle (1)
};

static __device__ __constant__ BlindSignature BLIND_SIGNATURES[31] = {
    /* 0 */                         {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 1: BLIND_BL_ARM */           {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 2: BLIND_BL_BIG */           {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 3: BLIND_BL_CLUB */          {2,   0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 4: BLIND_BL_EYE */           {255, 0, 0, 0, 1, 0, 255, 0, 1, 0, 1, 0, 0, 0, 0, 0},
    /* 5: BLIND_BL_FINAL_ACORN */   {255, 0, 0, 1, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 1, 0},
    /* 6: BLIND_BL_FINAL_BELL */    {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 1, 0, 0, 0, 0},
    /* 7: BLIND_BL_FINAL_HEART */   {255, 0, 0, 1, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 8: BLIND_BL_FINAL_LEAF */    {255, 0, 0, 1, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 9: BLIND_BL_FINAL_VESSEL */  {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 3, 0, 0, 0, 0, 0},
    /* 10: BLIND_BL_FISH */         {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 1, 0},
    /* 11: BLIND_BL_FLINT */        {255, 0, 0, 0, 1, 0, 255, 0, 0, 1, 1, 0, 0, 0, 0, 0},
    /* 12: BLIND_BL_GOAD */         {0,   0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 13: BLIND_BL_HEAD */         {1,   0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 14: BLIND_BL_HOOK */         {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 1, 0, 0, 0, 0},
    /* 15: BLIND_BL_HOUSE */        {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 1, 0},
    /* 16: BLIND_BL_MANACLE */      {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 1},
    /* 17: BLIND_BL_MARK */         {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 1, 0},
    /* 18: BLIND_BL_MOUTH */        {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 19: BLIND_BL_NEEDLE */       {255, 0, 0, 0, 1, 1, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 20: BLIND_BL_OX */           {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 1, 0, 0},
    /* 21: BLIND_BL_PILLAR */       {255, 0, 1, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 22: BLIND_BL_PLANT */        {255, 1, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 23: BLIND_BL_PSYCHIC */      {255, 0, 0, 0, 5, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 24: BLIND_BL_SERPENT */      {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 25: BLIND_BL_SMALL */        {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 26: BLIND_BL_TOOTH */        {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 1, 0, 0, 0},
    /* 27: BLIND_BL_WALL */         {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 2, 0, 0, 0, 0, 0},
    /* 28: BLIND_BL_WATER */        {255, 0, 0, 0, 1, 0, 0,   0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 29: BLIND_BL_WHEEL */        {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 1, 0},
    /* 30: BLIND_BL_WINDOW */       {3,   0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
};

__device__ __forceinline__ float blind_sig_value(const BlindSignature* sig, int d) {
    switch (d) {
    case 0: return sig->debuff_suit < 4 ? ((float)sig->debuff_suit + 1.0f) / 4.0f : 0.0f;
    case 1: return (float)sig->debuff_face;
    case 2: return (float)sig->debuff_played;
    case 3: return (float)sig->debuff_all_jokers;
    case 4: return sig->min_play_size > 1 ? (float)sig->min_play_size / 5.0f : 0.0f;
    case 5: return sig->max_hands > 0 ? (float)sig->max_hands / 4.0f : 0.0f;
    case 6: return sig->max_discards == 0 ? 1.0f : 0.0f;
    case 7: return (float)sig->single_hand_type;
    case 8: return (float)sig->no_repeat_hands;
    case 9: return (float)sig->base_score_halved;
    case 10: return (float)sig->chips_mult / 3.0f;
    case 11: return (float)sig->discard_random_on_play;
    case 12: return (float)sig->money_lost_on_play;
    case 13: return (float)sig->set_money_zero_on_most;
    case 14: return (float)sig->cards_drawn_facedown;
    default: return (float)sig->hand_size_reduced;
    }
}

// Joker signature feature vector: 16 normalized floats from static table row.
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

// Consumable signature feature vector: 16 normalized floats from typed signature table.
__device__ __forceinline__ float consumable_sig_value(const ConsumableSignature* sig, int d) {
    switch (d) {
    case 0: return (float)sig->kind / 30.0f;
    case 1: return (float)sig->use_rule / 8.0f;
    case 2: return (float)sig->target_effect / 5.0f;
    case 3: return (float)sig->target_value / 16.0f;
    case 4: return (float)sig->target_min / 4.0f;
    case 5: return (float)sig->target_max / 4.0f;
    case 6: return (float)sig->target_filter / 2.0f;
    case 7: return (float)sig->planet_hand / 12.0f;
    case 8: return (float)sig->spawn_set / 8.0f;
    case 9: return (float)sig->spawn_count / 4.0f;
    case 10: return (float)sig->spectral_ranks / 16.0f;
    case 11: return (float)sig->joker_effect / 4.0f;
    case 12: return (float)sig->chance_percent / 100.0f;
    case 13: return (float)sig->rarity / 4.0f;
    case 14: return sig->legendary ? 1.0f : 0.0f;
    default: return (float)sig->reset_dollars;
    }
}

// Voucher signature feature vector: 16 normalized floats from center definition.
__device__ __forceinline__ float voucher_sig_value(int center, int d) {
    int idx = center - CENTER_V_ANTIMATTER;
    switch (d) {
    case 0: return (float)(idx + 1) / 34.0f; // Unique voucher index (1..34)
    case 1: return 1.0f;                     // Cost $10 ($10 / 10 = 1.0)
    case 2: return 0.0f;                    // Upgrade relation is carried by id embedding
    case 3: return 0.0f;                    // Upgrade relation is carried by id embedding
    case 4: return (idx == 0) ? 1.0f : 0.0f; // Antimatter (+1 joker slot)
    case 5: return (idx == 6 || idx == 7) ? 1.0f : 0.0f; // Grabber / Nacho Tong (+1 hand)
    case 6: return (idx == 23 || idx == 31) ? 1.0f : 0.0f; // Recyclomancy / Wasteful (+1 discard)
    case 7: return (idx == 2 || idx == 10) ? 1.0f : 0.0f; // Clearance / Liquidation (discount)
    case 8: return (idx == 12 || idx == 27) ? 1.0f : 0.0f; // Seed Money / Money Tree (interest)
    case 9: return (idx == 18 || idx == 19) ? 1.0f : 0.0f; // Paint Brush / Palette (hand size)
    case 10: return (idx == 16 || idx == 17) ? 1.0f : 0.0f; // Overstock (+1 shop slot)
    case 11: return (idx == 24 || idx == 25) ? 1.0f : 0.0f; // Glut / Surplus (reroll discount)
    case 12: return (idx == 4 || idx == 26) ? 1.0f : 0.0f; // Director's Cut / Retcon (boss reroll)
    case 13: return (idx == 7 || idx == 20) ? 1.0f : 0.0f; // Hieroglyph / Petroglyph (-1 ante)
    case 14: return (idx == 14 || idx == 30) ? 1.0f : 0.0f; // Telescope / Observatory (planet boost)
    default: return 1.0f;                    // Active voucher marker
    }
}

// Booster pack signature feature vector: 16 normalized floats from center definition.
__device__ __forceinline__ float booster_sig_value(int center, int d) {
    int kind = (center >= CENTER_P_ARCANA_JUMBO_1 && center <= CENTER_P_ARCANA_NORMAL_4) ? 0 :
               (center >= CENTER_P_BUFFOON_JUMBO_1 && center <= CENTER_P_BUFFOON_NORMAL_2) ? 1 :
               (center >= CENTER_P_CELESTIAL_JUMBO_1 && center <= CENTER_P_CELESTIAL_NORMAL_4) ? 2 :
               (center >= CENTER_P_SPECTRAL_JUMBO_1 && center <= CENTER_P_SPECTRAL_NORMAL_2) ? 3 : 4;
    int is_mega = (center == CENTER_P_ARCANA_MEGA_1 || center == CENTER_P_ARCANA_MEGA_2 ||
                   center == CENTER_P_BUFFOON_MEGA_1 || center == CENTER_P_CELESTIAL_MEGA_1 ||
                   center == CENTER_P_CELESTIAL_MEGA_2 || center == CENTER_P_SPECTRAL_MEGA_1 ||
                   center == CENTER_P_STANDARD_MEGA_1 || center == CENTER_P_STANDARD_MEGA_2);
    int is_jumbo = (center == CENTER_P_ARCANA_JUMBO_1 || center == CENTER_P_ARCANA_JUMBO_2 ||
                    center == CENTER_P_BUFFOON_JUMBO_1 || center == CENTER_P_CELESTIAL_JUMBO_1 ||
                    center == CENTER_P_CELESTIAL_JUMBO_2 || center == CENTER_P_SPECTRAL_JUMBO_1 ||
                    center == CENTER_P_STANDARD_JUMBO_1 || center == CENTER_P_STANDARD_JUMBO_2);
    float cost = is_mega ? 8.0f : (is_jumbo ? 6.0f : 4.0f);
    float choices = is_mega ? 2.0f : 1.0f;
    float options = (kind == 1 || kind == 3)
        ? ((is_mega || is_jumbo) ? 4.0f : 2.0f)
        : ((is_mega || is_jumbo) ? 5.0f : 3.0f);
    switch (d) {
    case 0: return (float)kind / 4.0f;
    case 1: return cost / 10.0f;
    case 2: return choices / 2.0f;
    case 3: return options / 5.0f;
    case 4: return (kind == 0) ? 1.0f : 0.0f; // Arcana (Tarots)
    case 5: return (kind == 1) ? 1.0f : 0.0f; // Buffoon (Jokers)
    case 6: return (kind == 2) ? 1.0f : 0.0f; // Celestial (Planets)
    case 7: return (kind == 3) ? 1.0f : 0.0f; // Spectral (Spectrals)
    case 8: return (kind == 4) ? 1.0f : 0.0f; // Standard (Playing cards)
    case 9: return is_mega ? 1.0f : 0.0f;
    case 10: return is_jumbo ? 1.0f : 0.0f;
    default: return 1.0f;                     // Active booster marker
    }
}

struct BalatroEncoderWeights {
    Prec token_w;       // (BA_KEY_DIM, BA_RAW_DIM)
    Prec token_b;       // (BA_KEY_DIM)
    Prec center_embed;  // (CENTER_COUNT, BA_KEY_DIM)
    Prec rank_embed;    // (15, BA_KEY_DIM)
    Prec suit_embed;    // (4, BA_KEY_DIM)
    Prec zone_embed;    // (7, BA_KEY_DIM)
    Prec proj_w;        // (hidden, BA_TOTAL)
    int obs_size, hidden;
};

struct BalatroEncoderActivations {
    Prec pooled, out, d_pooled;
    Prec keys;
    Int counts, pool_argmax;
    Long center_acc, rank_acc, suit_acc, zone_acc;
    Long token_w_acc, token_b_acc;
    Prec token_wgrad, token_bgrad, center_grad, rank_grad, suit_grad, zone_grad;
    Prec proj_wgrad;
    const unsigned char* obs_data;
    int obs_batch;
};

static BalatroEncoderActivations* ba_enc_last = NULL;
static Prec* ba_ptr_keygrad = NULL;

__device__ __forceinline__ int ba_byte(
        const unsigned char* obs, int64_t base, int offset) {
    return (int)obs[base + offset];
}

__device__ __forceinline__ int ba_u16(
        const unsigned char* obs, int64_t base, int offset) {
    return ba_byte(obs, base, offset)
        | (ba_byte(obs, base, offset + 1) << 8);
}

__device__ __forceinline__ uint32_t ba_u32(
        const unsigned char* obs, int64_t base, int offset) {
    return (uint32_t)ba_byte(obs, base, offset)
        | ((uint32_t)ba_byte(obs, base, offset + 1) << 8)
        | ((uint32_t)ba_byte(obs, base, offset + 2) << 16)
        | ((uint32_t)ba_byte(obs, base, offset + 3) << 24);
}

__device__ __forceinline__ float ba_float(
        const unsigned char* obs, int64_t base, int offset) {
    uint32_t u = ba_u32(obs, base, offset);
    return __builtin_bit_cast(float, u);
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

__device__ __forceinline__ void ba_token_features(
        const unsigned char* obs, int64_t in, int token_idx, float v[BA_RAW_DIM]) {
    #pragma unroll
    for (int d = 0; d < BA_RAW_DIM; ++d) v[d] = 0.0f;
    int base = offsetof(Observation, tokens) + token_idx * sizeof(CardToken);
    int id = ba_u16(obs, in, base + offsetof(CardToken, id));
    int zone = ba_byte(obs, in, base + offsetof(CardToken, zone));
    int enh = ba_byte(obs, in, base + offsetof(CardToken, enhancement));
    int ed = ba_byte(obs, in, base + offsetof(CardToken, edition));
    int seal = ba_byte(obs, in, base + offsetof(CardToken, seal));
    int flags = ba_byte(obs, in, base + offsetof(CardToken, flags));
    int sell_cost = (int8_t)ba_byte(obs, in, base + offsetof(CardToken, sell_cost));
    int perma_bonus = (int16_t)ba_u16(obs, in, base + offsetof(CardToken, perma_bonus));
    int state0 = (int32_t)ba_u32(obs, in, base + offsetof(CardToken, state0));
    int state1 = (int32_t)ba_u32(obs, in, base + offsetof(CardToken, state1));

    int is_playing = (zone == ZONE_HAND) || (id > 300);
    if (is_playing) {
        int rank = id >> 8;
        int suit = id & 0xFF;
        v[0] = (float)rank / 14.0f;
        v[1] = (float)suit / 3.0f;
        v[2] = (float)enh / 8.0f;
        v[3] = (float)ed / 4.0f;
        v[4] = (float)seal / 4.0f;
        v[5] = (float)flags / 255.0f;
        v[6] = (float)sell_cost / 16.0f;
        v[7] = 1.0f;
        for (int k = 0; k < 7; ++k) v[8 + k] = (k == zone) ? 1.0f : 0.0f;
        for (int k = 0; k < 4; ++k) v[15 + k] = (k == suit) ? 1.0f : 0.0f;
        v[19] = (rank >= 11 && rank <= 13) ? 1.0f : 0.0f;
        v[20] = (rank == 14) ? 1.0f : 0.0f;
        v[21] = (rank % 2 == 0) ? 1.0f : 0.0f;
        v[22] = (rank % 2 == 1 && rank != 14) ? 1.0f : 0.0f;
        v[23] = (float)rank / 14.0f;
        v[24] = asinhf((float)perma_bonus / 10.0f) / 3.0f;
        v[25] = (float)perma_bonus / 100.0f;
        v[26] = asinhf((float)state0 / 10.0f) / 3.0f;
        v[27] = asinhf((float)state1 / 10.0f) / 3.0f;
    } else {
        int center = id;
        if (center >= 0 && center < CENTER_COUNT) {
            const ConsumableSignature* csig = &consumable_signatures[center];
            if (csig->kind != CONS_NONE) {
                #pragma unroll
                for (int d = 0; d < 16; ++d) v[d] = consumable_sig_value(csig, d);
            } else if (center >= CENTER_V_ANTIMATTER && center <= CENTER_V_WASTEFUL) {
                #pragma unroll
                for (int d = 0; d < 16; ++d) v[d] = voucher_sig_value(center, d);
            } else if (center >= CENTER_P_ARCANA_JUMBO_1 && center <= CENTER_P_STANDARD_NORMAL_4) {
                #pragma unroll
                for (int d = 0; d < 16; ++d) v[d] = booster_sig_value(center, d);
            } else {
                #pragma unroll
                for (int d = 0; d < 16; ++d) v[d] = sig_value(&JOKER_SIGNATURES[center], d);
            }
        } else {
            #pragma unroll
            for (int d = 0; d < 16; ++d) v[d] = 0.0f;
        }
        v[16] = (float)enh / 8.0f;
        v[17] = (float)ed / 4.0f;
        v[18] = (float)seal / 4.0f;
        v[19] = (float)flags / 255.0f;
        v[20] = (float)sell_cost / 16.0f;
        // Non-ordinal 2D circular identity code on unit circle for center_id:
        float theta = (center >= 0 && center < CENTER_COUNT)
            ? (2.0f * 3.14159265f * (float)center / (float)CENTER_COUNT) : 0.0f;
        v[21] = (center >= 0 && center < CENTER_COUNT) ? __sinf(theta) : 0.0f;
        v[22] = (float)zone / 6.0f;
        v[23] = (center >= 0 && center < CENTER_COUNT) ? __cosf(theta) : 0.0f;
        v[24] = asinhf((float)perma_bonus / 10.0f) / 3.0f;
        v[25] = asinhf((float)state0 / 100.0f) / 3.0f;
        v[26] = asinhf((float)state1 / 10.0f) / 3.0f;
        v[27] = (float)state0 / 1000.0f;
    }
}

__global__ void __launch_bounds__(256, 4) ba_encode_kernel(
        precision_t* __restrict__ pooled,
        precision_t* __restrict__ keys,
        int* __restrict__ counts_out,
        int* __restrict__ pool_argmax,
        const precision_t* __restrict__ token_w,
        const precision_t* __restrict__ token_b,
        const precision_t* __restrict__ center_embed,
        const precision_t* __restrict__ rank_embed,
        const precision_t* __restrict__ suit_embed,
        const precision_t* __restrict__ zone_embed,
        const unsigned char* __restrict__ obs,
        int B, int obs_size) {
    __shared__ float s_tokens[MAX_OBS_TOKENS][BA_KEY_DIM];
    __shared__ float s_raw[BLOCK_SIZE / 32][BA_RAW_DIM];
    __shared__ int s_counts[BA_POOL_SECTIONS];
    __shared__ int s_total;
    __shared__ int s_section[BLOCK_SIZE / 32];
    __shared__ int s_id[BLOCK_SIZE / 32];
    __shared__ int s_playing[BLOCK_SIZE / 32];

    int b = blockIdx.x;
    if (b >= B) return;
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

    // 1. Unpack Fixed Globals & Counters (520 dims)
    for (int feature = threadIdx.x; feature < BA_FIXED; feature += blockDim.x) {
        float value = 0.0f;
        if (feature < BA_OFF_DECK) {
            int local = feature - BA_OFF_GLOBALS;
            if (local < 32) {
                // Fixed orthogonal multi-frequency sinusoidal identity codes:
                if (local < 8) {
                    int d = local;
                    float freq = (float)(d / 2 + 1);
                    float theta = (deck > 0 && deck < CENTER_COUNT)
                        ? (2.0f * 3.14159265f * (float)deck * freq / (float)CENTER_COUNT) : 0.0f;
                    value = (deck > 0 && deck < CENTER_COUNT) ? ((d % 2 == 0) ? __sinf(theta) : __cosf(theta)) : 0.0f;
                } else if (local < 12) {
                    int d = local - 8;
                    float freq = (float)(d / 2 + 1);
                    float theta = (blind > 0 && blind < BLIND_COUNT)
                        ? (2.0f * 3.14159265f * (float)blind * freq / (float)BLIND_COUNT) : 0.0f;
                    value = (blind > 0 && blind < BLIND_COUNT) ? ((d % 2 == 0) ? __sinf(theta) : __cosf(theta)) : 0.0f;
                } else if (local < 16) {
                    int d = local - 12;
                    float freq = (float)(d / 2 + 1);
                    float theta = (boss > 0 && boss < BLIND_COUNT)
                        ? (2.0f * 3.14159265f * (float)boss * freq / (float)BLIND_COUNT) : 0.0f;
                    value = (boss > 0 && boss < BLIND_COUNT) ? ((d % 2 == 0) ? __sinf(theta) : __cosf(theta)) : 0.0f;
                } else if (local < 24) {
                    int d = local - 16;
                    float freq = (float)(d / 2 + 1);
                    float theta = (voucher > 0 && voucher < CENTER_COUNT)
                        ? (2.0f * 3.14159265f * (float)voucher * freq / (float)CENTER_COUNT) : 0.0f;
                    value = (voucher > 0 && voucher < CENTER_COUNT) ? ((d % 2 == 0) ? __sinf(theta) : __cosf(theta)) : 0.0f;
                } else {
                    int d = local - 24;
                    float freq = (float)(d / 2 + 1);
                    float theta = (tarot > 0 && tarot < CENTER_COUNT)
                        ? (2.0f * 3.14159265f * (float)tarot * freq / (float)CENTER_COUNT) : 0.0f;
                    value = (tarot > 0 && tarot < CENTER_COUNT) ? ((d % 2 == 0) ? __sinf(theta) : __cosf(theta)) : 0.0f;
                }
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
                } else if ((local -= 3) < 4) {
                    value = ba_float(obs, in, base + offsetof(ObservationGlobals, chips_log2) + 4 * local) / 16.0f;
                } else if ((local -= 4) < 3) {
                    value = (float)ba_u16(obs, in, base + offsetof(ObservationGlobals, interest_cap) + 2 * local) / 64.0f;
                } else {
                    local -= 3;
                    value = (float)ba_byte(obs, in, base + offsetof(ObservationGlobals, joker_rate) + local) / 100.0f;
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
            if (f == 0) value = (float)ba_byte(obs, in, record + offsetof(PokerHandStat, visible));
            else if (f == 1) value = (float)ba_byte(obs, in, record + offsetof(PokerHandStat, level)) / 16.0f;
            else if (f == 2) value = ba_float(obs, in, record + offsetof(PokerHandStat, chips_log2)) / 16.0f;
            else if (f == 3) value = ba_float(obs, in, record + offsetof(PokerHandStat, mult_log2)) / 16.0f;
            else if (f == 4) value = (float)ba_u16(obs, in, record + offsetof(PokerHandStat, total_plays)) / 256.0f;
            else value = (float)ba_u16(obs, in, record + offsetof(PokerHandStat, round_plays)) / 64.0f;
        } else if (feature < BA_OFF_BLIND_SIG) {
            int local = feature - BA_OFF_TAGS;
            if (local == 0) {
                value = (float)ba_byte(obs, in, base + offsetof(ObservationGlobals, double_tag));
            } else {
                int blind_idx = (local - 1) / 6;
                int f = (local - 1) % 6;
                int tag_id = ba_byte(obs, in, base + offsetof(ObservationGlobals, blind_tags) + blind_idx);
                if (tag_id == 0 || tag_id >= TAG_COUNT) {
                    value = 0.0f;
                } else if (f < 4) {
                    float freq = (float)(f / 2 + 1);
                    float theta = 2.0f * 3.14159265f * (float)tag_id * freq / (float)TAG_COUNT;
                    value = (f % 2 == 0) ? __sinf(theta) : __cosf(theta);
                } else if (f == 4) {
                    int orb = ba_byte(obs, in, base + offsetof(ObservationGlobals, orbital_hands) + blind_idx);
                    value = (float)orb / (float)HAND_COUNT;
                } else {
                    value = 1.0f;
                }
            }
        } else {
            int local = feature - BA_OFF_BLIND_SIG;
            if (local < 16) {
                value = (blind >= 0 && blind < 31) ? blind_sig_value(&BLIND_SIGNATURES[blind], local) : 0.0f;
            } else {
                int d = local - 16;
                value = (boss >= 0 && boss < 31) ? blind_sig_value(&BLIND_SIGNATURES[boss], d) : 0.0f;
            }
        }
        pooled[b * BA_TOTAL + feature] = from_float(value);
    }

    // 2. Decode live tokens and apply the shared typed token MLP.
    if (threadIdx.x == 0) {
        ba_live_counts(obs, in, s_counts);
        s_total = ba_byte(obs, in, offsetof(Observation, total_tokens));
        if (s_total > MAX_OBS_TOKENS) s_total = MAX_OBS_TOKENS;
    }
    __syncthreads();
    int total = s_total;
    if (threadIdx.x < BA_POOL_SECTIONS)
        counts_out[b * BA_POOL_SECTIONS + threadIdx.x] = s_counts[threadIdx.x];

    constexpr int TOKEN_WARP = 32;
    constexpr int TOKEN_WARPS = BLOCK_SIZE / TOKEN_WARP;
    int lane = threadIdx.x & (TOKEN_WARP - 1);
    int warp = threadIdx.x / TOKEN_WARP;
    for (int slot = warp; slot < total; slot += TOKEN_WARPS) {
        if (lane == 0) {
            int local;
            s_section[warp] = ba_slot_section(s_counts, slot, &local);
            ba_token_features(obs, in, slot, s_raw[warp]);
            int token_base = offsetof(Observation, tokens)
                + slot * sizeof(CardToken);
            int id = ba_u16(obs, in, token_base + offsetof(CardToken, id));
            int zone = ba_byte(obs, in, token_base + offsetof(CardToken, zone));
            s_id[warp] = id;
            s_playing[warp] = zone == ZONE_HAND || id > 300;
        }
        __syncwarp();
        int id = s_id[warp];
        int playing = s_playing[warp];
        int rank = playing ? id >> 8 : 0;
        int suit = playing ? id & 0xFF : 0;
        float z = to_float(token_b[lane])
            + to_float(zone_embed[s_section[warp] * BA_KEY_DIM + lane]);
        if (playing) {
            z += to_float(rank_embed[rank * BA_KEY_DIM + lane]);
            z += to_float(suit_embed[suit * BA_KEY_DIM + lane]);
        } else {
            z += to_float(center_embed[id * BA_KEY_DIM + lane]);
        }
        #pragma unroll
        for (int k = 0; k < BA_RAW_DIM; ++k) {
            z += to_float(token_w[lane * BA_RAW_DIM + k]) * s_raw[warp][k];
        }
        float token_value = fmaxf(z, 0.0f);
        s_tokens[slot][lane] = token_value;
        keys[((int64_t)b * MAX_OBS_TOKENS + slot) * BA_KEY_DIM + lane]
            = from_float(token_value);
        __syncwarp();
    }
    for (int i = total * BA_KEY_DIM + threadIdx.x; i < MAX_OBS_TOKENS * BA_KEY_DIM; i += blockDim.x) {
        int slot = i / BA_KEY_DIM;
        int d = i % BA_KEY_DIM;
        s_tokens[slot][d] = 0.0f;
        keys[((int64_t)b * MAX_OBS_TOKENS + slot) * BA_KEY_DIM + d] = from_float(0.0f);
    }
    __syncthreads();

    // 3. Compute Permutation-Invariant Set Pools (Mean + Max across 7 zones)
    for (int c = threadIdx.x; c < BA_POOLED_FEATURES; c += blockDim.x) {
        int sec = c / (BA_KEY_DIM * 2);
        int rem = c % (BA_KEY_DIM * 2);
        int is_max = rem / BA_KEY_DIM;
        int d = rem % BA_KEY_DIM;

        int start = 0;
        for (int s = 0; s < sec; ++s) start += s_counts[s];
        int end = start + s_counts[sec];
        if (end > total) end = total;

        float sum = 0.0f;
        int cnt = end - start;
        float max_val = cnt > 0 ? s_tokens[start][d] : 0.0f;
        int max_slot = start;
        for (int slot = start; slot < end; ++slot) {
            float v = s_tokens[slot][d];
            sum += v;
            if (v > max_val) {
                max_val = v;
                max_slot = slot;
            }
        }
        float out_v = cnt > 0 ? (is_max ? max_val : sum / (float)cnt) : 0.0f;
        pooled[b * BA_TOTAL + BA_OFF_POOLED + c] = from_float(out_v);
        if (is_max)
            pool_argmax[b * BA_POOL_SECTIONS * BA_KEY_DIM + sec * BA_KEY_DIM + d] =
                cnt > 0 ? max_slot : -1;
    }
}

static constexpr float BA_FXP_SCALE = 16777216.0f;

__device__ __forceinline__ void ba_fxp_atomic_add(long* addr, float value) {
    atomicAdd((unsigned long long*)addr,
        (unsigned long long)(long long)__float2ll_rn(value * BA_FXP_SCALE));
}

__global__ void ba_fxp_to_precision_kernel(
        precision_t* __restrict__ dst, const long* __restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = from_float((float)((double)src[i] * (1.0 / 16777216.0)));
}

__global__ void ba_token_backward_kernel(
        const precision_t* __restrict__ d_pooled,
        const precision_t* __restrict__ keygrad,
        const int* __restrict__ counts_data,
        const int* __restrict__ pool_argmax,
        const precision_t* __restrict__ token_w,
        const precision_t* __restrict__ token_b,
        const precision_t* __restrict__ center_embed,
        const precision_t* __restrict__ rank_embed,
        const precision_t* __restrict__ suit_embed,
        const precision_t* __restrict__ zone_embed,
        long* __restrict__ center_acc,
        long* __restrict__ rank_acc,
        long* __restrict__ suit_acc,
        long* __restrict__ zone_acc,
        long* __restrict__ token_w_acc,
        long* __restrict__ token_b_acc,
        const unsigned char* __restrict__ obs,
        int obs_size,
        int B) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B * MAX_OBS_TOKENS * BA_KEY_DIM) return;
    int d = idx % BA_KEY_DIM;
    int slot = (idx / BA_KEY_DIM) % MAX_OBS_TOKENS;
    int b = idx / ((int64_t)MAX_OBS_TOKENS * BA_KEY_DIM);
    int counts[BA_POOL_SECTIONS];
    #pragma unroll
    for (int s = 0; s < BA_POOL_SECTIONS; ++s)
        counts[s] = counts_data[b * BA_POOL_SECTIONS + s];
    int total = 0;
    #pragma unroll
    for (int s = 0; s < BA_POOL_SECTIONS; ++s) total += counts[s];
    if (slot >= total) return;

    int local;
    int sec = ba_slot_section(counts, slot, &local);
    float g = keygrad ? to_float(keygrad[idx]) : 0.0f;
    int count = counts[sec];
    g += to_float(d_pooled[(int64_t)b * BA_TOTAL + BA_OFF_POOLED
        + sec * BA_KEY_DIM * 2 + d]) / (float)count;
    int max_slot = pool_argmax[(int64_t)b * BA_POOL_SECTIONS * BA_KEY_DIM
        + sec * BA_KEY_DIM + d];
    if (max_slot == slot)
        g += to_float(d_pooled[(int64_t)b * BA_TOTAL + BA_OFF_POOLED
            + sec * BA_KEY_DIM * 2 + BA_KEY_DIM + d]);

    int64_t in = (int64_t)b * obs_size;
    float v[BA_RAW_DIM];
    ba_token_features(obs, in, slot, v);

    int token_base = offsetof(Observation, tokens) + slot * sizeof(CardToken);
    int id = ba_u16(obs, in, token_base + offsetof(CardToken, id));
    int zone = ba_byte(obs, in, token_base + offsetof(CardToken, zone));
    int is_playing = (zone == ZONE_HAND) || (id > 300);
    int rank = is_playing ? id >> 8 : 0;
    int suit = is_playing ? id & 0xFF : 0;
    float pre = to_float(token_b[d]) + to_float(zone_embed[zone * BA_KEY_DIM + d]);
    if (is_playing) {
        pre += to_float(rank_embed[rank * BA_KEY_DIM + d]);
        pre += to_float(suit_embed[suit * BA_KEY_DIM + d]);
    } else {
        pre += to_float(center_embed[id * BA_KEY_DIM + d]);
    }
    #pragma unroll
    for (int k = 0; k < BA_RAW_DIM; ++k)
        pre += to_float(token_w[d * BA_RAW_DIM + k]) * v[k];

    if (pre <= 0.0f || g == 0.0f) return;

    ba_fxp_atomic_add(&zone_acc[zone * BA_KEY_DIM + d], g);
    if (is_playing) {
        ba_fxp_atomic_add(&rank_acc[rank * BA_KEY_DIM + d], g);
        ba_fxp_atomic_add(&suit_acc[suit * BA_KEY_DIM + d], g);
    } else {
        ba_fxp_atomic_add(&center_acc[id * BA_KEY_DIM + d], g);
    }
    ba_fxp_atomic_add(&token_b_acc[d], g);
    #pragma unroll
    for (int k = 0; k < BA_RAW_DIM; ++k) {
        float vk = v[k];
        if (vk != 0.0f)
            ba_fxp_atomic_add(&token_w_acc[d * BA_RAW_DIM + k], g * vk);
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
        a->pooled.data, a->keys.data, a->counts.data,
        a->pool_argmax.data, ew->token_w.data, ew->token_b.data,
        ew->center_embed.data, ew->rank_embed.data, ew->suit_embed.data,
        ew->zone_embed.data, input.data, B, ew->obs_size);
    puf_mm(&a->pooled, &ew->proj_w, &a->out, stream);
    return a->out;
}

static void ba_encoder_backward(
        void* w, void* activations, Prec grad, cudaStream_t stream) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    BalatroEncoderActivations* a = (BalatroEncoderActivations*)activations;
    int B = a->obs_batch;
    puf_mm_tn(&grad, &a->pooled, &a->proj_wgrad, stream);
    puf_mm_nn(&grad, &ew->proj_w, &a->d_pooled, stream);
    cudaMemsetAsync(a->center_acc.data, 0,
        numel(a->center_acc.shape) * sizeof(long), stream);
    cudaMemsetAsync(a->rank_acc.data, 0,
        numel(a->rank_acc.shape) * sizeof(long), stream);
    cudaMemsetAsync(a->suit_acc.data, 0,
        numel(a->suit_acc.shape) * sizeof(long), stream);
    cudaMemsetAsync(a->zone_acc.data, 0,
        numel(a->zone_acc.shape) * sizeof(long), stream);
    cudaMemsetAsync(a->token_w_acc.data, 0,
        numel(a->token_w_acc.shape) * sizeof(long), stream);
    cudaMemsetAsync(a->token_b_acc.data, 0,
        numel(a->token_b_acc.shape) * sizeof(long), stream);
    ba_token_backward_kernel<<<grid_size((int64_t)B * MAX_OBS_TOKENS * BA_KEY_DIM), BLOCK_SIZE, 0, stream>>>(
        a->d_pooled.data,
        ba_ptr_keygrad ? ba_ptr_keygrad->data : NULL,
        a->counts.data, a->pool_argmax.data,
        ew->token_w.data, ew->token_b.data, ew->center_embed.data,
        ew->rank_embed.data, ew->suit_embed.data, ew->zone_embed.data,
        a->center_acc.data, a->rank_acc.data, a->suit_acc.data, a->zone_acc.data,
        a->token_w_acc.data, a->token_b_acc.data,
        a->obs_data, ew->obs_size, B);
    ba_fxp_to_precision_kernel<<<grid_size(BA_KEY_DIM * BA_RAW_DIM), BLOCK_SIZE, 0, stream>>>(
        a->token_wgrad.data, a->token_w_acc.data, BA_KEY_DIM * BA_RAW_DIM);
    ba_fxp_to_precision_kernel<<<grid_size(BA_KEY_DIM), BLOCK_SIZE, 0, stream>>>(
        a->token_bgrad.data, a->token_b_acc.data, BA_KEY_DIM);
    ba_fxp_to_precision_kernel<<<grid_size(CENTER_COUNT * BA_KEY_DIM), BLOCK_SIZE, 0, stream>>>(
        a->center_grad.data, a->center_acc.data, CENTER_COUNT * BA_KEY_DIM);
    ba_fxp_to_precision_kernel<<<grid_size(15 * BA_KEY_DIM), BLOCK_SIZE, 0, stream>>>(
        a->rank_grad.data, a->rank_acc.data, 15 * BA_KEY_DIM);
    ba_fxp_to_precision_kernel<<<grid_size(4 * BA_KEY_DIM), BLOCK_SIZE, 0, stream>>>(
        a->suit_grad.data, a->suit_acc.data, 4 * BA_KEY_DIM);
    ba_fxp_to_precision_kernel<<<grid_size(BA_ZONES * BA_KEY_DIM), BLOCK_SIZE, 0, stream>>>(
        a->zone_grad.data, a->zone_acc.data, BA_ZONES * BA_KEY_DIM);
}

static void ba_encoder_init_weights(
        void* w, uint64_t* seed, cudaStream_t stream) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    puf_kaiming_init(&ew->token_w, 1.0f, (*seed)++, stream);
    puf_normal_init(&ew->token_b, 0.01f, (*seed)++, stream);
    puf_normal_init(&ew->center_embed, 0.02f, (*seed)++, stream);
    puf_normal_init(&ew->rank_embed, 0.02f, (*seed)++, stream);
    puf_normal_init(&ew->suit_embed, 0.02f, (*seed)++, stream);
    puf_normal_init(&ew->zone_embed, 0.02f, (*seed)++, stream);
    puf_kaiming_init(&ew->proj_w, sqrtf(2.0f), (*seed)++, stream);
}

static void ba_encoder_reg_params(void* w, Allocator* alloc) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    ew->token_w = {.shape = {BA_KEY_DIM, BA_RAW_DIM}};
    ew->token_b = {.shape = {BA_KEY_DIM}};
    ew->center_embed = {.shape = {CENTER_COUNT, BA_KEY_DIM}};
    ew->rank_embed = {.shape = {15, BA_KEY_DIM}};
    ew->suit_embed = {.shape = {4, BA_KEY_DIM}};
    ew->zone_embed = {.shape = {BA_ZONES, BA_KEY_DIM}};
    ew->proj_w = {.shape = {ew->hidden, BA_TOTAL}};
    alloc_register(alloc, &ew->token_w);
    alloc_register(alloc, &ew->token_b);
    alloc_register(alloc, &ew->center_embed);
    alloc_register(alloc, &ew->rank_embed);
    alloc_register(alloc, &ew->suit_embed);
    alloc_register(alloc, &ew->zone_embed);
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
    a->keys = {.shape = {B_TT, MAX_OBS_TOKENS, BA_KEY_DIM}};
    a->counts = {.shape = {B_TT, BA_POOL_SECTIONS}};
    a->pool_argmax = {.shape = {B_TT, BA_POOL_SECTIONS, BA_KEY_DIM}};
    a->center_acc = {.shape = {CENTER_COUNT, BA_KEY_DIM}};
    a->rank_acc = {.shape = {15, BA_KEY_DIM}};
    a->suit_acc = {.shape = {4, BA_KEY_DIM}};
    a->zone_acc = {.shape = {BA_ZONES, BA_KEY_DIM}};
    a->token_w_acc = {.shape = {BA_KEY_DIM, BA_RAW_DIM}};
    a->token_b_acc = {.shape = {BA_KEY_DIM}};
    a->token_wgrad = {.shape = {BA_KEY_DIM, BA_RAW_DIM}};
    a->token_bgrad = {.shape = {BA_KEY_DIM}};
    a->center_grad = {.shape = {CENTER_COUNT, BA_KEY_DIM}};
    a->rank_grad = {.shape = {15, BA_KEY_DIM}};
    a->suit_grad = {.shape = {4, BA_KEY_DIM}};
    a->zone_grad = {.shape = {BA_ZONES, BA_KEY_DIM}};
    a->proj_wgrad = {.shape = {ew->hidden, BA_TOTAL}};
    alloc_register(acts, &a->pooled);
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->d_pooled);
    alloc_register(acts, &a->keys);
    alloc_register(acts, &a->counts);
    alloc_register(acts, &a->pool_argmax);
    alloc_register(acts, &a->center_acc);
    alloc_register(acts, &a->rank_acc);
    alloc_register(acts, &a->suit_acc);
    alloc_register(acts, &a->zone_acc);
    alloc_register(acts, &a->token_w_acc);
    alloc_register(acts, &a->token_b_acc);
    alloc_register(grads, &a->token_wgrad);
    alloc_register(grads, &a->token_bgrad);
    alloc_register(grads, &a->center_grad);
    alloc_register(grads, &a->rank_grad);
    alloc_register(grads, &a->suit_grad);
    alloc_register(grads, &a->zone_grad);
    alloc_register(grads, &a->proj_wgrad);
    ba_enc_last = a;
}

static void ba_encoder_reg_rollout(
        void* w, void* activations, Allocator* alloc, int B) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    BalatroEncoderActivations* a = (BalatroEncoderActivations*)activations;
    *a = {};
    a->pooled = {.shape = {B, BA_TOTAL}};
    a->out = {.shape = {B, ew->hidden}};
    a->keys = {.shape = {B, MAX_OBS_TOKENS, BA_KEY_DIM}};
    a->counts = {.shape = {B, BA_POOL_SECTIONS}};
    a->pool_argmax = {.shape = {B, BA_POOL_SECTIONS, BA_KEY_DIM}};
    alloc_register(alloc, &a->pooled);
    alloc_register(alloc, &a->out);
    alloc_register(alloc, &a->keys);
    alloc_register(alloc, &a->counts);
    alloc_register(alloc, &a->pool_argmax);
    ba_enc_last = a;
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

static constexpr int BA_PRIMARY_TYPES[POLICY_PRIMARY_HEADS] = {
    ACTION_BUY_CARD, ACTION_SELL_JOKER, ACTION_SELL_CONSUMABLE,
    ACTION_USE_CONSUMABLE, ACTION_REDEEM_VOUCHER, ACTION_OPEN_BOOSTER,
    ACTION_PICK_PACK_CARD, ACTION_SWAP_JOKERS_LEFT, ACTION_SWAP_JOKERS_RIGHT,
    ACTION_SWAP_HAND_LEFT, ACTION_SWAP_HAND_RIGHT, ACTION_BUY_AND_USE};

static constexpr int BA_PRIMARY_ZONES[POLICY_PRIMARY_HEADS] = {
    ZONE_SHOP_MAIN, ZONE_JOKER, ZONE_CONSUMABLE, ZONE_CONSUMABLE,
    ZONE_SHOP_VOUCHER, ZONE_SHOP_BOOSTER, ZONE_PACK_CARD, ZONE_JOKER,
    ZONE_JOKER, ZONE_HAND, ZONE_HAND, ZONE_SHOP_MAIN};

__device__ __forceinline__ int ba_key_index(
        const int* counts, int query, int option) {
    if (query < BA_PRIMARY_QUERY_COUNT) {
        int zone = BA_PRIMARY_ZONES[query];
        if (option >= counts[zone]) return -1;
        int start = 0;
        for (int s = 0; s < zone; ++s) start += counts[s];
        return start + option;
    }
    if (option >= counts[ZONE_HAND]) return -1;
    return option;
}

__global__ void ba_primary_probability_kernel(
        precision_t* __restrict__ probability,
        precision_t* __restrict__ lse_out,
        precision_t* __restrict__ out,
        const precision_t* __restrict__ fused,
        const precision_t* __restrict__ keys,
        const int* __restrict__ counts_data, int B) {
    constexpr int WARP = 32;
    constexpr int WARPS = BLOCK_SIZE / WARP;
    int lane = threadIdx.x & (WARP - 1);
    int warp = threadIdx.x / WARP;
    int idx = blockIdx.x * WARPS + warp;
    bool active = idx < B * BA_PRIMARY_QUERY_COUNT;
    int b = active ? idx / BA_PRIMARY_QUERY_COUNT : 0;
    int query_id = active ? idx % BA_PRIMARY_QUERY_COUNT : 0;
    int counts[BA_POOL_SECTIONS];
    for (int s = 0; s < BA_POOL_SECTIONS; ++s) {
        counts[s] = active ? counts_data[b * BA_POOL_SECTIONS + s] : 0;
    }
    int candidates = active ? counts[BA_PRIMARY_ZONES[query_id]] : 0;
    precision_t* p = probability +
        ((int64_t)b * BA_PRIMARY_QUERY_COUNT + query_id) * POLICY_PRIMARY_COUNT;
    precision_t* output = out + (int64_t)b * BA_DEC_ROWS
        + BA_PRIMARY_OFFSET + query_id * POLICY_PRIMARY_COUNT;
    const precision_t* q = fused + (int64_t)b * BA_FUSED_ROWS
        + BA_LINEAR_PAD + query_id * BA_KEY_DIM;
    float score[2] = {-INFINITY, -INFINITY};
    #pragma unroll
    for (int part = 0; part < 2; ++part) {
        int option = lane + part * WARP;
        if (active && option < candidates) {
            int key = ba_key_index(counts, query_id, option);
            const precision_t* k = keys
                + ((int64_t)b * MAX_OBS_TOKENS + key) * BA_KEY_DIM;
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < BA_KEY_DIM; ++d) {
                dot += to_float(q[d]) * to_float(k[d]);
            }
            score[part] = dot * BA_QUERY_SCALE;
        }
    }
    float max_score = fmaxf(score[0], score[1]);
    #pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        max_score = fmaxf(max_score,
            __shfl_down_sync(PUF_WARP_MASK, max_score, offset, WARP));
    }
    max_score = __shfl_sync(PUF_WARP_MASK, max_score, 0, WARP);
    float sum = candidates > 0
        ? __expf(score[0] - max_score) + __expf(score[1] - max_score)
        : 0.0f;
    #pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(PUF_WARP_MASK, sum, offset, WARP);
    }
    sum = __shfl_sync(PUF_WARP_MASK, sum, 0, WARP);
    float lse = candidates > 0 ? max_score + __logf(sum) : -INFINITY;
    if (active && lane == 0) lse_out[idx] = from_float(lse);
    #pragma unroll
    for (int part = 0; part < 2; ++part) {
        int option = lane + part * WARP;
        if (active) {
            bool valid = option < candidates;
            output[option] = from_float(valid ? score[part] : 0.0f);
            p[option] = from_float(valid ? __expf(score[part] - lse) : 0.0f);
        }
    }
}

__global__ void ba_decoder_type_lse_kernel(
        precision_t* __restrict__ out,
        const precision_t* __restrict__ lse,
        const int* __restrict__ counts_data, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * BA_PRIMARY_QUERY_COUNT) return;
    int b = idx / BA_PRIMARY_QUERY_COUNT;
    int query = idx % BA_PRIMARY_QUERY_COUNT;
    int zone = BA_PRIMARY_ZONES[query];
    if (counts_data[b * BA_POOL_SECTIONS + zone] == 0) return;
    int col = BA_PRIMARY_TYPES[query];
    out[(int64_t)b * BA_DEC_ROWS + col] = from_float(
        to_float(out[(int64_t)b * BA_DEC_ROWS + col]) + to_float(lse[idx]));
}

__global__ void ba_decoder_assemble_kernel(
        precision_t* __restrict__ out,
        const precision_t* __restrict__ fused,
        const precision_t* __restrict__ keys,
        const int* __restrict__ counts_data,
        int B) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B * BA_DEC_ROWS) return;
    int b = idx / BA_DEC_ROWS;
    int c = idx % BA_DEC_ROWS;
    const precision_t* b_fused = fused + (int64_t)b * BA_FUSED_ROWS;
    if (c < 23) {
        out[idx] = b_fused[c];
        return;
    }
    if ((c >= BA_COUNT_OFFSET && c < BA_CARD_OFFSET) || c == BA_VALUE_OFFSET) {
        int linear_col = c < BA_CARD_OFFSET ? 23 + c - BA_COUNT_OFFSET : 29;
        out[idx] = b_fused[linear_col];
        return;
    }
    if (c >= BA_PRIMARY_OFFSET && c < BA_COUNT_OFFSET) return;
    int query_id;
    int option;
    int p = c - BA_CARD_OFFSET;
    query_id = BA_PRIMARY_QUERY_COUNT + p / POLICY_PRIMARY_COUNT;
    option = p % POLICY_PRIMARY_COUNT;
    int counts[BA_POOL_SECTIONS];
    for (int s = 0; s < BA_POOL_SECTIONS; ++s)
        counts[s] = counts_data[b * BA_POOL_SECTIONS + s];
    int key = ba_key_index(counts, query_id, option);
    if (key < 0) {
        out[idx] = from_float(0.0f);
        return;
    }
    const precision_t* q = b_fused + BA_LINEAR_PAD + query_id * BA_KEY_DIM;
    const precision_t* k = keys + ((int64_t)b * MAX_OBS_TOKENS + key) * BA_KEY_DIM;
    float dot = 0.0f;
    for (int d = 0; d < BA_KEY_DIM; ++d)
        dot += to_float(q[d]) * to_float(k[d]);
    out[idx] = from_float(dot * BA_QUERY_SCALE);
}

__global__ void ba_decoder_prepare_grad_kernel(
        precision_t* __restrict__ dall,
        const precision_t* __restrict__ grad_out, int B) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B * BA_LINEAR_PAD) return;
    int b = idx / BA_LINEAR_PAD;
    int c = idx % BA_LINEAR_PAD;
    int64_t out_idx = (int64_t)b * BA_FUSED_ROWS + c;
    if (c < 23)
        dall[out_idx] = grad_out[(int64_t)b * BA_DEC_ROWS + c];
    else if (c < 29)
        dall[out_idx] = grad_out[(int64_t)b * BA_DEC_ROWS + BA_COUNT_OFFSET + c - 23];
    else if (c == 29)
        dall[out_idx] = grad_out[(int64_t)b * BA_DEC_ROWS + BA_VALUE_OFFSET];
    else
        dall[out_idx] = from_float(0.0f);
}

__global__ void ba_decoder_query_backward_kernel(
        precision_t* __restrict__ dall,
        const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ keys,
        const precision_t* __restrict__ primary_probability,
        const int* __restrict__ counts_data, int B) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B * BA_QUERY_COUNT * BA_KEY_DIM) return;
    int d = idx % BA_KEY_DIM;
    int query_id = (idx / BA_KEY_DIM) % BA_QUERY_COUNT;
    int b = idx / ((int64_t)BA_QUERY_COUNT * BA_KEY_DIM);
    int counts[BA_POOL_SECTIONS];
    #pragma unroll
    for (int s = 0; s < BA_POOL_SECTIONS; ++s)
        counts[s] = counts_data[b * BA_POOL_SECTIONS + s];

    int zone = (query_id < BA_PRIMARY_QUERY_COUNT) ? BA_PRIMARY_ZONES[query_id] : ZONE_HAND;
    int candidates = counts[zone];
    if (candidates > POLICY_PRIMARY_COUNT) candidates = POLICY_PRIMARY_COUNT;
    int64_t out_idx = (int64_t)b * BA_FUSED_ROWS + BA_LINEAR_PAD + query_id * BA_KEY_DIM + d;
    if (candidates == 0) {
        dall[out_idx] = from_float(0.0f);
        return;
    }
    int start = 0;
    for (int s = 0; s < zone; ++s) start += counts[s];

    float sum = 0.0f;
    int64_t b_dec = (int64_t)b * BA_DEC_ROWS;
    int64_t b_prob = (int64_t)b * BA_PRIMARY_QUERY_COUNT * POLICY_PRIMARY_COUNT;
    int64_t b_keys = (int64_t)b * MAX_OBS_TOKENS * BA_KEY_DIM;
    int out_col_base = (query_id < BA_PRIMARY_QUERY_COUNT)
        ? BA_PRIMARY_OFFSET + query_id * POLICY_PRIMARY_COUNT
        : BA_CARD_OFFSET + (query_id - BA_PRIMARY_QUERY_COUNT) * POLICY_PRIMARY_COUNT;
    float type_grad = (query_id < BA_PRIMARY_QUERY_COUNT)
        ? to_float(grad_out[b_dec + BA_PRIMARY_TYPES[query_id]]) : 0.0f;

    for (int option = 0; option < candidates; ++option) {
        int key = start + option;
        float scalar = to_float(grad_out[b_dec + out_col_base + option]);
        if (query_id < BA_PRIMARY_QUERY_COUNT) {
            scalar += type_grad * to_float(primary_probability[b_prob + query_id * POLICY_PRIMARY_COUNT + option]);
        }
        sum += scalar * to_float(keys[b_keys + key * BA_KEY_DIM + d]);
    }
    dall[out_idx] = from_float(sum * BA_QUERY_SCALE);
}

__global__ void ba_decoder_key_backward_kernel(
        precision_t* __restrict__ keygrad,
        const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ fused,
        const precision_t* __restrict__ primary_probability,
        const int* __restrict__ counts_data, int B) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B * MAX_OBS_TOKENS * BA_KEY_DIM) return;
    int d = idx % BA_KEY_DIM;
    int slot = (idx / BA_KEY_DIM) % MAX_OBS_TOKENS;
    int b = idx / ((int64_t)MAX_OBS_TOKENS * BA_KEY_DIM);
    int counts[BA_POOL_SECTIONS];
    #pragma unroll
    for (int s = 0; s < BA_POOL_SECTIONS; ++s)
        counts[s] = counts_data[b * BA_POOL_SECTIONS + s];
    int local;
    int zone = ba_slot_section(counts, slot, &local);
    if (zone < 0) {
        keygrad[idx] = from_float(0.0f);
        return;
    }
    float sum = 0.0f;
    int64_t b_dec = (int64_t)b * BA_DEC_ROWS;
    int64_t b_prob = (int64_t)b * BA_PRIMARY_QUERY_COUNT * POLICY_PRIMARY_COUNT;
    const precision_t* b_query = fused + (int64_t)b * BA_FUSED_ROWS + BA_LINEAR_PAD;

    if (zone == ZONE_SHOP_MAIN) {
        static constexpr int q_list[2] = {0, 11};
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int q = q_list[i];
            int out_col = BA_PRIMARY_OFFSET + q * POLICY_PRIMARY_COUNT + local;
            float scalar = to_float(grad_out[b_dec + out_col]);
            float type_grad = to_float(grad_out[b_dec + BA_PRIMARY_TYPES[q]]);
            scalar += type_grad * to_float(primary_probability[b_prob + q * POLICY_PRIMARY_COUNT + local]);
            sum += scalar * to_float(b_query[q * BA_KEY_DIM + d]);
        }
    } else if (zone == ZONE_JOKER) {
        static constexpr int q_list[3] = {1, 7, 8};
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            int q = q_list[i];
            int out_col = BA_PRIMARY_OFFSET + q * POLICY_PRIMARY_COUNT + local;
            float scalar = to_float(grad_out[b_dec + out_col]);
            float type_grad = to_float(grad_out[b_dec + BA_PRIMARY_TYPES[q]]);
            scalar += type_grad * to_float(primary_probability[b_prob + q * POLICY_PRIMARY_COUNT + local]);
            sum += scalar * to_float(b_query[q * BA_KEY_DIM + d]);
        }
    } else if (zone == ZONE_CONSUMABLE) {
        static constexpr int q_list[2] = {2, 3};
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int q = q_list[i];
            int out_col = BA_PRIMARY_OFFSET + q * POLICY_PRIMARY_COUNT + local;
            float scalar = to_float(grad_out[b_dec + out_col]);
            float type_grad = to_float(grad_out[b_dec + BA_PRIMARY_TYPES[q]]);
            scalar += type_grad * to_float(primary_probability[b_prob + q * POLICY_PRIMARY_COUNT + local]);
            sum += scalar * to_float(b_query[q * BA_KEY_DIM + d]);
        }
    } else if (zone == ZONE_SHOP_VOUCHER) {
        int q = 4;
        int out_col = BA_PRIMARY_OFFSET + q * POLICY_PRIMARY_COUNT + local;
        float scalar = to_float(grad_out[b_dec + out_col]);
        float type_grad = to_float(grad_out[b_dec + BA_PRIMARY_TYPES[q]]);
        scalar += type_grad * to_float(primary_probability[b_prob + q * POLICY_PRIMARY_COUNT + local]);
        sum += scalar * to_float(b_query[q * BA_KEY_DIM + d]);
    } else if (zone == ZONE_SHOP_BOOSTER) {
        int q = 5;
        int out_col = BA_PRIMARY_OFFSET + q * POLICY_PRIMARY_COUNT + local;
        float scalar = to_float(grad_out[b_dec + out_col]);
        float type_grad = to_float(grad_out[b_dec + BA_PRIMARY_TYPES[q]]);
        scalar += type_grad * to_float(primary_probability[b_prob + q * POLICY_PRIMARY_COUNT + local]);
        sum += scalar * to_float(b_query[q * BA_KEY_DIM + d]);
    } else if (zone == ZONE_PACK_CARD) {
        int q = 6;
        int out_col = BA_PRIMARY_OFFSET + q * POLICY_PRIMARY_COUNT + local;
        float scalar = to_float(grad_out[b_dec + out_col]);
        float type_grad = to_float(grad_out[b_dec + BA_PRIMARY_TYPES[q]]);
        scalar += type_grad * to_float(primary_probability[b_prob + q * POLICY_PRIMARY_COUNT + local]);
        sum += scalar * to_float(b_query[q * BA_KEY_DIM + d]);
    } else if (zone == ZONE_HAND && local < POLICY_PRIMARY_COUNT) {
        static constexpr int q_list[2] = {9, 10};
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int q = q_list[i];
            int out_col = BA_PRIMARY_OFFSET + q * POLICY_PRIMARY_COUNT + local;
            float scalar = to_float(grad_out[b_dec + out_col]);
            float type_grad = to_float(grad_out[b_dec + BA_PRIMARY_TYPES[q]]);
            scalar += type_grad * to_float(primary_probability[b_prob + q * POLICY_PRIMARY_COUNT + local]);
            sum += scalar * to_float(b_query[q * BA_KEY_DIM + d]);
        }
        #pragma unroll
        for (int card_pos = 0; card_pos < 5; ++card_pos) {
            int q = BA_PRIMARY_QUERY_COUNT + card_pos;
            int out_col = BA_CARD_OFFSET + card_pos * POLICY_PRIMARY_COUNT + local;
            float scalar = to_float(grad_out[b_dec + out_col]);
            sum += scalar * to_float(b_query[q * BA_KEY_DIM + d]);
        }
    }
    keygrad[idx] = from_float(sum * BA_QUERY_SCALE);
}

struct BalatroDecoderWeights {
    Prec weight, logstd, condition;
    int hidden_dim, output_dim;
    bool continuous, ar;
};

struct BalatroDecoderActivations {
    /* Keep the framework-visible DecoderActivations prefix byte-identical:
       train_epoch_gpu accesses the condition-gradient staging tensors through
       that type even when the decoder itself is custom. */
    Prec out, grad_out, saved_input, grad_input, wgrad_scratch, logstd_scratch,
        condition_scratch;
    Float condition_accum, cond_accum_parts;
    BalatroEncoderActivations* enc;
    Prec fused, primary_probability, primary_lse;
    Prec dall, keygrad;
    Prec weight_grad;
};

static Prec ba_decoder_forward(
        void* w, void* activations, Prec input, cudaStream_t stream) {
    BalatroDecoderWeights* dw = (BalatroDecoderWeights*)w;
    BalatroDecoderActivations* a = (BalatroDecoderActivations*)activations;
    int B = input.shape[0];
    if (a->saved_input.data) puf_copy(&a->saved_input, &input, stream);
    puf_mm(&input, &dw->weight, &a->fused, stream);
    ba_decoder_assemble_kernel<<<grid_size(B * BA_DEC_ROWS), BLOCK_SIZE, 0, stream>>>(
        a->out.data, a->fused.data, a->enc->keys.data,
        a->enc->counts.data, B);
    constexpr int primary_warps = BLOCK_SIZE / 32;
    int primary_blocks = (B * BA_PRIMARY_QUERY_COUNT + primary_warps - 1)
        / primary_warps;
    ba_primary_probability_kernel<<<primary_blocks, BLOCK_SIZE, 0, stream>>>(
        a->primary_probability.data, a->primary_lse.data, a->out.data,
        a->fused.data, a->enc->keys.data, a->enc->counts.data, B);
    ba_decoder_type_lse_kernel<<<grid_size(B * BA_PRIMARY_QUERY_COUNT), BLOCK_SIZE, 0, stream>>>(
        a->out.data, a->primary_lse.data, a->enc->counts.data, B);
    return a->out;
}

static Prec ba_decoder_backward(void* w, void* activations,
        Float grad_logits, Float grad_logstd, Float grad_value,
        cudaStream_t stream) {
    (void)grad_logstd;
    BalatroDecoderWeights* dw = (BalatroDecoderWeights*)w;
    BalatroDecoderActivations* a = (BalatroDecoderActivations*)activations;
    int B = a->saved_input.shape[0];
    assemble_decoder_grad<<<grid_size(B * BA_DEC_ROWS), BLOCK_SIZE, 0, stream>>>(
        a->grad_out.data, grad_logits.data, grad_value.data,
        B, dw->output_dim, BA_DEC_ROWS);
    ba_decoder_prepare_grad_kernel<<<grid_size(B * BA_LINEAR_PAD), BLOCK_SIZE, 0, stream>>>(
        a->dall.data, a->grad_out.data, B);
    ba_decoder_query_backward_kernel<<<grid_size((int64_t)B * BA_QUERY_DIM), BLOCK_SIZE, 0, stream>>>(
        a->dall.data, a->grad_out.data, a->enc->keys.data,
        a->primary_probability.data,
        a->enc->counts.data, B);
    ba_decoder_key_backward_kernel<<<grid_size((int64_t)B * MAX_OBS_TOKENS * BA_KEY_DIM), BLOCK_SIZE, 0, stream>>>(
        a->keygrad.data, a->grad_out.data, a->fused.data,
        a->primary_probability.data,
        a->enc->counts.data, B);
    puf_mm_tn(&a->dall, &a->saved_input, &a->weight_grad, stream);
    puf_mm_nn(&a->dall, &dw->weight, &a->grad_input, stream);
    return a->grad_input;
}

static void ba_decoder_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    BalatroDecoderWeights* dw = (BalatroDecoderWeights*)w;
    puf_kaiming_init(&dw->weight, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(dw->condition.data, 0,
        numel(dw->condition.shape) * sizeof(precision_t), stream);
    Prec e_prefix = {
        .data = dw->condition.data,
        .shape = {AR_W_PRIMARY_OFFSET / AR_EMBED_DIM, AR_EMBED_DIM},
    };
    puf_kaiming_init(&e_prefix, 1.0f, (*seed)++, stream);
}

static void ba_decoder_reg_params(void* w, Allocator* alloc) {
    BalatroDecoderWeights* dw = (BalatroDecoderWeights*)w;
    dw->weight = {.shape = {BA_FUSED_ROWS, dw->hidden_dim}};
    dw->condition = {.shape = {AR_CONDITION_SIZE}};
    alloc_register(alloc, &dw->weight);
    alloc_register(alloc, &dw->condition);
}

static void ba_decoder_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B_TT) {
    BalatroDecoderWeights* dw = (BalatroDecoderWeights*)w;
    BalatroDecoderActivations* a = (BalatroDecoderActivations*)activations;
    *a = {};
    a->enc = ba_enc_last;
    a->out = {.shape = {B_TT, BA_DEC_ROWS}};
    a->grad_out = {.shape = {B_TT, BA_DEC_ROWS}};
    a->saved_input = {.shape = {B_TT, dw->hidden_dim}};
    a->grad_input = {.shape = {B_TT, dw->hidden_dim}};
    a->condition_scratch = {.shape = {AR_CONDITION_SIZE}};
    a->condition_accum = {.shape = {AR_CONDITION_SIZE}};
    a->cond_accum_parts = {.shape = {COND_STRIPE_COUNT, AR_CONDITION_SIZE}};
    a->fused = {.shape = {B_TT, BA_FUSED_ROWS}};
    a->primary_probability = {.shape = {B_TT, BA_PRIMARY_QUERY_COUNT, POLICY_PRIMARY_COUNT}};
    a->primary_lse = {.shape = {B_TT, BA_PRIMARY_QUERY_COUNT}};
    a->dall = {.shape = {B_TT, BA_FUSED_ROWS}};
    a->keygrad = {.shape = {B_TT, MAX_OBS_TOKENS, BA_KEY_DIM}};
    a->weight_grad = {.shape = {BA_FUSED_ROWS, dw->hidden_dim}};
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->grad_out);
    alloc_register(acts, &a->saved_input);
    alloc_register(acts, &a->grad_input);
    alloc_register(acts, &a->condition_accum);
    alloc_register(acts, &a->cond_accum_parts);
    alloc_register(acts, &a->fused);
    alloc_register(acts, &a->primary_probability);
    alloc_register(acts, &a->primary_lse);
    alloc_register(acts, &a->dall);
    alloc_register(acts, &a->keygrad);
    alloc_register(grads, &a->weight_grad);
    alloc_register(grads, &a->condition_scratch);
    ba_ptr_keygrad = &a->keygrad;
}

static void ba_decoder_reg_rollout(void* w, void* activations,
        Allocator* alloc, int B) {
    BalatroDecoderActivations* a = (BalatroDecoderActivations*)activations;
    a->enc = ba_enc_last;
    a->out = {.shape = {B, BA_DEC_ROWS}};
    a->fused = {.shape = {B, BA_FUSED_ROWS}};
    a->primary_probability = {.shape = {B, BA_PRIMARY_QUERY_COUNT, POLICY_PRIMARY_COUNT}};
    a->primary_lse = {.shape = {B, BA_PRIMARY_QUERY_COUNT}};
    alloc_register(alloc, &a->out);
    alloc_register(alloc, &a->fused);
    alloc_register(alloc, &a->primary_probability);
    alloc_register(alloc, &a->primary_lse);
}

static void* ba_decoder_create_weights(void* self) {
    Decoder* d = (Decoder*)self;
    assert(d->output_dim == BA_DEC_ROWS - 1);
    BalatroDecoderWeights* dw =
        (BalatroDecoderWeights*)calloc(1, sizeof(BalatroDecoderWeights));
    dw->hidden_dim = d->hidden_dim;
    dw->output_dim = d->output_dim;
    dw->continuous = false;
    dw->ar = true;
    return dw;
}

static void create_balatro_decoder(Decoder* dec) {
    *dec = Decoder{
        .forward = ba_decoder_forward,
        .backward = ba_decoder_backward,
        .init_weights = ba_decoder_init_weights,
        .reg_params = ba_decoder_reg_params,
        .reg_train = ba_decoder_reg_train,
        .reg_rollout = ba_decoder_reg_rollout,
        .create_weights = ba_decoder_create_weights,
        .hidden_dim = dec->hidden_dim,
        .output_dim = dec->output_dim,
        .continuous = false,
        .ar = true,
        .activation_size = sizeof(BalatroDecoderActivations),
    };
}
