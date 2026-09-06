// Balatro GPU encoder: typed live-token embeddings with pooled state and
// pointer-decoder keys.  The fixed observation fields retain their compact
// byte decoding, while every live item gets an exact-id embedding and a
// shared nonlinear representation.
//
// Decodes live tokens directly from the packed Observation token stream,
// pools content-position token keys per section (Hand, Jokers, Consumables,
// Shop, Vouchers, Boosters, Pack), keeps every live key for indexed actions,
// decodes fixed globals and DeckMatrix features, and projects the concatenated
// section vectors directly to the policy hidden size.

#include "joker_signatures.h"
#include "consumable_signatures.h"

static constexpr int BA_RAW_DIM = 32;
static constexpr int BA_TOKEN_DIM = 32;
static constexpr int BA_KEY_DIM = AR_EMBED_DIM;
static constexpr int BA_ZONES = 8;
static constexpr int BA_SPECIAL_ZONE = 7;
static constexpr int BA_KEY_CAP = OBS_MAX_HAND + OBS_MAX_JOKERS
    + OBS_MAX_CONSUMABLES + OBS_MAX_SHOP_MAIN + OBS_MAX_SHOP_VOUCHERS
    + OBS_MAX_SHOP_BOOSTERS + OBS_MAX_PACK_CARDS;

// Discrete embedding dimensions
static constexpr int BA_BLIND_EMBED_ROWS = BLIND_COUNT; // 31
static constexpr int BA_BLIND_EMBED_DIM = 8;
static constexpr int BA_DECK_EMBED_ROWS = 16;
static constexpr int BA_DECK_EMBED_DIM = 8;
static constexpr int BA_LOCK_EMBED_ROWS = HAND_COUNT + 1; // 13 (0..11 hand types, 12 = unlocked)
static constexpr int BA_LOCK_EMBED_DIM = 8;
static constexpr int BA_PLAYED_HAND_EMBED_ROWS = HAND_COUNT; // 12
static constexpr int BA_PLAYED_HAND_EMBED_DIM = 8;
static constexpr int BA_PLANET_HAND_EMBED_ROWS = HAND_COUNT; // 12
static constexpr int BA_PLANET_HAND_EMBED_DIM = 8;
static constexpr int BA_SKIP_EMBED_ROWS = 3; // Small, Big, Boss
static constexpr int BA_SKIP_EMBED_DIM = 4;
static constexpr int BA_TAG_EMBED_ROWS = TAG_COUNT; // 25
static constexpr int BA_TAG_EMBED_DIM = 4;
static constexpr int BA_VOUCHER_ID_EMBED_ROWS = 34;
static constexpr int BA_VOUCHER_ID_EMBED_DIM = 8;
static constexpr int BA_TAROT_EMBED_ROWS = CENTER_COUNT; // 300
static constexpr int BA_TAROT_EMBED_DIM = 8;

static constexpr int BA_CARD_FLAG_ROWS = 7;
static constexpr int BA_ENH_EMBED_ROWS = 9;
static constexpr int BA_ED_EMBED_ROWS = 5;
static constexpr int BA_SEAL_EMBED_ROWS = 5;

// Pack categorical parameters by embedding width. NorMuon operates once per
// registered matrix, so registering every logical table separately turns a
// handful of lookups into dozens of optimizer kernels per minibatch.
static constexpr int BA_TOKEN_CENTER_ROW = 0;
static constexpr int BA_TOKEN_RANK_ROW = BA_TOKEN_CENTER_ROW + CENTER_COUNT;
static constexpr int BA_TOKEN_SUIT_ROW = BA_TOKEN_RANK_ROW + 15;
static constexpr int BA_TOKEN_ZONE_ROW = BA_TOKEN_SUIT_ROW + 4;
static constexpr int BA_TOKEN_FLAG_ROW = BA_TOKEN_ZONE_ROW + BA_ZONES;
static constexpr int BA_TOKEN_ENH_ROW = BA_TOKEN_FLAG_ROW + BA_CARD_FLAG_ROWS;
static constexpr int BA_TOKEN_ED_ROW = BA_TOKEN_ENH_ROW + BA_ENH_EMBED_ROWS;
static constexpr int BA_TOKEN_SEAL_ROW = BA_TOKEN_ED_ROW + BA_ED_EMBED_ROWS;
static constexpr int BA_TOKEN_HAND_POS_ROW = BA_TOKEN_SEAL_ROW + BA_SEAL_EMBED_ROWS;
static constexpr int BA_TOKEN_HAND_POS_ROWS = OBS_MAX_HAND;
static constexpr int BA_TOKEN_JOKER_POS_ROW = BA_TOKEN_HAND_POS_ROW + BA_TOKEN_HAND_POS_ROWS;
static constexpr int BA_TOKEN_JOKER_POS_ROWS = OBS_MAX_JOKERS;
static constexpr int BA_TOKEN_LOCATION_ROW = BA_TOKEN_JOKER_POS_ROW + BA_TOKEN_JOKER_POS_ROWS;
static constexpr int BA_TOKEN_LOCATION_ROWS = 2;
static constexpr int BA_TOKEN_EMBED_ROWS = BA_TOKEN_LOCATION_ROW + BA_TOKEN_LOCATION_ROWS;

static constexpr int BA_STATE_BLIND_ROW = 0;
static constexpr int BA_STATE_DECK_ROW = BA_STATE_BLIND_ROW + BA_BLIND_EMBED_ROWS;
static constexpr int BA_STATE_LOCK_ROW = BA_STATE_DECK_ROW + BA_DECK_EMBED_ROWS;
static constexpr int BA_STATE_PLAYED_HAND_ROW = BA_STATE_LOCK_ROW + BA_LOCK_EMBED_ROWS;
static constexpr int BA_STATE_PLANET_HAND_ROW = BA_STATE_PLAYED_HAND_ROW + BA_PLAYED_HAND_EMBED_ROWS;
static constexpr int BA_STATE_SKIP_ROW = BA_STATE_PLANET_HAND_ROW + BA_PLANET_HAND_EMBED_ROWS;
static constexpr int BA_STATE_TAG_ROW = BA_STATE_SKIP_ROW + BA_SKIP_EMBED_ROWS;
static constexpr int BA_STATE_VOUCHER_ROW = BA_STATE_TAG_ROW + BA_TAG_EMBED_ROWS;
static constexpr int BA_STATE_TAROT_ROW = BA_STATE_VOUCHER_ROW + BA_VOUCHER_ID_EMBED_ROWS;
static constexpr int BA_STATE_RANK_ROW = BA_STATE_TAROT_ROW + BA_TAROT_EMBED_ROWS;
static constexpr int BA_STATE_SUIT_ROW = BA_STATE_RANK_ROW + 15;
static constexpr int BA_STATE_EMBED_ROWS = BA_STATE_SUIT_ROW + 4;
static constexpr int BA_STATE_EMBED_DIM = 8;

// Fixed sections layout
static constexpr int BA_OFF_GLOBALS = 0;
static constexpr int BA_OFF_DECK_EMBED = 0;
static constexpr int BA_OFF_BLIND_EMBED = 8;
static constexpr int BA_OFF_BOSS_EMBED = 16;
static constexpr int BA_OFF_VOUCHER_EMBED = 24;
static constexpr int BA_OFF_TAROT_EMBED = 32;
static constexpr int BA_OFF_PENDING_EMBED = 40;
static constexpr int BA_OFF_PHASE = 48;
static constexpr int BA_OFF_MOST_PLAYED = 54;
static constexpr int BA_OFF_LAST_HAND = 62;
static constexpr int BA_OFF_LOCK_EMBED = 70;
static constexpr int BA_OFF_PLAYED_HANDS = 78;
static constexpr int BA_OFF_PLANET_USAGE = 86;
static constexpr int BA_OFF_SKIP_EMBED = 94;
static constexpr int BA_OFF_TAG_EMBED = 98;
static constexpr int BA_OFF_BLIND_TAGS = 102;
static constexpr int BA_OFF_ANCIENT_SUIT = 110;
static constexpr int BA_OFF_IDOL_RANK = 118;
static constexpr int BA_OFF_IDOL_SUIT = 126;
static constexpr int BA_OFF_MAIL_RANK = 134;
static constexpr int BA_OFF_CASTLE_SUIT = 142;
static constexpr int BA_OFF_ORBITAL = 150;
static constexpr int BA_OFF_SCALARS = 174;
static constexpr int BA_SCALAR_FEATURES = 66;
static constexpr int BA_GLOBALS_FEATURES = BA_OFF_SCALARS + BA_SCALAR_FEATURES;
static constexpr int BA_DECK_FEATURES = 208;
static constexpr int BA_OFF_DECK = BA_OFF_GLOBALS + BA_GLOBALS_FEATURES;
static constexpr int BA_OFF_VOUCHER = BA_OFF_DECK + BA_DECK_FEATURES;
static constexpr int BA_VOUCHER_FEATURES = 64;
static constexpr int BA_OFF_COUNTS = BA_OFF_VOUCHER + BA_VOUCHER_FEATURES;
static constexpr int BA_COUNT_FEATURES = 11;
static constexpr int BA_OFF_POKER = BA_OFF_COUNTS + BA_COUNT_FEATURES;
static constexpr int BA_POKER_FEATURES = HAND_COUNT * 6;  // 12 * 6 = 72
static constexpr int BA_OFF_BLIND_SIG = BA_OFF_POKER + BA_POKER_FEATURES;
static constexpr int BA_BLIND_SIG_FEATURES = 48;          // 24 for current blind + 24 for upcoming boss
static constexpr int BA_FIXED = BA_OFF_BLIND_SIG + BA_BLIND_SIG_FEATURES;

static constexpr int BA_POOL_SECTIONS = 7;                // hand, jokers, consumables, shop, vouchers, boosters, pack
static constexpr int BA_POOLED_FEATURES = BA_POOL_SECTIONS * BA_TOKEN_DIM;
static constexpr int BA_OFF_POOLED = BA_FIXED;
static constexpr int BA_MAX_POOL_SECTIONS = 4;
static constexpr int BA_MAX_POOL_FEATURES = BA_MAX_POOL_SECTIONS * BA_TOKEN_DIM;
static constexpr int BA_OFF_MAX_POOL = BA_OFF_POOLED + BA_POOLED_FEATURES;
static constexpr int BA_OFF_SPECIAL = BA_OFF_MAX_POOL + BA_MAX_POOL_FEATURES;
static constexpr int BA_SPECIAL_FEATURES = BA_TOKEN_DIM;
static constexpr int BA_HAND_STRUCT_FEATURES = 71;
static constexpr int BA_OFF_HAND_STRUCT = BA_OFF_SPECIAL + BA_SPECIAL_FEATURES;
static constexpr int BA_HAND_SUM_FEATURES = BA_TOKEN_DIM;
static constexpr int BA_OFF_HAND_SUM = BA_OFF_HAND_STRUCT + BA_HAND_STRUCT_FEATURES;
static constexpr int BA_TOTAL = BA_OFF_HAND_SUM + BA_HAND_SUM_FEATURES;
static constexpr int ENTITY_COUNT = BA_KEY_CAP + DECODER_CATEGORIES;
static constexpr int ENTITY_WIDTH = AR_EMBED_DIM;
static constexpr int DECODER_COLUMNS = DECODER_STATE + 1;
static_assert(DECODER_STEPS == ACTION_STORAGE_SIZE);
static_assert(sizeof(Observation) == 4296,
    "Balatro encoder must be updated for the Observation layout");
static_assert(BA_KEY_DIM == 32, "Balatro pointer layout requires 32 channels");
static_assert(BA_FIXED == 643 && BA_TOTAL == 1130,
    "Balatro encoder feature layout mismatch");

__device__ __forceinline__ int token_start(
        const int* counts, int zone) {
    int start = 0;
    for (int current = 0; current < zone; ++current) {
        start += counts[current];
    }
    return start;
}

__device__ __forceinline__ int primary_zone(int type) {
    switch (type) {
    case ACTION_BUY_CARD:
    case ACTION_BUY_AND_USE:
        return ZONE_SHOP_MAIN;
    case ACTION_SELL_JOKER:
    case ACTION_SWAP_JOKERS_LEFT:
    case ACTION_SWAP_JOKERS_RIGHT:
        return ZONE_JOKER;
    case ACTION_SELL_CONSUMABLE:
    case ACTION_USE_CONSUMABLE:
        return ZONE_CONSUMABLE;
    case ACTION_REDEEM_VOUCHER:
        return ZONE_SHOP_VOUCHER;
    case ACTION_OPEN_BOOSTER:
        return ZONE_SHOP_BOOSTER;
    case ACTION_PICK_PACK_CARD:
        return ZONE_PACK_CARD;
    case ACTION_SWAP_HAND_LEFT:
    case ACTION_SWAP_HAND_RIGHT:
        return ZONE_HAND;
    default:
        return -1;
    }
}

enum BlindSpecialRule : uint8_t {
    BLIND_SPECIAL_ARM = 1u << 0,
    BLIND_SPECIAL_ACORN = 1u << 1,
    BLIND_SPECIAL_BELL = 1u << 2,
    BLIND_SPECIAL_HEART = 1u << 3,
    BLIND_SPECIAL_LEAF = 1u << 4,
    BLIND_SPECIAL_SERPENT = 1u << 5,
};

enum BlindFacedownRule : uint8_t {
    BLIND_FACEDOWN_HOUSE = 1u << 0,
    BLIND_FACEDOWN_WHEEL = 1u << 1,
    BLIND_FACEDOWN_FISH = 1u << 2,
    BLIND_FACEDOWN_MARK = 1u << 3,
};

struct BlindSignature {
    uint8_t debuff_suit;            // 0=Spades, 1=Hearts, 2=Clubs, 3=Diamonds, 255=None
    uint8_t debuff_face;            // The Plant (1)
    uint8_t debuff_played;          // The Pillar (1)
    uint8_t special_mask;
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
    uint8_t facedown_mask;
    uint8_t hand_size_reduced;      // The Manacle (1)
};

static __device__ __constant__ BlindSignature BLIND_SIGNATURES[31] = {
    /* 0 */                         {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 1: BLIND_BL_ARM */           {255, 0, 0, BLIND_SPECIAL_ARM, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 2: BLIND_BL_BIG */           {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 3: BLIND_BL_CLUB */          {2,   0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 4: BLIND_BL_EYE */           {255, 0, 0, 0, 1, 0, 255, 0, 1, 0, 1, 0, 0, 0, 0, 0},
    /* 5: BLIND_BL_FINAL_ACORN */   {255, 0, 0, BLIND_SPECIAL_ACORN, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 6: BLIND_BL_FINAL_BELL */    {255, 0, 0, BLIND_SPECIAL_BELL, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 7: BLIND_BL_FINAL_HEART */   {255, 0, 0, BLIND_SPECIAL_HEART, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 8: BLIND_BL_FINAL_LEAF */    {255, 0, 0, BLIND_SPECIAL_LEAF, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 9: BLIND_BL_FINAL_VESSEL */  {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 3, 0, 0, 0, 0, 0},
    /* 10: BLIND_BL_FISH */         {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, BLIND_FACEDOWN_FISH, 0},
    /* 11: BLIND_BL_FLINT */        {255, 0, 0, 0, 1, 0, 255, 0, 0, 1, 1, 0, 0, 0, 0, 0},
    /* 12: BLIND_BL_GOAD */         {0,   0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 13: BLIND_BL_HEAD */         {1,   0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 14: BLIND_BL_HOOK */         {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 1, 0, 0, 0, 0},
    /* 15: BLIND_BL_HOUSE */        {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, BLIND_FACEDOWN_HOUSE, 0},
    /* 16: BLIND_BL_MANACLE */      {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 1},
    /* 17: BLIND_BL_MARK */         {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, BLIND_FACEDOWN_MARK, 0},
    /* 18: BLIND_BL_MOUTH */        {255, 0, 0, 0, 1, 0, 255, 1, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 19: BLIND_BL_NEEDLE */       {255, 0, 0, 0, 1, 1, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 20: BLIND_BL_OX */           {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 1, 0, 0},
    /* 21: BLIND_BL_PILLAR */       {255, 0, 1, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 22: BLIND_BL_PLANT */        {255, 1, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 23: BLIND_BL_PSYCHIC */      {255, 0, 0, 0, 5, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 24: BLIND_BL_SERPENT */      {255, 0, 0, BLIND_SPECIAL_SERPENT, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 25: BLIND_BL_SMALL */        {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 26: BLIND_BL_TOOTH */        {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 1, 0, 0, 0},
    /* 27: BLIND_BL_WALL */         {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 2, 0, 0, 0, 0, 0},
    /* 28: BLIND_BL_WATER */        {255, 0, 0, 0, 1, 0, 0,   0, 0, 0, 1, 0, 0, 0, 0, 0},
    /* 29: BLIND_BL_WHEEL */        {255, 0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, BLIND_FACEDOWN_WHEEL, 0},
    /* 30: BLIND_BL_WINDOW */       {3,   0, 0, 0, 1, 0, 255, 0, 0, 0, 1, 0, 0, 0, 0, 0},
};

__device__ __forceinline__ float blind_sig_value(const BlindSignature* sig, int d) {
    switch (d) {
    case 0: return sig->debuff_suit < 4 ? ((float)sig->debuff_suit + 1.0f) / 4.0f : 0.0f;
    case 1: return (float)sig->debuff_face;
    case 2: return (float)sig->debuff_played;
    case 3: case 4: case 5: case 6: case 7: case 8:
        return (float)((sig->special_mask >> (d - 3)) & 1u);
    case 9: return sig->min_play_size > 1 ? (float)sig->min_play_size / 5.0f : 0.0f;
    case 10: return sig->max_hands > 0 ? (float)sig->max_hands / 4.0f : 0.0f;
    case 11: return sig->max_discards == 0 ? 1.0f : 0.0f;
    case 12: return (float)sig->single_hand_type;
    case 13: return (float)sig->no_repeat_hands;
    case 14: return (float)sig->base_score_halved;
    case 15: return (float)sig->chips_mult / 3.0f;
    case 16: return (float)sig->discard_random_on_play;
    case 17: return (float)sig->money_lost_on_play;
    case 18: return (float)sig->set_money_zero_on_most;
    case 19: case 20: case 21: case 22:
        return (float)((sig->facedown_mask >> (d - 19)) & 1u);
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
    Prec token_w;       // (BA_TOKEN_DIM, BA_RAW_DIM)
    Prec token_b;       // (BA_TOKEN_DIM)
    Prec token_embed;   // (BA_TOKEN_EMBED_ROWS, BA_TOKEN_DIM)
    Prec state_embed;   // (BA_STATE_EMBED_ROWS, BA_STATE_EMBED_DIM)
    Prec proj_w;        // (hidden, BA_TOTAL)
    int obs_size, hidden;
};

struct BalatroEncoderActivations {
    Prec pooled, out, d_pooled;
    Prec keys;
    Float key_grad;
    Int counts, relu_mask, pool_argmax;
    Long token_embed_acc, state_embed_acc, token_w_acc, token_b_acc;
    Prec token_wgrad, token_bgrad, token_embed_grad, state_embed_grad;
    Prec proj_wgrad;
    const unsigned char* obs_data;
    int obs_batch;
};

static BalatroEncoderActivations* ba_enc_last = NULL;

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
    int base = offsetof(Observation, counts);
    counts[0] = ba_byte(obs, in, base + offsetof(ObservationCounts, hand_count));
    counts[1] = ba_byte(obs, in, base + offsetof(ObservationCounts, joker_count));
    counts[2] = ba_byte(obs, in, base + offsetof(ObservationCounts, consumable_count));
    counts[3] = ba_byte(obs, in, base + offsetof(ObservationCounts, shop_count));
    counts[4] = ba_byte(obs, in, base + offsetof(ObservationCounts, voucher_count));
    counts[5] = ba_byte(obs, in, base + offsetof(ObservationCounts, booster_count));
    counts[6] = ba_byte(obs, in, base + offsetof(ObservationCounts, pack_count));
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

struct TokenInfo {
    int id;
    int zone;
    int playing;
    int rank;
    int suit;
    int enhancement;
    int edition;
    int seal;
    int flags;
    int cost;
    int sell_cost;
    int perma_bonus;
    int state[4];
};

__device__ __forceinline__ TokenInfo ba_token_info(
        const unsigned char* obs, int64_t in, const int counts[BA_POOL_SECTIONS], int token_idx) {
    int local;
    int zone = ba_slot_section(counts, token_idx, &local);
    assert(zone >= 0 && local >= 0);
    TokenInfo info = {};
    info.zone = zone;
    if (zone == ZONE_HAND) {
        int base = offsetof(Observation, hand)
            + local * sizeof(PlayingCardView);
        int attributes = ba_u16(obs, in, base + offsetof(PlayingCardView, attributes));
        info.playing = 1;
        info.rank = attributes & 15;
        info.suit = (attributes >> 4) & 3;
        info.enhancement = (attributes >> 6) & 15;
        info.edition = (attributes >> 10) & 7;
        info.seal = (attributes >> 13) & 7;
        info.flags = ba_byte(obs, in, base + offsetof(PlayingCardView, flags));
        info.perma_bonus = (int16_t)ba_u16(obs, in, base + offsetof(PlayingCardView, perma_bonus));
    } else if (zone == ZONE_JOKER) {
        int base = offsetof(Observation, jokers) + local * sizeof(JokerView);
        info.id = ba_u16(obs, in, base + offsetof(JokerView, center_id));
        info.edition = ba_byte(obs, in, base + offsetof(JokerView, edition));
        info.flags = ba_byte(obs, in, base + offsetof(JokerView, flags));
        info.cost = (int16_t)ba_u16(obs, in, base + offsetof(JokerView, cost));
        info.sell_cost = (int16_t)ba_u16(obs, in, base + offsetof(JokerView, sell_cost));
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            info.state[i] = (int32_t)ba_u32(obs, in, base + offsetof(JokerView, state) + 4 * i);
    } else if (zone == ZONE_CONSUMABLE) {
        int base = offsetof(Observation, consumables) + local * sizeof(ConsumableView);
        info.id = ba_u16(obs, in, base + offsetof(ConsumableView, center_id));
        info.edition = ba_byte(obs, in, base + offsetof(ConsumableView, edition));
        info.flags = ba_byte(obs, in, base + offsetof(ConsumableView, flags));
        info.cost = (int16_t)ba_u16(obs, in, base + offsetof(ConsumableView, cost));
        info.sell_cost = (int16_t)ba_u16(obs, in, base + offsetof(ConsumableView, sell_cost));
    } else if (zone == ZONE_SHOP_MAIN || zone == ZONE_PACK_CARD) {
        int base = offsetof(Observation, market)
            + (zone == ZONE_SHOP_MAIN ? offsetof(MarketView, shop) : offsetof(MarketView, pack))
            + local * sizeof(OfferView);
        info.id = ba_u16(obs, in, base + offsetof(OfferView, center_id));
        info.rank = ba_byte(obs, in, base + offsetof(OfferView, rank));
        info.suit = ba_byte(obs, in, base + offsetof(OfferView, suit));
        info.enhancement = ba_byte(obs, in, base + offsetof(OfferView, enhancement));
        info.edition = ba_byte(obs, in, base + offsetof(OfferView, edition));
        info.seal = ba_byte(obs, in, base + offsetof(OfferView, seal));
        info.flags = ba_byte(obs, in, base + offsetof(OfferView, flags));
        info.perma_bonus = (int16_t)ba_u16(obs, in, base + offsetof(OfferView, perma_bonus));
        info.cost = (int16_t)ba_u16(obs, in, base + offsetof(OfferView, cost));
        info.sell_cost = (int16_t)ba_u16(obs, in, base + offsetof(OfferView, sell_cost));
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            info.state[i] = (int32_t)ba_u32(obs, in, base + offsetof(OfferView, state) + 4 * i);
        info.playing = info.rank >= 2 && info.rank <= 14 && info.suit < 4;
    } else {
        int base = offsetof(Observation, market)
            + (zone == ZONE_SHOP_VOUCHER ? offsetof(MarketView, vouchers) : offsetof(MarketView, boosters))
            + local * sizeof(MarketCardView);
        info.id = ba_u16(obs, in, base + offsetof(MarketCardView, center_id));
        info.cost = (int16_t)ba_u16(obs, in, base + offsetof(MarketCardView, cost));
        info.sell_cost = (int16_t)ba_u16(obs, in, base + offsetof(MarketCardView, sell_cost));
    }
    if (info.playing) info.id = (info.rank << 8) | info.suit;
    return info;
}

__device__ __forceinline__ void ba_token_features(
        const unsigned char* obs, int64_t in, const int counts[BA_POOL_SECTIONS],
        int token_idx, float v[BA_RAW_DIM]) {
    #pragma unroll
    for (int d = 0; d < BA_RAW_DIM; ++d) v[d] = 0.0f;
    TokenInfo info = ba_token_info(obs, in, counts, token_idx);
    if (info.playing) {
        v[0] = (float)info.rank / 14.0f;
        v[1] = (float)info.suit / 3.0f;
        v[2] = (float)info.enhancement / 8.0f;
        v[3] = (float)info.edition / 4.0f;
        v[4] = (float)info.seal / 4.0f;
        v[5] = (float)info.flags / 127.0f;
        v[6] = (float)info.sell_cost / 16.0f;
        v[7] = 1.0f;
        for (int k = 0; k < 7; ++k) v[8 + k] = (k == info.zone) ? 1.0f : 0.0f;
        for (int k = 0; k < 4; ++k) v[15 + k] = (k == info.suit) ? 1.0f : 0.0f;
        v[19] = (info.rank >= 11 && info.rank <= 13) ? 1.0f : 0.0f;
        v[20] = (info.rank == 14) ? 1.0f : 0.0f;
        v[21] = (info.rank % 2 == 0) ? 1.0f : 0.0f;
        v[22] = (info.rank % 2 == 1 && info.rank != 14) ? 1.0f : 0.0f;
        int suit_count = 0;
        int rank_count = 0;
        if (info.zone == ZONE_HAND) {
            int hand_count = counts[ZONE_HAND];
            for (int h = 0; h < hand_count; ++h) {
                int h_base = offsetof(Observation, hand) + h * sizeof(PlayingCardView);
                int attr = ba_u16(obs, in, h_base + offsetof(PlayingCardView, attributes));
                int h_rank = attr & 15;
                int h_suit = (attr >> 4) & 3;
                if (h_suit == info.suit) suit_count++;
                if (h_rank == info.rank) rank_count++;
            }
        }
        v[23] = (float)suit_count / 5.0f;
        v[24] = (float)info.perma_bonus / (fabsf((float)info.perma_bonus) + 10.0f);
        v[25] = (float)info.perma_bonus / 100.0f;
        v[26] = (float)rank_count / 4.0f;
    } else {
        int center = info.id;
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
        v[16] = (float)info.enhancement / 8.0f;
        v[17] = (float)info.edition / 4.0f;
        v[18] = (float)info.seal / 4.0f;
        v[19] = (float)info.flags / 127.0f;
        v[20] = (float)info.cost / 16.0f;
        v[21] = (float)info.sell_cost / 16.0f;
        v[22] = (float)info.zone / 6.0f;
        v[23] = (float)info.perma_bonus / (fabsf((float)info.perma_bonus) + 10.0f);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float state = (float)info.state[i];
            v[24 + i] = state / (fabsf(state) + 10.0f);
            v[28 + i] = state / 1000.0f;
        }
    }
}

__global__ void encode_globals(
        precision_t* __restrict__ pooled,
        const precision_t* __restrict__ state_embed,
        const unsigned char* __restrict__ obs, int B, int obs_size) {
    int b = blockIdx.x;
    if (b >= B) return;
    int64_t in = (int64_t)b * obs_size;
    int base = offsetof(Observation, globals);
    int deck = ba_byte(obs, in, base + offsetof(ObservationGlobals, deck_id));
    int blind = ba_u16(obs, in, base + offsetof(ObservationGlobals, blind_id));
    int boss = ba_u16(obs, in, base + offsetof(ObservationGlobals, next_boss_id));
    int voucher = ba_u16(obs, in, base + offsetof(ObservationGlobals, next_voucher_id));
    int tarot = ba_u16(obs, in, base + offsetof(ObservationGlobals, last_tarot_planet));
    int pending = ba_u16(obs, in,
        base + offsetof(ObservationGlobals, pending_free_pack_id));

    static constexpr int scalar_offset[44] = {
        offsetof(ObservationGlobals, stake), offsetof(ObservationGlobals, win_ante),
        offsetof(ObservationGlobals, stake_scaling), offsetof(ObservationGlobals, blind_on_deck),
        offsetof(ObservationGlobals, ante), offsetof(ObservationGlobals, round),
        offsetof(ObservationGlobals, blind_disabled), offsetof(ObservationGlobals, hand_sort_suit),
        offsetof(ObservationGlobals, boss_rerolled), offsetof(ObservationGlobals, free_rerolls),
        offsetof(ObservationGlobals, reroll_base), offsetof(ObservationGlobals, reroll_increase),
        offsetof(ObservationGlobals, discount_percent), offsetof(ObservationGlobals, hands_per_round),
        offsetof(ObservationGlobals, discards_per_round), offsetof(ObservationGlobals, base_hand_size),
        offsetof(ObservationGlobals, hand_size), offsetof(ObservationGlobals, joker_slots),
        offsetof(ObservationGlobals, consumable_slots), offsetof(ObservationGlobals, skips),
        offsetof(ObservationGlobals, pack_choices), offsetof(ObservationGlobals, pack_kind),
        offsetof(ObservationGlobals, shop_return_phase), offsetof(ObservationGlobals, first_shop_buffoon),
        offsetof(ObservationGlobals, shop_joker_max), offsetof(ObservationGlobals, double_tag),
        offsetof(ObservationGlobals, tag_hand_bonus), offsetof(ObservationGlobals, tag_force_rarity),
        offsetof(ObservationGlobals, tag_force_rarity_count), offsetof(ObservationGlobals, tag_force_edition),
        offsetof(ObservationGlobals, tag_force_edition_count), offsetof(ObservationGlobals, tag_voucher_pending),
        offsetof(ObservationGlobals, tag_coupon_pending), offsetof(ObservationGlobals, tag_coupon_active),
        offsetof(ObservationGlobals, tag_saved_discount),
        offsetof(ObservationGlobals, tag_investment_pending), offsetof(ObservationGlobals, tag_d_six_pending),
        offsetof(ObservationGlobals, tag_d_six_active), offsetof(ObservationGlobals, ecto_penalty),
        offsetof(ObservationGlobals, gros_michel_extinct), offsetof(ObservationGlobals, hands_left),
        offsetof(ObservationGlobals, discards_left), offsetof(ObservationGlobals, hands_played),
        offsetof(ObservationGlobals, discards_used),
    };

    // 1. Unpack fixed globals, deck multiplicities, and counters.
    // Exact rank-suit multiplicities for plain unseen and discarded cards.
    int deck_base = offsetof(Observation, plain_deck);
    for (int i = threadIdx.x; i < 52; i += blockDim.x) {
        int slot_off = deck_base + i * sizeof(PlainDeckView);
        pooled[b * BA_TOTAL + BA_OFF_DECK + i * 4] = from_float(
            (float)ba_u16(obs, in, slot_off + offsetof(PlainDeckView, unseen)) / 52.0f);
        pooled[b * BA_TOTAL + BA_OFF_DECK + i * 4 + 1] = from_float(
            (float)ba_u16(obs, in, slot_off + offsetof(PlainDeckView, discard)) / 52.0f);
        pooled[b * BA_TOTAL + BA_OFF_DECK + i * 4 + 2] = from_float(
            (float)ba_u16(obs, in, slot_off + offsetof(PlainDeckView, played_unseen)) / 52.0f);
        pooled[b * BA_TOTAL + BA_OFF_DECK + i * 4 + 3] = from_float(
            (float)ba_u16(obs, in, slot_off + offsetof(PlainDeckView, played_discard)) / 52.0f);
    }

    // 1B. Poker hands (72 features: 470..541)
    for (int rec = threadIdx.x; rec < HAND_COUNT; rec += blockDim.x) {
        int record = offsetof(Observation, poker_hands) + rec * sizeof(PokerHandStat);
        pooled[b * BA_TOTAL + BA_OFF_POKER + rec * 6 + 0] = from_float((float)ba_byte(obs, in, record + offsetof(PokerHandStat, visible)));
        pooled[b * BA_TOTAL + BA_OFF_POKER + rec * 6 + 1] = from_float((float)ba_byte(obs, in, record + offsetof(PokerHandStat, level)) / 16.0f);
        pooled[b * BA_TOTAL + BA_OFF_POKER + rec * 6 + 2] = from_float(ba_float(obs, in, record + offsetof(PokerHandStat, chips_log2)) / 16.0f);
        pooled[b * BA_TOTAL + BA_OFF_POKER + rec * 6 + 3] = from_float(ba_float(obs, in, record + offsetof(PokerHandStat, mult_log2)) / 16.0f);
        pooled[b * BA_TOTAL + BA_OFF_POKER + rec * 6 + 4] = from_float((float)ba_u16(obs, in, record + offsetof(PokerHandStat, total_plays)) / 256.0f);
        pooled[b * BA_TOTAL + BA_OFF_POKER + rec * 6 + 5] = from_float(
            (float)ba_byte(obs, in, record + offsetof(PokerHandStat, round_plays)) / 64.0f);
    }

    for (int bit = threadIdx.x; bit < 64; bit += blockDim.x) {
        int byte = ba_byte(obs, in, base
            + offsetof(ObservationGlobals, redeemed_vouchers_mask) + (bit >> 3));
        pooled[b * BA_TOTAL + BA_OFF_VOUCHER + bit]
            = from_float((float)((byte >> (bit & 7)) & 1));
    }

    for (int i = threadIdx.x; i < BA_COUNT_FEATURES; i += blockDim.x) {
        static constexpr int count_offsets[11] = {
            offsetof(Observation, counts) + offsetof(ObservationCounts, hand_count),
            offsetof(Observation, counts) + offsetof(ObservationCounts, joker_count),
            offsetof(Observation, counts) + offsetof(ObservationCounts, consumable_count),
            offsetof(Observation, counts) + offsetof(ObservationCounts, shop_count),
            offsetof(Observation, counts) + offsetof(ObservationCounts, voucher_count),
            offsetof(Observation, counts) + offsetof(ObservationCounts, booster_count),
            offsetof(Observation, counts) + offsetof(ObservationCounts, pack_count),
            offsetof(Observation, counts) + offsetof(ObservationCounts, deck_count),
            offsetof(Observation, counts) + offsetof(ObservationCounts, discard_count),
            offsetof(Observation, counts) + offsetof(ObservationCounts, special_count),
            offsetof(Observation, counts) + offsetof(ObservationCounts, total_playing_cards),
        };
        static constexpr float count_scales[11] = {
            1.0f / 16.0f, 1.0f / 8.0f, 1.0f / 4.0f, 1.0f / 4.0f,
            1.0f / 2.0f,  1.0f / 2.0f, 1.0f / 5.0f,
            1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f,
        };
        int raw = i < 7 ? ba_byte(obs, in, count_offsets[i])
            : ba_u16(obs, in, count_offsets[i]);
        pooled[b * BA_TOTAL + BA_OFF_COUNTS + i]
            = from_float((float)raw * count_scales[i]);
    }

    // 1E. Blind Signatures (48 features: 555..602)
    for (int local = threadIdx.x; local < 48; local += blockDim.x) {
        float val = (local < 24)
            ? ((blind >= 0 && blind < 31) ? blind_sig_value(&BLIND_SIGNATURES[blind], local) : 0.0f)
            : ((boss >= 0 && boss < 31) ? blind_sig_value(&BLIND_SIGNATURES[boss], local - 24) : 0.0f);
        pooled[b * BA_TOTAL + BA_OFF_BLIND_SIG + local] = from_float(val);
    }

    // 1G. Globals, categorical state, scalars, and statistics.
    for (int local = threadIdx.x; local < BA_GLOBALS_FEATURES; local += blockDim.x) {
        float value = 0.0f;
        if (local < 48) {
            int d = local & 7;
            int row = local >> 3;
            if (row == 0) {
                int d_id = (deck >= 0 && deck < 16) ? deck : 0;
                value = to_float(state_embed[(BA_STATE_DECK_ROW + d_id) * 8 + d]);
            } else if (row == 1) {
                int b_id = (blind >= 0 && blind < BLIND_COUNT) ? blind : 0;
                value = to_float(state_embed[(BA_STATE_BLIND_ROW + b_id) * 8 + d]);
            } else if (row == 2) {
                int b_id = (boss >= 0 && boss < BLIND_COUNT) ? boss : 0;
                value = to_float(state_embed[(BA_STATE_BLIND_ROW + b_id) * 8 + d]);
            } else if (row == 3) {
                int v_idx = (voucher >= CENTER_V_ANTIMATTER && voucher <= CENTER_V_WASTEFUL)
                    ? (voucher - CENTER_V_ANTIMATTER + 1) : 0;
                value = to_float(state_embed[(BA_STATE_VOUCHER_ROW + v_idx) * 8 + d]);
            } else if (row == 4) {
                int t_id = (tarot >= 0 && tarot < CENTER_COUNT) ? tarot : 0;
                value = to_float(state_embed[(BA_STATE_TAROT_ROW + t_id) * 8 + d]);
            } else {
                int p_id = (pending >= 0 && pending < CENTER_COUNT) ? pending : 0;
                value = to_float(state_embed[(BA_STATE_TAROT_ROW + p_id) * 8 + d]);
            }
        } else if (local < BA_OFF_MOST_PLAYED) {
            int phase = ba_byte(obs, in, base + offsetof(ObservationGlobals, phase));
            value = (local - BA_OFF_PHASE) == phase ? 1.0f : 0.0f;
        } else if (local < BA_OFF_LAST_HAND) {
            int most = ba_byte(obs, in, base + offsetof(ObservationGlobals, most_played_hand));
            int hand = most < HAND_COUNT ? most : HAND_COUNT;
            value = to_float(state_embed[(BA_STATE_LOCK_ROW + hand) * 8
                + local - BA_OFF_MOST_PLAYED]);
        } else if (local < BA_OFF_LOCK_EMBED) {
            int last = ba_byte(obs, in, base + offsetof(ObservationGlobals, last_hand_type));
            int hand = last < HAND_COUNT ? last : HAND_COUNT;
            value = to_float(state_embed[(BA_STATE_LOCK_ROW + hand) * 8
                + local - BA_OFF_LAST_HAND]);
        } else if (local < BA_OFF_PLAYED_HANDS) {
            int d = local - BA_OFF_LOCK_EMBED;
            int lock_raw = ba_byte(obs, in, base + offsetof(ObservationGlobals, blind_only_hand));
            int lock_idx = (lock_raw < HAND_COUNT) ? lock_raw : HAND_COUNT;
            value = to_float(state_embed[(BA_STATE_LOCK_ROW + lock_idx) * 8 + d]);
        } else if (local < BA_OFF_PLANET_USAGE) {
            int d = local - BA_OFF_PLAYED_HANDS;
            uint16_t hands_mask = ba_u16(obs, in, base + offsetof(ObservationGlobals, blind_hands_mask));
            float sum = 0.0f;
            uint32_t m = hands_mask & 0x0FFF;
            while (m) {
                int h = __ffs(m) - 1;
                m &= m - 1;
                sum += to_float(state_embed[(BA_STATE_PLAYED_HAND_ROW + h) * 8 + d]);
            }
            value = sum;
        } else if (local < BA_OFF_SKIP_EMBED) {
            int d = local - BA_OFF_PLANET_USAGE;
            uint16_t planet_mask = ba_u16(obs, in, base + offsetof(ObservationGlobals, planet_usage_mask));
            float sum = 0.0f;
            uint32_t m = planet_mask & 0x0FFF;
            while (m) {
                int h = __ffs(m) - 1;
                m &= m - 1;
                sum += to_float(state_embed[(BA_STATE_PLANET_HAND_ROW + h) * 8 + d]);
            }
            value = sum;
        } else if (local < BA_OFF_TAG_EMBED) {
            int d = local - BA_OFF_SKIP_EMBED;
            uint8_t skip_mask = ba_byte(obs, in, base + offsetof(ObservationGlobals, blind_skipped_mask));
            float sum = 0.0f;
            for (int s = 0; s < 3; ++s) {
                if ((skip_mask >> s) & 1)
                    sum += to_float(state_embed[(BA_STATE_SKIP_ROW + s) * 8 + d]);
            }
            value = sum;
        } else if (local < BA_OFF_BLIND_TAGS) {
            int d = local - BA_OFF_TAG_EMBED;
            int atag = ba_byte(obs, in, base + offsetof(ObservationGlobals, active_tag));
            if (atag >= TAG_COUNT) atag = 0;
            value = to_float(state_embed[(BA_STATE_TAG_ROW + atag) * 8 + d]);
        } else if (local < BA_OFF_ANCIENT_SUIT) {
            int item = (local - BA_OFF_BLIND_TAGS) >> 2;
            int d = (local - BA_OFF_BLIND_TAGS) & 3;
            int tag = ba_byte(obs, in, base
                + offsetof(ObservationGlobals, blind_tags) + item);
            if (tag >= TAG_COUNT) tag = 0;
            value = to_float(state_embed[(BA_STATE_TAG_ROW + tag) * 8 + d]);
        } else if (local < BA_OFF_IDOL_RANK) {
            int suit = ba_byte(obs, in, base + offsetof(ObservationGlobals, ancient_suit));
            value = to_float(state_embed[(BA_STATE_SUIT_ROW + suit) * 8
                + local - BA_OFF_ANCIENT_SUIT]);
        } else if (local < BA_OFF_IDOL_SUIT) {
            int rank = ba_byte(obs, in, base + offsetof(ObservationGlobals, idol_rank));
            value = to_float(state_embed[(BA_STATE_RANK_ROW + rank) * 8
                + local - BA_OFF_IDOL_RANK]);
        } else if (local < BA_OFF_MAIL_RANK) {
            int suit = ba_byte(obs, in, base + offsetof(ObservationGlobals, idol_suit));
            value = to_float(state_embed[(BA_STATE_SUIT_ROW + suit) * 8
                + local - BA_OFF_IDOL_SUIT]);
        } else if (local < BA_OFF_CASTLE_SUIT) {
            int rank = ba_byte(obs, in, base + offsetof(ObservationGlobals, mail_rank));
            value = to_float(state_embed[(BA_STATE_RANK_ROW + rank) * 8
                + local - BA_OFF_MAIL_RANK]);
        } else if (local < BA_OFF_ORBITAL) {
            int suit = ba_byte(obs, in, base + offsetof(ObservationGlobals, castle_suit));
            value = to_float(state_embed[(BA_STATE_SUIT_ROW + suit) * 8
                + local - BA_OFF_CASTLE_SUIT]);
        } else if (local < BA_OFF_SCALARS) {
            int item = (local - BA_OFF_ORBITAL) >> 3;
            int d = (local - BA_OFF_ORBITAL) & 7;
            int hand = ba_byte(obs, in, base
                + offsetof(ObservationGlobals, orbital_hands) + item);
            if (hand >= HAND_COUNT) hand = HAND_COUNT;
            value = to_float(state_embed[(BA_STATE_LOCK_ROW + hand) * 8 + d]);
        } else if (local < BA_OFF_SCALARS + 44) {
            int s = local - BA_OFF_SCALARS;
            value = (float)ba_byte(obs, in, base + scalar_offset[s]) / 16.0f;
        } else if (local < BA_OFF_SCALARS + 46) {
            int s = local - (BA_OFF_SCALARS + 44);
            int off = s == 0 ? offsetof(ObservationGlobals, unused_discards)
                : offsetof(ObservationGlobals, tarots_used);
            value = (float)ba_u16(obs, in, base + off) / 256.0f;
        } else if (local < BA_OFF_SCALARS + 50) {
            int s = local - (BA_OFF_SCALARS + 46);
            if (s == 0) value = (float)(int16_t)ba_u16(obs, in,
                base + offsetof(ObservationGlobals, interest_cap)) / 64.0f;
            else {
                static constexpr int offsets[3] = {
                    offsetof(ObservationGlobals, interest_amount),
                    offsetof(ObservationGlobals, rental_rate),
                    offsetof(ObservationGlobals, blind_reward),
                };
                value = (float)(int8_t)ba_byte(obs, in, base + offsets[s - 1]) / 64.0f;
            }
        } else if (local == BA_OFF_SCALARS + 50) {
            value = (float)ba_u32(obs, in,
                base + offsetof(ObservationGlobals, run_hands_played)) / 4096.0f;
        } else if (local < BA_OFF_SCALARS + 54) {
            int s = local - (BA_OFF_SCALARS + 51);
            static constexpr int i32_stat_offsets[3] = {
                offsetof(ObservationGlobals, dollars), offsetof(ObservationGlobals, reroll_cost),
                offsetof(ObservationGlobals, round_earnings),
            };
            value = (float)(int32_t)ba_u32(obs, in, base + i32_stat_offsets[s]) / 64.0f;
        } else if (local < BA_OFF_SCALARS + 60) {
            int s = local - (BA_OFF_SCALARS + 54);
            value = ba_float(obs, in, base
                + offsetof(ObservationGlobals, score_features) + 4 * s) / 16.0f;
        } else if (local < BA_OFF_SCALARS + BA_SCALAR_FEATURES) {
            int s = local - (BA_OFF_SCALARS + 60);
            value = ba_float(obs, in, base + offsetof(ObservationGlobals, joker_rate) + 4 * s) / 20.0f;
        }
        pooled[b * BA_TOTAL + BA_OFF_GLOBALS + local] = from_float(value);
    }
}

// Token work uses two waves per observation; fixed-field decoding has its
// own wider block so its register lifetime does not constrain token processing.
template<int Threads>
__global__ void __launch_bounds__(Threads, 4) encode_tokens(
        precision_t* __restrict__ pooled,
        precision_t* __restrict__ keys,
        int* __restrict__ counts_out,
        int* __restrict__ relu_mask,
        int* __restrict__ pool_argmax,
        const precision_t* __restrict__ token_w,
        const precision_t* __restrict__ token_b,
        const precision_t* __restrict__ token_embed,
        const unsigned char* __restrict__ obs,
        int B, int obs_size) {
    static_assert(Threads >= 32 && Threads <= 256 && Threads % 32 == 0);
    __shared__ float s_pool_sum[BA_POOL_SECTIONS][BA_TOKEN_DIM];
    __shared__ float s_special_sum[BA_TOKEN_DIM];
    __shared__ unsigned long long s_pool_max[BA_MAX_POOL_SECTIONS][BA_TOKEN_DIM];
    __shared__ float s_raw[Threads / 32][BA_RAW_DIM];
    __shared__ int s_counts[BA_POOL_SECTIONS];
    __shared__ int s_total;
    __shared__ uint8_t s_struct_counts[BA_HAND_STRUCT_FEATURES];

    int b = blockIdx.x;
    if (b >= B) return;
    int64_t in = (int64_t)b * obs_size;
    // 2. Decode live tokens and apply the shared typed token MLP.
    if (threadIdx.x == 0) {
        ba_live_counts(obs, in, s_counts);
        s_total = 0;
        for (int s = 0; s < BA_POOL_SECTIONS; ++s) s_total += s_counts[s];
        assert(s_total <= BA_KEY_CAP);
    }
    __syncthreads();
    int total = s_total;
    if (threadIdx.x < BA_POOL_SECTIONS)
        counts_out[b * BA_POOL_SECTIONS + threadIdx.x] = s_counts[threadIdx.x];
    for (int i = threadIdx.x; i < BA_POOL_SECTIONS * BA_TOKEN_DIM; i += blockDim.x) {
        int s = i >> 5;
        int d = i & 31;
        s_pool_sum[s][d] = 0.0f;
    }
    if (threadIdx.x < BA_TOKEN_DIM) s_special_sum[threadIdx.x] = 0.0f;
    for (int i = threadIdx.x; i < BA_MAX_POOL_FEATURES; i += blockDim.x)
        s_pool_max[i >> 5][i & 31] = 0;
    for (int i = threadIdx.x; i < BA_HAND_STRUCT_FEATURES; i += blockDim.x)
        s_struct_counts[i] = 0;
    __syncthreads();
    if (threadIdx.x == 0) {
        int n_hand = s_counts[ZONE_HAND];
        s_struct_counts[70] = n_hand;
        for (int h = 0; h < n_hand; ++h) {
            int h_base = offsetof(Observation, hand) + h * sizeof(PlayingCardView);
            int attr = ba_u16(obs, in, h_base + offsetof(PlayingCardView, attributes));
            int rank = attr & 15;
            int suit = (attr >> 4) & 3;
            int enh = (attr >> 6) & 15;
            if (enh == ENHANCEMENT_STONE) {
                s_struct_counts[69]++;
            } else {
                if (rank >= 2 && rank <= 14)
                    s_struct_counts[rank - 2]++;
                if (enh == ENHANCEMENT_WILD) {
                    for (int s = 0; s < 4; ++s) {
                        s_struct_counts[13 + s]++;
                        if (rank >= 2 && rank <= 14)
                            s_struct_counts[17 + s * 13 + (rank - 2)]++;
                    }
                } else {
                    s_struct_counts[13 + suit]++;
                    if (rank >= 2 && rank <= 14)
                        s_struct_counts[17 + suit * 13 + (rank - 2)]++;
                }
            }
        }
    }
    __syncthreads();

    constexpr int TOKEN_WARP = 32;
    constexpr int TOKEN_WARPS = Threads / TOKEN_WARP;
    int lane = threadIdx.x & (TOKEN_WARP - 1);
    int warp = threadIdx.x / TOKEN_WARP;
    for (int slot = warp; slot < total; slot += TOKEN_WARPS) {
        int id, sec, playing, enh, ed, seal, flags, local;
        if (lane == 0) {
            sec = ba_slot_section(s_counts, slot, &local);
            ba_token_features(obs, in, s_counts, slot, s_raw[warp]);
            TokenInfo info = ba_token_info(obs, in, s_counts, slot);
            id = info.id;
            playing = info.playing;
            enh = info.enhancement;
            ed = info.edition;
            seal = info.seal;
            flags = info.flags;
        }
        uint32_t p0 = __shfl_sync(PUF_WARP_MASK, (id & 0xFFFF) | (sec << 16) | (playing << 24) | ((local & 0x7F) << 25), 0);
        uint32_t p1 = __shfl_sync(PUF_WARP_MASK, enh | (ed << 8) | (seal << 16) | (flags << 24), 0);
        id = p0 & 0xFFFF;
        sec = (p0 >> 16) & 0xFF;
        playing = (p0 >> 24) & 1;
        local = (p0 >> 25) & 0x7F;
        enh = p1 & 0xFF;
        ed = (p1 >> 8) & 0xFF;
        seal = (p1 >> 16) & 0xFF;
        flags = (p1 >> 24) & 0xFF;
        int rank = playing ? id >> 8 : 0;
        int suit = playing ? id & 0xFF : 0;

        float z = to_float(token_b[lane])
            + to_float(token_embed[(BA_TOKEN_ZONE_ROW + sec) * BA_TOKEN_DIM + lane]);
        if (playing) {
            float r_emb = to_float(token_embed[(BA_TOKEN_RANK_ROW + rank) * BA_TOKEN_DIM + lane]);
            float s_emb = to_float(token_embed[(BA_TOKEN_SUIT_ROW + suit) * BA_TOKEN_DIM + lane]);
            z += r_emb + s_emb + (r_emb * s_emb);
        } else {
            z += to_float(token_embed[(BA_TOKEN_CENTER_ROW + id) * BA_TOKEN_DIM + lane]);
        }
        if (sec == ZONE_HAND) {
            z += to_float(token_embed[
                (BA_TOKEN_HAND_POS_ROW + local) * BA_TOKEN_DIM + lane]);
        } else if (sec == ZONE_JOKER) {
            z += to_float(token_embed[
                (BA_TOKEN_JOKER_POS_ROW + local) * BA_TOKEN_DIM + lane]);
        }
        if (enh > 0 && enh < 9)
            z += to_float(token_embed[(BA_TOKEN_ENH_ROW + enh) * BA_TOKEN_DIM + lane]);
        if (ed > 0 && ed < 5)
            z += to_float(token_embed[(BA_TOKEN_ED_ROW + ed) * BA_TOKEN_DIM + lane]);
        if (seal > 0 && seal < 5)
            z += to_float(token_embed[(BA_TOKEN_SEAL_ROW + seal) * BA_TOKEN_DIM + lane]);
        #pragma unroll
        for (int f = 0; f < BA_CARD_FLAG_ROWS; ++f) {
            if ((flags >> f) & 1)
                z += to_float(token_embed[(BA_TOKEN_FLAG_ROW + f) * BA_TOKEN_DIM + lane]);
        }
        #pragma unroll
        for (int k = 0; k < BA_RAW_DIM; ++k) {
            z += to_float(token_w[lane * BA_RAW_DIM + k]) * s_raw[warp][k];
        }
        float token_value = fmaxf(z, 0.0f);
        if (relu_mask) {
            uint32_t mask = (uint32_t)__ballot_sync(
                PUF_WARP_MASK, token_value > 0.0f);
            if (lane == 0)
                relu_mask[(int64_t)b * BA_KEY_CAP + slot] = (int)mask;
        }
        if (lane < BA_KEY_DIM)
            keys[((int64_t)b * BA_KEY_CAP + slot) * BA_KEY_DIM + lane]
                = from_float(token_value);
        atomicAdd(&s_pool_sum[sec][lane], token_value);
        int max_section = sec == ZONE_HAND ? 0
            : sec == ZONE_JOKER ? 1
            : sec == ZONE_CONSUMABLE ? 2
            : sec == ZONE_SHOP_MAIN ? 3 : -1;
        if (max_section >= 0 && token_value > 0.0f) {
            unsigned long long packed =
                ((unsigned long long)__float_as_uint(token_value) << 32)
                | (unsigned int)(UINT_MAX - slot);
            atomicMax(&s_pool_max[max_section][lane], packed);
        }
        __syncwarp();
    }

    int count_base = offsetof(Observation, counts);
    int special_count = ba_u16(obs, in,
        count_base + offsetof(ObservationCounts, special_count));
    for (int special = warp; special < special_count; special += TOKEN_WARPS) {
        int rank, suit, enh, ed, seal, flags, location;
        if (lane == 0) {
            int offset = offsetof(Observation, specials)
                + special * sizeof(DeckCardView);
            int attributes = ba_u16(obs, in, offset
                + offsetof(DeckCardView, card)
                + offsetof(PlayingCardView, attributes));
            rank = attributes & 15;
            suit = (attributes >> 4) & 3;
            enh = (attributes >> 6) & 15;
            ed = (attributes >> 10) & 7;
            seal = (attributes >> 13) & 7;
            flags = ba_byte(obs, in, offset + offsetof(DeckCardView, card)
                + offsetof(PlayingCardView, flags));
            int perma = (int16_t)ba_u16(obs, in, offset
                + offsetof(DeckCardView, card)
                + offsetof(PlayingCardView, perma_bonus));
            location = ba_byte(obs, in,
                offset + offsetof(DeckCardView, location));
            assert(location < BA_TOKEN_LOCATION_ROWS);
            #pragma unroll
            for (int k = 0; k < BA_RAW_DIM; ++k) s_raw[warp][k] = 0.0f;
            s_raw[warp][0] = (float)rank / 14.0f;
            s_raw[warp][1] = (float)suit / 3.0f;
            s_raw[warp][2] = (float)enh / 8.0f;
            s_raw[warp][3] = (float)ed / 4.0f;
            s_raw[warp][4] = (float)seal / 4.0f;
            s_raw[warp][5] = (float)flags / 127.0f;
            s_raw[warp][7] = 1.0f;
            s_raw[warp][24] = (float)perma / (fabsf((float)perma) + 10.0f);
            s_raw[warp][25] = (float)perma / 100.0f;
        }
        rank = __shfl_sync(PUF_WARP_MASK, rank, 0);
        suit = __shfl_sync(PUF_WARP_MASK, suit, 0);
        enh = __shfl_sync(PUF_WARP_MASK, enh, 0);
        ed = __shfl_sync(PUF_WARP_MASK, ed, 0);
        seal = __shfl_sync(PUF_WARP_MASK, seal, 0);
        flags = __shfl_sync(PUF_WARP_MASK, flags, 0);
        location = __shfl_sync(PUF_WARP_MASK, location, 0);
        float rank_value = to_float(token_embed[
            (BA_TOKEN_RANK_ROW + rank) * BA_TOKEN_DIM + lane]);
        float suit_value = to_float(token_embed[
            (BA_TOKEN_SUIT_ROW + suit) * BA_TOKEN_DIM + lane]);
        float z = to_float(token_b[lane])
            + to_float(token_embed[(BA_TOKEN_ZONE_ROW + BA_SPECIAL_ZONE)
                * BA_TOKEN_DIM + lane])
            + to_float(token_embed[(BA_TOKEN_LOCATION_ROW + location)
                * BA_TOKEN_DIM + lane])
            + rank_value + suit_value + rank_value * suit_value;
        if (enh > 0) z += to_float(token_embed[
            (BA_TOKEN_ENH_ROW + enh) * BA_TOKEN_DIM + lane]);
        if (ed > 0) z += to_float(token_embed[
            (BA_TOKEN_ED_ROW + ed) * BA_TOKEN_DIM + lane]);
        if (seal > 0) z += to_float(token_embed[
            (BA_TOKEN_SEAL_ROW + seal) * BA_TOKEN_DIM + lane]);
        #pragma unroll
        for (int flag = 0; flag < BA_CARD_FLAG_ROWS; ++flag) {
            if ((flags >> flag) & 1) z += to_float(token_embed[
                (BA_TOKEN_FLAG_ROW + flag) * BA_TOKEN_DIM + lane]);
        }
        #pragma unroll
        for (int k = 0; k < BA_RAW_DIM; ++k) {
            z += to_float(token_w[lane * BA_RAW_DIM + k]) * s_raw[warp][k];
        }
        atomicAdd(&s_special_sum[lane], fmaxf(z, 0.0f));
        __syncwarp();
    }
    __syncthreads();

    for (int i = threadIdx.x; i < BA_MAX_POOL_FEATURES; i += blockDim.x) {
        int section = i >> 5;
        int d = i & 31;
        unsigned long long packed = s_pool_max[section][d];
        float value = __uint_as_float((unsigned int)(packed >> 32));
        int slot = packed ? (int)(UINT_MAX - (unsigned int)packed) : -1;
        pooled[b * BA_TOTAL + BA_OFF_MAX_POOL + i] = from_float(value);
        if (pool_argmax)
            pool_argmax[b * BA_MAX_POOL_FEATURES + i] = slot;
    }

    // Mean keys retain content-position interaction because position is mixed
    // before the token nonlinearity.
    for (int c = threadIdx.x; c < BA_POOLED_FEATURES; c += blockDim.x) {
        int sec = c >> 5;
        int d = c & 31;
        int cnt = s_counts[sec];
        float mean = cnt > 0 ? s_pool_sum[sec][d] / (float)cnt : 0.0f;
        pooled[b * BA_TOTAL + BA_OFF_POOLED + c] = from_float(mean);
    }
    if (threadIdx.x < BA_SPECIAL_FEATURES) {
        pooled[b * BA_TOTAL + BA_OFF_SPECIAL + threadIdx.x]
            = from_float(s_special_sum[threadIdx.x] / 16.0f);
    }
    for (int i = threadIdx.x; i < BA_HAND_STRUCT_FEATURES; i += blockDim.x) {
        pooled[b * BA_TOTAL + BA_OFF_HAND_STRUCT + i]
            = from_float((float)s_struct_counts[i] / 8.0f);
    }
    for (int d = threadIdx.x; d < BA_HAND_SUM_FEATURES; d += blockDim.x) {
        pooled[b * BA_TOTAL + BA_OFF_HAND_SUM + d]
            = from_float(s_pool_sum[ZONE_HAND][d] / 8.0f);
    }
}

static constexpr float BA_FXP_SCALE = 16777216.0f;

__device__ __forceinline__ void ba_fxp_atomic_add(long* addr, float value) {
    atomicAdd((unsigned long long*)addr,
        (unsigned long long)(long long)__float2ll_rn(value * BA_FXP_SCALE));
}

static constexpr int ENCODER_STRIPES = 4;

template<int stripes>
__global__ void ba_fxp_to_precision_kernel(
        precision_t* __restrict__ dst, const long* __restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    constexpr int sizes[] = {BA_TOKEN_DIM * BA_RAW_DIM, BA_TOKEN_DIM,
        BA_TOKEN_EMBED_ROWS * BA_TOKEN_DIM, BA_STATE_EMBED_ROWS * BA_STATE_EMBED_DIM};
    int offset = 0;
    for (int size : sizes) {
        if (i < offset + size) {
            long sum = 0;
            for (int stripe = 0; stripe < stripes; ++stripe)
                sum += src[offset * stripes + stripe * size + i - offset];
            dst[i] = from_float((float)((double)sum * (1.0 / 16777216.0)));
            return;
        }
        offset += size;
    }
}

template<int stripes>
__global__ void ba_token_backward_kernel(
        const precision_t* __restrict__ d_pooled,
        const float* __restrict__ key_grad,
        const int* __restrict__ counts_data,
        const int* __restrict__ relu_mask,
        const int* __restrict__ pool_argmax,
        long* __restrict__ token_embed_acc,
        long* __restrict__ state_embed_acc,
        long* __restrict__ token_w_acc,
        long* __restrict__ token_b_acc,
        const precision_t* __restrict__ token_w,
        const precision_t* __restrict__ token_b,
        const precision_t* __restrict__ token_embed,
        const unsigned char* __restrict__ obs,
        int obs_size,
        int B) {
    int stripe = blockIdx.x % stripes;
    token_embed_acc += stripe * BA_TOKEN_EMBED_ROWS * BA_TOKEN_DIM;
    state_embed_acc += stripe * BA_STATE_EMBED_ROWS * BA_STATE_EMBED_DIM;
    token_w_acc += stripe * BA_TOKEN_DIM * BA_RAW_DIM;
    token_b_acc += stripe * BA_TOKEN_DIM;
    constexpr int TOKEN_WARPS = BLOCK_SIZE / 32;
    __shared__ float raw[TOKEN_WARPS][BA_RAW_DIM];
    __shared__ int s_counts[BA_POOL_SECTIONS];
    __shared__ int s_total;
    __shared__ float s_token_b[BA_TOKEN_DIM];
    __shared__ float s_token_w[BA_TOKEN_DIM][BA_RAW_DIM];

    int b = blockIdx.x;
    if (b >= B) return;

    if (threadIdx.x < BA_POOL_SECTIONS) {
        s_counts[threadIdx.x] = counts_data[b * BA_POOL_SECTIONS + threadIdx.x];
    }
    if (threadIdx.x == 0) {
        int tot = 0;
        #pragma unroll
        for (int s = 0; s < BA_POOL_SECTIONS; ++s) tot += counts_data[b * BA_POOL_SECTIONS + s];
        s_total = tot;
    }
    for (int i = threadIdx.x; i < BA_TOKEN_DIM; i += blockDim.x) {
        s_token_b[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < BA_TOKEN_DIM * BA_RAW_DIM; i += blockDim.x) {
        ((float*)s_token_w)[i] = 0.0f;
    }
    __syncthreads();

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x / 32;
    int d = lane;
    int total = s_total;
    int64_t in = (int64_t)b * obs_size;
    int base = offsetof(Observation, globals);

    if (warp == 0) {
        int base = offsetof(Observation, globals);
        int octet = lane >> 3;      // 0, 1, 2, or 3
        int sub_lane = lane & 7;    // 0 .. 7
        if (octet == 0) {
            int deck = ba_byte(obs, in, base + offsetof(ObservationGlobals, deck_id));
            if (deck < 16) {
                float g_deck = to_float(d_pooled[(int64_t)b * BA_TOTAL + BA_OFF_DECK_EMBED + sub_lane]);
                if (g_deck != 0.0f)
                    ba_fxp_atomic_add(&state_embed_acc[(BA_STATE_DECK_ROW + deck) * 8 + sub_lane], g_deck);
            }
            int blind = ba_u16(obs, in, base + offsetof(ObservationGlobals, blind_id));
            if (blind < BLIND_COUNT) {
                float g_blind = to_float(d_pooled[(int64_t)b * BA_TOTAL + BA_OFF_BLIND_EMBED + sub_lane]);
                if (g_blind != 0.0f)
                    ba_fxp_atomic_add(&state_embed_acc[(BA_STATE_BLIND_ROW + blind) * 8 + sub_lane], g_blind);
            }
        } else if (octet == 1) {
            int boss = ba_u16(obs, in, base + offsetof(ObservationGlobals, next_boss_id));
            if (boss < BLIND_COUNT) {
                float g_boss = to_float(d_pooled[(int64_t)b * BA_TOTAL + BA_OFF_BOSS_EMBED + sub_lane]);
                if (g_boss != 0.0f)
                    ba_fxp_atomic_add(&state_embed_acc[(BA_STATE_BLIND_ROW + boss) * 8 + sub_lane], g_boss);
            }
            int voucher = ba_u16(obs, in, base + offsetof(ObservationGlobals, next_voucher_id));
            int v_idx = (voucher >= CENTER_V_ANTIMATTER && voucher <= CENTER_V_WASTEFUL)
                ? (voucher - CENTER_V_ANTIMATTER + 1) : 0;
            if (v_idx < 34) {
                float g_v = to_float(d_pooled[(int64_t)b * BA_TOTAL + BA_OFF_VOUCHER_EMBED + sub_lane]);
                if (g_v != 0.0f)
                    ba_fxp_atomic_add(&state_embed_acc[(BA_STATE_VOUCHER_ROW + v_idx) * 8 + sub_lane], g_v);
            }
        } else if (octet == 2) {
            int tarot = ba_u16(obs, in, base + offsetof(ObservationGlobals, last_tarot_planet));
            if (tarot < CENTER_COUNT) {
                float g_t = to_float(d_pooled[(int64_t)b * BA_TOTAL + BA_OFF_TAROT_EMBED + sub_lane]);
                if (g_t != 0.0f)
                    ba_fxp_atomic_add(&state_embed_acc[(BA_STATE_TAROT_ROW + tarot) * 8 + sub_lane], g_t);
            }
            int lock_raw = ba_byte(obs, in, base + offsetof(ObservationGlobals, blind_only_hand));
            int lock_idx = (lock_raw < HAND_COUNT) ? lock_raw : HAND_COUNT;
            float g_lock = to_float(d_pooled[(int64_t)b * BA_TOTAL + BA_OFF_LOCK_EMBED + sub_lane]);
            if (g_lock != 0.0f)
                ba_fxp_atomic_add(&state_embed_acc[(BA_STATE_LOCK_ROW + lock_idx) * 8 + sub_lane], g_lock);
        } else if (octet == 3) {
            uint16_t hands_mask = ba_u16(obs, in, base + offsetof(ObservationGlobals, blind_hands_mask));
            float g_hands = to_float(d_pooled[(int64_t)b * BA_TOTAL + BA_OFF_PLAYED_HANDS + sub_lane]);
            if (g_hands != 0.0f) {
                uint32_t m = hands_mask & 0x0FFF;
                while (m) {
                    int h = __ffs(m) - 1;
                    m &= m - 1;
                    ba_fxp_atomic_add(&state_embed_acc[(BA_STATE_PLAYED_HAND_ROW + h) * 8 + sub_lane], g_hands);
                }
            }

            uint16_t planet_mask = ba_u16(obs, in, base + offsetof(ObservationGlobals, planet_usage_mask));
            float g_planet = to_float(d_pooled[(int64_t)b * BA_TOTAL + BA_OFF_PLANET_USAGE + sub_lane]);
            if (g_planet != 0.0f) {
                uint32_t m = planet_mask & 0x0FFF;
                while (m) {
                    int h = __ffs(m) - 1;
                    m &= m - 1;
                    ba_fxp_atomic_add(&state_embed_acc[(BA_STATE_PLANET_HAND_ROW + h) * 8 + sub_lane], g_planet);
                }
            }
            if (sub_lane < 4) {
                uint8_t skip_mask = ba_byte(obs, in, base + offsetof(ObservationGlobals, blind_skipped_mask));
                float g_skip = to_float(d_pooled[(int64_t)b * BA_TOTAL + BA_OFF_SKIP_EMBED + sub_lane]);
                if (g_skip != 0.0f) {
                    for (int s = 0; s < 3; ++s) {
                        if ((skip_mask >> s) & 1)
                            ba_fxp_atomic_add(&state_embed_acc[(BA_STATE_SKIP_ROW + s) * 8 + sub_lane], g_skip);
                    }
                }
                int atag = ba_byte(obs, in, base + offsetof(ObservationGlobals, active_tag));
                if (atag < TAG_COUNT) {
                    float g_tag = to_float(d_pooled[(int64_t)b * BA_TOTAL + BA_OFF_TAG_EMBED + sub_lane]);
                    if (g_tag != 0.0f)
                        ba_fxp_atomic_add(&state_embed_acc[(BA_STATE_TAG_ROW + atag) * 8 + sub_lane], g_tag);
                }
                for (int bi = 0; bi < 2; ++bi) {
                    int btag = ba_byte(obs, in, base + offsetof(ObservationGlobals, blind_tags) + bi);
                    if (btag < TAG_COUNT) {
                        float g_btag = to_float(d_pooled[(int64_t)b * BA_TOTAL
                            + BA_OFF_BLIND_TAGS + bi * 4 + sub_lane]);
                        if (g_btag != 0.0f)
                            ba_fxp_atomic_add(&state_embed_acc[(BA_STATE_TAG_ROW + btag) * 8 + sub_lane], g_btag);
                    }
                }
            }
        }
    }

    if (threadIdx.x < 8) {
        int d = threadIdx.x;
        int pending = ba_u16(obs, in,
            base + offsetof(ObservationGlobals, pending_free_pack_id));
        if (pending < CENTER_COUNT) ba_fxp_atomic_add(&state_embed_acc[
            (BA_STATE_TAROT_ROW + pending) * 8 + d], to_float(d_pooled[
                (int64_t)b * BA_TOTAL + BA_OFF_PENDING_EMBED + d]));
        int most = ba_byte(obs, in,
            base + offsetof(ObservationGlobals, most_played_hand));
        if (most >= HAND_COUNT) most = HAND_COUNT;
        ba_fxp_atomic_add(&state_embed_acc[(BA_STATE_LOCK_ROW + most) * 8 + d],
            to_float(d_pooled[(int64_t)b * BA_TOTAL + BA_OFF_MOST_PLAYED + d]));
        int last = ba_byte(obs, in,
            base + offsetof(ObservationGlobals, last_hand_type));
        if (last >= HAND_COUNT) last = HAND_COUNT;
        ba_fxp_atomic_add(&state_embed_acc[(BA_STATE_LOCK_ROW + last) * 8 + d],
            to_float(d_pooled[(int64_t)b * BA_TOTAL + BA_OFF_LAST_HAND + d]));
        static constexpr int offsets[5] = {
            offsetof(ObservationGlobals, ancient_suit),
            offsetof(ObservationGlobals, idol_rank),
            offsetof(ObservationGlobals, idol_suit),
            offsetof(ObservationGlobals, mail_rank),
            offsetof(ObservationGlobals, castle_suit),
        };
        static constexpr int outputs[5] = {
            BA_OFF_ANCIENT_SUIT, BA_OFF_IDOL_RANK, BA_OFF_IDOL_SUIT,
            BA_OFF_MAIL_RANK, BA_OFF_CASTLE_SUIT,
        };
        for (int target = 0; target < 5; ++target) {
            int value = ba_byte(obs, in, base + offsets[target]);
            int row = (target == 0 || target == 2 || target == 4)
                ? BA_STATE_SUIT_ROW + value : BA_STATE_RANK_ROW + value;
            ba_fxp_atomic_add(&state_embed_acc[row * 8 + d], to_float(d_pooled[
                (int64_t)b * BA_TOTAL + outputs[target] + d]));
        }
        for (int item = 0; item < 3; ++item) {
            int hand = ba_byte(obs, in,
                base + offsetof(ObservationGlobals, orbital_hands) + item);
            if (hand >= HAND_COUNT) hand = HAND_COUNT;
            ba_fxp_atomic_add(&state_embed_acc[
                (BA_STATE_LOCK_ROW + hand) * 8 + d], to_float(d_pooled[
                    (int64_t)b * BA_TOTAL + BA_OFF_ORBITAL + item * 8 + d]));
        }
    }

    for (int slot = warp; slot < total; slot += TOKEN_WARPS) {
        int id, zone, is_playing, enh, ed, seal, flags;
        if (lane == 0) {
            ba_token_features(obs, in, s_counts, slot, raw[warp]);
            TokenInfo info = ba_token_info(obs, in, s_counts, slot);
            id = info.id;
            zone = info.zone;
            is_playing = info.playing;
            enh = info.enhancement;
            ed = info.edition;
            seal = info.seal;
            flags = info.flags;
        }
        uint32_t p0 = __shfl_sync(PUF_WARP_MASK, (id & 0xFFFF) | (zone << 16) | (is_playing << 24), 0);
        uint32_t p1 = __shfl_sync(PUF_WARP_MASK, enh | (ed << 8) | (seal << 16) | (flags << 24), 0);
        id = p0 & 0xFFFF;
        zone = (p0 >> 16) & 0xFF;
        is_playing = (p0 >> 24) & 1;
        enh = p1 & 0xFF;
        ed = (p1 >> 8) & 0xFF;
        seal = (p1 >> 16) & 0xFF;
        flags = (p1 >> 24) & 0xFF;

        int local;
        int sec = ba_slot_section(s_counts, slot, &local);
        float g = 0.0f;
        int count = s_counts[sec];
        g += to_float(d_pooled[(int64_t)b * BA_TOTAL + BA_OFF_POOLED
            + sec * BA_TOKEN_DIM + d]) / (float)count;
        if (sec == ZONE_HAND) {
            g += to_float(d_pooled[(int64_t)b * BA_TOTAL + BA_OFF_HAND_SUM + d]) / 8.0f;
        }
        int max_section = sec == ZONE_HAND ? 0
            : sec == ZONE_JOKER ? 1
            : sec == ZONE_CONSUMABLE ? 2
            : sec == ZONE_SHOP_MAIN ? 3 : -1;
        if (max_section >= 0 && pool_argmax[
                b * BA_MAX_POOL_FEATURES + max_section * BA_TOKEN_DIM + d] == slot) {
            g += to_float(d_pooled[(int64_t)b * BA_TOTAL
                + BA_OFF_MAX_POOL + max_section * BA_TOKEN_DIM + d]);
        }

        int rank = is_playing ? id >> 8 : 0;
        int suit = is_playing ? id & 0xFF : 0;
        if (d < AR_EMBED_DIM) {
            int64_t grad_idx = ((int64_t)b * BA_KEY_CAP + slot) * AR_EMBED_DIM + d;
            g += key_grad[grad_idx];
        }
        uint32_t mask = (uint32_t)relu_mask[(int64_t)b * BA_KEY_CAP + slot];
        if (((mask >> d) & 1u) == 0 || g == 0.0f) continue;

        ba_fxp_atomic_add(&token_embed_acc[(BA_TOKEN_ZONE_ROW + zone) * BA_TOKEN_DIM + d], g);
        if (is_playing) {
            float r_emb = to_float(token_embed[(BA_TOKEN_RANK_ROW + rank) * BA_TOKEN_DIM + d]);
            float s_emb = to_float(token_embed[(BA_TOKEN_SUIT_ROW + suit) * BA_TOKEN_DIM + d]);
            ba_fxp_atomic_add(&token_embed_acc[(BA_TOKEN_RANK_ROW + rank) * BA_TOKEN_DIM + d], g * (1.0f + s_emb));
            ba_fxp_atomic_add(&token_embed_acc[(BA_TOKEN_SUIT_ROW + suit) * BA_TOKEN_DIM + d], g * (1.0f + r_emb));
        } else {
            ba_fxp_atomic_add(&token_embed_acc[(BA_TOKEN_CENTER_ROW + id) * BA_TOKEN_DIM + d], g);
        }
        if (sec == ZONE_HAND) {
            ba_fxp_atomic_add(&token_embed_acc[
                (BA_TOKEN_HAND_POS_ROW + local) * BA_TOKEN_DIM + d], g);
        } else if (sec == ZONE_JOKER) {
            ba_fxp_atomic_add(&token_embed_acc[
                (BA_TOKEN_JOKER_POS_ROW + local) * BA_TOKEN_DIM + d], g);
        }
        if (enh > 0 && enh < 9)
            ba_fxp_atomic_add(&token_embed_acc[(BA_TOKEN_ENH_ROW + enh) * BA_TOKEN_DIM + d], g);
        if (ed > 0 && ed < 5)
            ba_fxp_atomic_add(&token_embed_acc[(BA_TOKEN_ED_ROW + ed) * BA_TOKEN_DIM + d], g);
        if (seal > 0 && seal < 5)
            ba_fxp_atomic_add(&token_embed_acc[(BA_TOKEN_SEAL_ROW + seal) * BA_TOKEN_DIM + d], g);
        #pragma unroll
        for (int f = 0; f < BA_CARD_FLAG_ROWS; ++f) {
            if ((flags >> f) & 1)
                ba_fxp_atomic_add(&token_embed_acc[(BA_TOKEN_FLAG_ROW + f) * BA_TOKEN_DIM + d], g);
        }
        atomicAdd(&s_token_b[d], g);
        #pragma unroll
        for (int k = 0; k < BA_RAW_DIM; ++k) {
            float vk = raw[warp][k];
            if (vk != 0.0f)
                atomicAdd(&s_token_w[d][k], g * vk);
        }
    }

    int count_base = offsetof(Observation, counts);
    int special_count = ba_u16(obs, in,
        count_base + offsetof(ObservationCounts, special_count));
    for (int special = warp; special < special_count; special += TOKEN_WARPS) {
        int rank, suit, enh, ed, seal, flags, location;
        if (lane == 0) {
            int offset = offsetof(Observation, specials)
                + special * sizeof(DeckCardView);
            int attributes = ba_u16(obs, in, offset
                + offsetof(DeckCardView, card)
                + offsetof(PlayingCardView, attributes));
            rank = attributes & 15;
            suit = (attributes >> 4) & 3;
            enh = (attributes >> 6) & 15;
            ed = (attributes >> 10) & 7;
            seal = (attributes >> 13) & 7;
            flags = ba_byte(obs, in, offset + offsetof(DeckCardView, card)
                + offsetof(PlayingCardView, flags));
            int perma = (int16_t)ba_u16(obs, in, offset
                + offsetof(DeckCardView, card)
                + offsetof(PlayingCardView, perma_bonus));
            location = ba_byte(obs, in,
                offset + offsetof(DeckCardView, location));
            assert(location < BA_TOKEN_LOCATION_ROWS);
            #pragma unroll
            for (int k = 0; k < BA_RAW_DIM; ++k) raw[warp][k] = 0.0f;
            raw[warp][0] = (float)rank / 14.0f;
            raw[warp][1] = (float)suit / 3.0f;
            raw[warp][2] = (float)enh / 8.0f;
            raw[warp][3] = (float)ed / 4.0f;
            raw[warp][4] = (float)seal / 4.0f;
            raw[warp][5] = (float)flags / 127.0f;
            raw[warp][7] = 1.0f;
            raw[warp][24] = (float)perma / (fabsf((float)perma) + 10.0f);
            raw[warp][25] = (float)perma / 100.0f;
        }
        rank = __shfl_sync(PUF_WARP_MASK, rank, 0);
        suit = __shfl_sync(PUF_WARP_MASK, suit, 0);
        enh = __shfl_sync(PUF_WARP_MASK, enh, 0);
        ed = __shfl_sync(PUF_WARP_MASK, ed, 0);
        seal = __shfl_sync(PUF_WARP_MASK, seal, 0);
        flags = __shfl_sync(PUF_WARP_MASK, flags, 0);
        location = __shfl_sync(PUF_WARP_MASK, location, 0);
        float rank_value = to_float(token_embed[
            (BA_TOKEN_RANK_ROW + rank) * BA_TOKEN_DIM + d]);
        float suit_value = to_float(token_embed[
            (BA_TOKEN_SUIT_ROW + suit) * BA_TOKEN_DIM + d]);
        float z = to_float(token_b[d]) + to_float(token_embed[
            (BA_TOKEN_ZONE_ROW + BA_SPECIAL_ZONE) * BA_TOKEN_DIM + d])
            + to_float(token_embed[(BA_TOKEN_LOCATION_ROW + location)
                * BA_TOKEN_DIM + d])
            + rank_value + suit_value + rank_value * suit_value;
        if (enh > 0) z += to_float(token_embed[
            (BA_TOKEN_ENH_ROW + enh) * BA_TOKEN_DIM + d]);
        if (ed > 0) z += to_float(token_embed[
            (BA_TOKEN_ED_ROW + ed) * BA_TOKEN_DIM + d]);
        if (seal > 0) z += to_float(token_embed[
            (BA_TOKEN_SEAL_ROW + seal) * BA_TOKEN_DIM + d]);
        #pragma unroll
        for (int flag = 0; flag < BA_CARD_FLAG_ROWS; ++flag) {
            if ((flags >> flag) & 1) z += to_float(token_embed[
                (BA_TOKEN_FLAG_ROW + flag) * BA_TOKEN_DIM + d]);
        }
        #pragma unroll
        for (int k = 0; k < BA_RAW_DIM; ++k) {
            z += to_float(token_w[d * BA_RAW_DIM + k]) * raw[warp][k];
        }
        float g = to_float(d_pooled[(int64_t)b * BA_TOTAL
            + BA_OFF_SPECIAL + d]) / 16.0f;
        if (z <= 0.0f || g == 0.0f) continue;
        ba_fxp_atomic_add(&token_embed_acc[
            (BA_TOKEN_ZONE_ROW + BA_SPECIAL_ZONE) * BA_TOKEN_DIM + d], g);
        ba_fxp_atomic_add(&token_embed_acc[
            (BA_TOKEN_LOCATION_ROW + location) * BA_TOKEN_DIM + d], g);
        ba_fxp_atomic_add(&token_embed_acc[
            (BA_TOKEN_RANK_ROW + rank) * BA_TOKEN_DIM + d],
            g * (1.0f + suit_value));
        ba_fxp_atomic_add(&token_embed_acc[
            (BA_TOKEN_SUIT_ROW + suit) * BA_TOKEN_DIM + d],
            g * (1.0f + rank_value));
        if (enh > 0) ba_fxp_atomic_add(&token_embed_acc[
            (BA_TOKEN_ENH_ROW + enh) * BA_TOKEN_DIM + d], g);
        if (ed > 0) ba_fxp_atomic_add(&token_embed_acc[
            (BA_TOKEN_ED_ROW + ed) * BA_TOKEN_DIM + d], g);
        if (seal > 0) ba_fxp_atomic_add(&token_embed_acc[
            (BA_TOKEN_SEAL_ROW + seal) * BA_TOKEN_DIM + d], g);
        #pragma unroll
        for (int flag = 0; flag < BA_CARD_FLAG_ROWS; ++flag) {
            if ((flags >> flag) & 1) ba_fxp_atomic_add(&token_embed_acc[
                (BA_TOKEN_FLAG_ROW + flag) * BA_TOKEN_DIM + d], g);
        }
        atomicAdd(&s_token_b[d], g);
        #pragma unroll
        for (int k = 0; k < BA_RAW_DIM; ++k) {
            if (raw[warp][k] != 0.0f) {
                atomicAdd(&s_token_w[d][k], g * raw[warp][k]);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < BA_TOKEN_DIM) {
        float gb = s_token_b[threadIdx.x];
        if (gb != 0.0f)
            ba_fxp_atomic_add(&token_b_acc[threadIdx.x], gb);
    }
    for (int i = threadIdx.x; i < BA_TOKEN_DIM * BA_RAW_DIM; i += blockDim.x) {
        int d_idx = i / BA_RAW_DIM;
        int k_idx = i % BA_RAW_DIM;
        float gw = s_token_w[d_idx][k_idx];
        if (gw != 0.0f)
            ba_fxp_atomic_add(&token_w_acc[i], gw);
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
    encode_globals<<<B, BLOCK_SIZE, 0, stream>>>(
        a->pooled.data, ew->state_embed.data, input.data, B, ew->obs_size);
    encode_tokens<64><<<B, 64, 0, stream>>>(
        a->pooled.data, a->keys.data, a->counts.data,
        a->relu_mask.data, a->pool_argmax.data,
        ew->token_w.data, ew->token_b.data,
        ew->token_embed.data,
        input.data, B, ew->obs_size);
    puf_mm(&a->pooled, &ew->proj_w, &a->out, stream);
    return a->out;
}

static void ba_encoder_backward(
        void* w, void* activations, Prec grad, cudaStream_t stream) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    BalatroEncoderActivations* a = (BalatroEncoderActivations*)activations;
    int B = a->obs_batch;
    puf_mm_tn(&grad, &a->pooled, &a->proj_wgrad, stream);
    // Static count/signature columns have no encoder parameters, so only the
    // embedding and token slices require projection gradients.
    puf_mm_nn_slice(&grad, &ew->proj_w, &a->d_pooled, 0, BA_OFF_SCALARS, stream);
    puf_mm_nn_slice(&grad, &ew->proj_w, &a->d_pooled,
        BA_OFF_POOLED, BA_POOLED_FEATURES + BA_MAX_POOL_FEATURES
            + BA_SPECIAL_FEATURES + BA_HAND_STRUCT_FEATURES + BA_HAND_SUM_FEATURES, stream);
    constexpr int ACC_TOTAL_ELEMS = BA_TOKEN_DIM * BA_RAW_DIM + BA_TOKEN_DIM
        + BA_TOKEN_EMBED_ROWS * BA_TOKEN_DIM + BA_STATE_EMBED_ROWS * BA_STATE_EMBED_DIM;
    cudaMemsetAsync(a->token_w_acc.data, 0, ENCODER_STRIPES * ACC_TOTAL_ELEMS * sizeof(long), stream);
    ba_token_backward_kernel<ENCODER_STRIPES><<<B, BLOCK_SIZE, 0, stream>>>(
        a->d_pooled.data,
        a->key_grad.data, a->counts.data,
        a->relu_mask.data, a->pool_argmax.data,
        a->token_embed_acc.data, a->state_embed_acc.data,
        a->token_w_acc.data, a->token_b_acc.data,
        ew->token_w.data, ew->token_b.data, ew->token_embed.data,
        a->obs_data, ew->obs_size, B);
    ba_fxp_to_precision_kernel<ENCODER_STRIPES><<<grid_size(ACC_TOTAL_ELEMS), BLOCK_SIZE, 0, stream>>>(
        a->token_wgrad.data, a->token_w_acc.data, ACC_TOTAL_ELEMS);
}

static void ba_encoder_init_weights(
        void* w, uint64_t* seed, cudaStream_t stream) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    puf_kaiming_init(&ew->token_w, 1.0f, (*seed)++, stream);
    puf_normal_init(&ew->token_b, 0.01f, (*seed)++, stream);
    puf_normal_init(&ew->token_embed, 0.02f, (*seed)++, stream);
    puf_normal_init(&ew->state_embed, 0.02f, (*seed)++, stream);
    puf_kaiming_init(&ew->proj_w, sqrtf(2.0f), (*seed)++, stream);
}

static void ba_encoder_reg_params(void* w, Allocator* alloc) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    ew->token_w = {.shape = {BA_TOKEN_DIM, BA_RAW_DIM}};
    ew->token_b = {.shape = {BA_TOKEN_DIM}};
    ew->token_embed = {.shape = {BA_TOKEN_EMBED_ROWS, BA_TOKEN_DIM}};
    ew->state_embed = {.shape = {BA_STATE_EMBED_ROWS, BA_STATE_EMBED_DIM}};
    ew->proj_w = {.shape = {ew->hidden, BA_TOTAL}};
    alloc_register(alloc, &ew->token_w);
    alloc_register(alloc, &ew->token_b);
    alloc_register(alloc, &ew->token_embed);
    alloc_register(alloc, &ew->state_embed);
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
    a->keys = {.shape = {B_TT, BA_KEY_CAP, BA_KEY_DIM}};
    a->key_grad = {.shape = {B_TT, BA_KEY_CAP, AR_EMBED_DIM}};
    a->counts = {.shape = {B_TT, BA_POOL_SECTIONS}};
    a->relu_mask = {.shape = {B_TT, BA_KEY_CAP}};
    a->pool_argmax = {.shape = {B_TT, BA_MAX_POOL_SECTIONS, BA_TOKEN_DIM}};
    a->token_embed_acc = {.shape = {ENCODER_STRIPES, BA_TOKEN_EMBED_ROWS, BA_TOKEN_DIM}};
    a->state_embed_acc = {.shape = {ENCODER_STRIPES, BA_STATE_EMBED_ROWS, BA_STATE_EMBED_DIM}};
    a->token_w_acc = {.shape = {ENCODER_STRIPES, BA_TOKEN_DIM, BA_RAW_DIM}};
    a->token_b_acc = {.shape = {ENCODER_STRIPES, BA_TOKEN_DIM}};
    a->token_wgrad = {.shape = {BA_TOKEN_DIM, BA_RAW_DIM}};
    a->token_bgrad = {.shape = {BA_TOKEN_DIM}};
    a->token_embed_grad = {.shape = {BA_TOKEN_EMBED_ROWS, BA_TOKEN_DIM}};
    a->state_embed_grad = {.shape = {BA_STATE_EMBED_ROWS, BA_STATE_EMBED_DIM}};
    a->proj_wgrad = {.shape = {ew->hidden, BA_TOTAL}};
    alloc_register(acts, &a->pooled);
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->d_pooled);
    alloc_register(acts, &a->keys);
    alloc_register(acts, &a->key_grad);
    alloc_register(acts, &a->counts);
    alloc_register(acts, &a->relu_mask);
    alloc_register(acts, &a->pool_argmax);
    alloc_register(acts, &a->token_w_acc);
    alloc_register(acts, &a->token_b_acc);
    alloc_register(acts, &a->token_embed_acc);
    alloc_register(acts, &a->state_embed_acc);
    alloc_register(grads, &a->token_wgrad);
    alloc_register(grads, &a->token_bgrad);
    alloc_register(grads, &a->token_embed_grad);
    alloc_register(grads, &a->state_embed_grad);
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
    a->keys = {.shape = {B, BA_KEY_CAP, BA_KEY_DIM}};
    a->counts = {.shape = {B, BA_POOL_SECTIONS}};
    alloc_register(alloc, &a->pooled);
    alloc_register(alloc, &a->out);
    alloc_register(alloc, &a->keys);
    alloc_register(alloc, &a->counts);
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

#ifdef USE_ROCM
#include "decoder.hip"
#else
#include "decoder.cu"
#endif
