#pragma once
#ifndef BALATRO_CORE_H
#define BALATRO_CORE_H

#include <stddef.h>
#include <stdint.h>
#include "balatro_ids.h"

#ifdef __cplusplus
extern "C" {
#endif


#define MAX_DECK 256
#define MAX_HAND 64
#define MAX_JOKERS 32
#define MAX_CONSUMABLES 32
#define MAX_SHOP_MAIN 4
#define MAX_SHOP_VOUCHERS 64
#define MAX_SHOP_BOOSTERS 2
#define MAX_PACK_CARDS 5
#define MAX_SELECTION 5
#define MAX_RNG_STREAMS 256
#define ACTION_TYPE_COUNT 23

typedef enum Error {
    OK = 0,
    ERR_ARGUMENT = -1,
    ERR_ACTION = -2,
    ERR_CAPACITY = -3,
    ERR_SNAPSHOT = -4,
    ERR_INVARIANT = -6
} Error;

typedef enum Phase {
    PHASE_BLIND_SELECT = 0,
    PHASE_SELECTING_HAND = 1,
    PHASE_ROUND_EVAL = 2,
    PHASE_SHOP = 3,
    PHASE_PACK_OPENING = 4,
    PHASE_GAME_OVER = 5
} Phase;

typedef enum ActionType {
    ACTION_PLAY_HAND = 0,
    ACTION_DISCARD = 1,
    ACTION_SELECT_BLIND = 2,
    ACTION_SKIP_BLIND = 3,
    ACTION_CASH_OUT = 4,
    ACTION_REROLL = 5,
    ACTION_NEXT_ROUND = 6,
    ACTION_SKIP_PACK = 7,
    ACTION_BUY_CARD = 8,
    ACTION_SELL_JOKER = 9,
    ACTION_SELL_CONSUMABLE = 10,
    ACTION_USE_CONSUMABLE = 11,
    ACTION_REDEEM_VOUCHER = 12,
    ACTION_OPEN_BOOSTER = 13,
    ACTION_PICK_PACK_CARD = 14,
    ACTION_SWAP_JOKERS_LEFT = 15,
    ACTION_SWAP_JOKERS_RIGHT = 16,
    ACTION_SWAP_HAND_LEFT = 17,
    ACTION_SWAP_HAND_RIGHT = 18,
    ACTION_SORT_HAND_RANK = 19,
    ACTION_SORT_HAND_SUIT = 20,
    ACTION_BUY_AND_USE = 21,
    ACTION_REROLL_BOSS = 22
} ActionType;

typedef enum Suit { HEARTS = 0, DIAMONDS = 1, CLUBS = 2, SPADES = 3 } Suit;

typedef enum Enhancement {
    ENHANCEMENT_NONE = 0,
    ENHANCEMENT_BONUS = 1,
    ENHANCEMENT_MULT = 2,
    ENHANCEMENT_WILD = 3,
    ENHANCEMENT_GLASS = 4,
    ENHANCEMENT_STEEL = 5,
    ENHANCEMENT_STONE = 6,
    ENHANCEMENT_GOLD = 7,
    ENHANCEMENT_LUCKY = 8
} Enhancement;

typedef enum Edition {
    EDITION_NONE = 0,
    EDITION_FOIL = 1,
    EDITION_HOLO = 2,
    EDITION_POLYCHROME = 3,
    EDITION_NEGATIVE = 4
} Edition;

typedef enum Seal {
    SEAL_NONE = 0,
    SEAL_GOLD = 1,
    SEAL_RED = 2,
    SEAL_BLUE = 3,
    SEAL_PURPLE = 4
} Seal;

typedef enum TargetEffect {
    TARGET_CUSTOM,
    TARGET_ENHANCEMENT,
    TARGET_SUIT,
    TARGET_RANK_UP,
    TARGET_SEAL,
} TargetEffect;

typedef enum CardSet {
    SET_DEFAULT = 1,
    SET_PLAYING = 1,
    SET_ENHANCED = 2,
    SET_JOKER = 3,
    SET_TAROT = 4,
    SET_PLANET = 5,
    SET_SPECTRAL = 6,
    SET_VOUCHER = 7,
    SET_BOOSTER = 8,
} CardSet;

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

extern const CenterDefinition centers[CENTER_COUNT];

typedef enum CardFlag {
    CARD_DEBUFFED = 1u << 0,
    CARD_ETERNAL = 1u << 1,
    CARD_PERISHABLE = 1u << 2,
    CARD_RENTAL = 1u << 3,
    CARD_FORCED = 1u << 4,
    CARD_FACEDOWN = 1u << 5
} CardFlag;

typedef enum HandType {
    FLUSH_FIVE = 0,
    FLUSH_HOUSE,
    FIVE_OF_A_KIND,
    STRAIGHT_FLUSH,
    FOUR_OF_A_KIND,
    FULL_HOUSE,
    FLUSH,
    STRAIGHT,
    THREE_OF_A_KIND,
    TWO_PAIR,
    PAIR,
    HIGH_CARD,
    HAND_COUNT
} HandType;

typedef enum PackKind {
    PACK_STANDARD = 0,
    PACK_TAROT = 1,
    PACK_PLANET = 2,
    PACK_SPECTRAL = 3,
    PACK_JOKER = 5
} PackKind;

typedef struct Card {
    uint16_t center_id;
    uint16_t sort_id;
    uint8_t suit;
    uint8_t rank;
    uint8_t enhancement;
    uint8_t edition;
    uint8_t seal;
    uint8_t flags;
    int16_t perma_bonus;
    int16_t cost;
    int16_t sell_cost;
    int32_t state[4];
} Card;

typedef struct Action {
    uint8_t type;
    uint8_t primary;
    uint8_t selection_count;
    uint8_t selection[MAX_SELECTION];
} Action;

/* Reward shaping constants. Per-step reward stays inside [-1, 1]. */
#define WIN_BONUS 1.0
#define WEALTH_NORM_DIVISOR 25.0
#define WEALTH_NORM 2.1972245773362196 /* log1p(200 / 25): wealth potential saturates slowly past $200. */

typedef struct Config {
    uint8_t deck;
    uint8_t stake;
    uint8_t win_ante;
    uint8_t shaped_reward;
    uint8_t validation;
    uint8_t fast_rng;
    float progress_reward; /* Close-out k: pays (scored/blind)^2 per scoring hand. */
    float contact_reward; /* Linear contact: pays scored/blind per scoring hand. Split-neutral
                             (fixed total per blind) and paid even in failed blinds, so there is
                             always gradient toward scoring. Kept tiny next to clear bonuses. */
    float blind_bonus; /* Small/big share of the clear pool; drives skip-neutral boss math. */
    float ante_bonus; /* Base clear unit B; per-ante pool scales as B * escalation^(ante-1). */
    float ante_escalation; /* Geometric multiplier per ante, at least 1. */
    float efficiency_hand; /* Bonus per unused hand left on blind clear. */
    float efficiency_discard; /* Bonus per unused discard left on blind clear. */
    float wealth_weight; /* Weight of the dollars+invested potential shaping term. */
} Config;

typedef struct RngStream {
    uint64_t key_hash;
    double value;
} RngStream;

typedef struct State {
    Config config;
    uint64_t numeric_seed;
    char seed[32];
    double hashed_seed;
    RngStream rng[MAX_RNG_STREAMS];
    uint16_t rng_count;
    uint16_t next_sort_id;

    uint8_t phase;
    uint8_t blind_on_deck;
    uint8_t ante;
    uint8_t round;
    uint8_t won;
    uint8_t terminal;
    uint8_t hands_left;
    uint8_t discards_left;
    uint8_t hands_played;
    uint8_t discards_used;
    uint8_t hand_size;
    uint8_t joker_slots;
    uint8_t consumable_slots;
    uint8_t skips;
    uint8_t blind_skipped_mask;

    uint8_t hand_sort_suit;

    int32_t dollars;

    double chips;
    double blind_chips;
    double last_hand_score;
    double last_hand_chips;
    double last_hand_mult;
    double peak_hand_log;
    int32_t reroll_cost;
    int32_t round_earnings;
    int16_t interest_cap;
    int8_t interest_amount;
    int8_t rental_rate;
    int8_t blind_reward;
    uint8_t stake_scaling;
    uint32_t actions_taken;

    int32_t invested;
    uint32_t run_hands_played;

    Card deck[MAX_DECK];
    Card hand[MAX_HAND];
    Card discard[MAX_DECK];
    Card jokers[MAX_JOKERS];
    Card consumables[MAX_CONSUMABLES];
    Card shop_main[MAX_SHOP_MAIN];
    Card shop_vouchers[MAX_SHOP_VOUCHERS];
    Card shop_boosters[MAX_SHOP_BOOSTERS];
    Card pack_cards[MAX_PACK_CARDS];
    uint16_t deck_count;
    uint8_t hand_count;
    uint16_t discard_count;
    uint8_t joker_count;
    uint8_t consumable_count;
    uint8_t shop_main_count;
    uint8_t shop_voucher_count;
    uint8_t shop_booster_count;
    uint8_t pack_count;

    uint8_t hand_levels[HAND_COUNT];
    uint8_t used_centers[(CENTER_COUNT + 7) / 8];
    uint8_t reroll_base;
    uint8_t reroll_increase;
    uint8_t free_rerolls;
    uint8_t first_shop_buffoon;
    uint8_t shop_joker_max;
    uint8_t hands_per_round;
    uint8_t discards_per_round;
    uint8_t discount_percent;
    float joker_rate;
    float tarot_rate;
    float planet_rate;
    float spectral_rate;
    float playing_card_rate;
    float edition_rate;
    uint8_t pack_choices;
    uint8_t pack_kind;
    uint8_t shop_return_phase;

    uint16_t pending_free_pack_id;
    uint16_t hand_plays[HAND_COUNT];
    uint8_t hand_plays_round[HAND_COUNT];
    uint16_t tarots_used;
    uint16_t planet_usage_mask;
    uint8_t ancient_suit;

    uint8_t idol_rank;
    uint8_t idol_suit;
    uint8_t mail_rank;
    uint8_t castle_suit;
    uint16_t blind_id;
    uint8_t blind_disabled;
    uint8_t blind_only_hand;
    uint16_t blind_hands_mask;

    uint8_t most_played_hand;
    uint8_t last_hand_type;
    uint8_t base_hand_size;
    uint16_t next_boss_id;
    uint8_t boss_rerolled;
    uint8_t boss_usage[BLIND_COUNT];
    uint8_t double_tag;
    uint8_t blind_tags[2];
    uint8_t orbital_hands[3];
    uint8_t active_tag;
    uint16_t unused_discards;
    uint8_t tag_hand_bonus;
    uint8_t tag_force_rarity;
    uint8_t tag_force_rarity_count;
    uint8_t tag_force_edition;
    uint8_t tag_force_edition_count;
    uint8_t tag_voucher_pending;
    uint8_t tag_coupon_pending;
    uint8_t tag_coupon_active;
    uint8_t tag_saved_discount;
    uint8_t tag_investment_pending;
    uint8_t tag_d_six_pending;
    uint8_t tag_d_six_active;
    uint8_t ecto_penalty;
    uint16_t next_voucher_id;
    uint8_t gros_michel_extinct;
    uint16_t last_tarot_planet;
    uint64_t joker_flags;
    uint64_t joker_active_flags;
} State;

typedef enum ItemZone {
    ZONE_HAND = 0,
    ZONE_JOKER = 1,
    ZONE_CONSUMABLE = 2,
    ZONE_SHOP_MAIN = 3,
    ZONE_SHOP_VOUCHER = 4,
    ZONE_SHOP_BOOSTER = 5,
    ZONE_PACK_CARD = 6
} ItemZone;

typedef struct SelectionContract {
    uint64_t allowed_hand;
    uint64_t required_hand;
    uint8_t minimum;
    uint8_t maximum;
    uint8_t valid;
    uint8_t reserved;
} SelectionContract;

typedef struct LegalMasks {
    uint64_t primary[ACTION_TYPE_COUNT];
    SelectionContract play;
    SelectionContract discard;
    SelectionContract consumable[MAX_CONSUMABLES];
    SelectionContract shop[MAX_SHOP_MAIN];
    SelectionContract pack[MAX_PACK_CARDS];
} LegalMasks;

/* Selection entry for a (type, primary) pair, or NULL when the action
   carries no selection (BUY_CARD, swaps, …). */
static inline const SelectionContract *cached_selection(
        const LegalMasks *legal, uint8_t type, uint8_t primary) {
    if (type == ACTION_PLAY_HAND) return &legal->play;
    if (type == ACTION_DISCARD) return &legal->discard;
    if (type == ACTION_USE_CONSUMABLE && primary < MAX_CONSUMABLES)
        return &legal->consumable[primary];
    if (type == ACTION_BUY_AND_USE && primary < MAX_SHOP_MAIN)
        return &legal->shop[primary];
    if (type == ACTION_PICK_PACK_CARD && primary < MAX_PACK_CARDS)
        return &legal->pack[primary];
    return NULL;
}

#pragma pack(push, 1)
enum {
    PUBLIC_CARD_DEBUFFED = 1u << 0,
    PUBLIC_CARD_ETERNAL = 1u << 1,
    PUBLIC_CARD_PERISHABLE = 1u << 2,
    PUBLIC_CARD_RENTAL = 1u << 3,
    PUBLIC_CARD_FORCED = 1u << 4,
    PUBLIC_CARD_FACEDOWN = 1u << 5,
    PUBLIC_CARD_PLAYED_THIS_ANTE = 1u << 6,
};

typedef struct PlayingCardView {
    uint16_t attributes;
    uint8_t flags;
    int16_t perma_bonus;
} PlayingCardView;

typedef struct DeckCardView {
    PlayingCardView card;
    uint8_t location;
} DeckCardView;

typedef struct JokerView {
    uint16_t center_id;
    uint8_t edition;
    uint8_t flags;
    int16_t cost;
    int16_t sell_cost;
    int32_t state[4];
} JokerView;

typedef struct ConsumableView {
    uint16_t center_id;
    uint8_t edition;
    uint8_t flags;
    int16_t cost;
    int16_t sell_cost;
} ConsumableView;

typedef struct OfferView {
    uint16_t center_id;
    uint8_t rank;
    uint8_t suit;
    uint8_t enhancement;
    uint8_t edition;
    uint8_t seal;
    uint8_t flags;
    int16_t perma_bonus;
    int16_t cost;
    int16_t sell_cost;
    int32_t state[4];
} OfferView;

typedef struct MarketCardView {
    uint16_t center_id;
    int16_t cost;
    int16_t sell_cost;
} MarketCardView;

typedef struct PlainDeckView {
    uint16_t unseen;
    uint16_t discard;
    uint16_t played_unseen;
    uint16_t played_discard;
} PlainDeckView;

typedef struct ObservationGlobals {
    uint16_t blind_id;
    uint16_t next_boss_id;
    uint16_t next_voucher_id;
    uint16_t pending_free_pack_id;
    uint16_t last_tarot_planet;
    uint64_t redeemed_vouchers_mask;

    float joker_rate;
    float tarot_rate;
    float planet_rate;
    float spectral_rate;
    float playing_card_rate;
    float edition_rate;

    int32_t dollars;
    int32_t reroll_cost;
    int32_t round_earnings;
    uint32_t run_hands_played;

    uint16_t unused_discards;
    uint16_t tarots_used;
    uint16_t planet_usage_mask;
    uint16_t blind_hands_mask;
    int16_t interest_cap;
    int8_t interest_amount;
    int8_t rental_rate;
    int8_t blind_reward;

    uint8_t  deck_id;
    uint8_t  stake;
    uint8_t  win_ante;
    uint8_t  stake_scaling;
    uint8_t  phase;
    uint8_t  blind_on_deck;
    uint8_t  ante;
    uint8_t  round;
    uint8_t  blind_disabled;
    uint8_t  blind_only_hand;
    uint8_t  blind_skipped_mask;
    uint8_t  hand_sort_suit;
    uint8_t  boss_rerolled;
    uint8_t  free_rerolls;
    uint8_t  reroll_base;
    uint8_t  reroll_increase;
    uint8_t  discount_percent;
    uint8_t  hands_per_round;
    uint8_t  discards_per_round;
    uint8_t  base_hand_size;
    uint8_t  hand_size;
    uint8_t  joker_slots;
    uint8_t  consumable_slots;
    uint8_t  skips;
    uint8_t  pack_choices;
    uint8_t  pack_kind;
    uint8_t  shop_return_phase;
    uint8_t  first_shop_buffoon;
    uint8_t  shop_joker_max;
    uint8_t  ancient_suit;
    uint8_t  idol_rank;
    uint8_t  idol_suit;
    uint8_t  mail_rank;
    uint8_t  castle_suit;
    uint8_t  most_played_hand;
    uint8_t  last_hand_type;
    uint8_t  double_tag;
    uint8_t  active_tag;
    uint8_t  blind_tags[2];
    uint8_t  orbital_hands[3];
    uint8_t  tag_hand_bonus;
    uint8_t  tag_force_rarity;
    uint8_t  tag_force_rarity_count;
    uint8_t  tag_force_edition;
    uint8_t  tag_force_edition_count;
    uint8_t  tag_voucher_pending;
    uint8_t  tag_coupon_pending;
    uint8_t  tag_coupon_active;
    uint8_t  tag_saved_discount;
    uint8_t  tag_investment_pending;
    uint8_t  tag_d_six_pending;
    uint8_t  tag_d_six_active;
    uint8_t  ecto_penalty;
    uint8_t  gros_michel_extinct;
    uint8_t  hands_left;
    uint8_t  discards_left;
    uint8_t  hands_played;
    uint8_t  discards_used;

    float score_features[6];
} ObservationGlobals;

typedef struct PokerHandStat {
    uint8_t  level;
    uint8_t  visible;
    uint16_t total_plays;
    uint8_t  round_plays;
    float    chips_log2;
    float    mult_log2;
} PokerHandStat;

typedef struct ObservationCounts {
    uint16_t deck_count;
    uint16_t discard_count;
    uint16_t special_count;
    uint16_t total_playing_cards;
    uint8_t hand_count;
    uint8_t joker_count;
    uint8_t consumable_count;
    uint8_t shop_count;
    uint8_t voucher_count;
    uint8_t booster_count;
    uint8_t pack_count;
} ObservationCounts;

typedef struct MarketView {
    OfferView shop[MAX_SHOP_MAIN];
    MarketCardView vouchers[MAX_SHOP_VOUCHERS];
    MarketCardView boosters[MAX_SHOP_BOOSTERS];
    OfferView pack[MAX_PACK_CARDS];
} MarketView;

typedef struct Observation {
    ObservationGlobals globals;
    PlainDeckView plain_deck[13][4];
    PokerHandStat poker_hands[HAND_COUNT];
    ObservationCounts counts;
    DeckCardView specials[MAX_DECK];
    JokerView jokers[MAX_JOKERS];
    ConsumableView consumables[MAX_CONSUMABLES];
    PlayingCardView hand[MAX_HAND];
    MarketView market;
    uint8_t alignment[7];
} Observation;
#pragma pack(pop)

typedef struct StepResult {
    float reward;
    float sparse_reward;
    uint8_t terminal;
    uint8_t won;
    uint8_t ante;
} StepResult;

void default_config(Config *config);
int init(State *state, const Config *config, uint64_t seed);
int apply_step(State *state, const Action *action, const LegalMasks *masks, StepResult *out);
int can_afford(const State *state, int32_t cost);
HandType planet_hand(uint16_t center_id);

int observe(const State *state, Observation *out, LegalMasks *legal);
uint64_t state_hash(const State *state);
HandType classify_hand(const Card *cards, size_t count, uint8_t *scoring_mask,
    int four_fingers, int shortcut, int smeared);
void hand_base_stats(HandType hand, int level, int *out_chips, int *out_mult);
int joker_active(const State *state, uint16_t center_id);

#ifdef __cplusplus
}
#endif

#endif /* BALATRO_CORE_H */
