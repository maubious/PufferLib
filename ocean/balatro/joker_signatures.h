#ifndef JOKER_SIGNATURES_H
#define JOKER_SIGNATURES_H

#include <stdint.h>
#include "balatro_core.h"

typedef enum SigTrigger {
    SIG_TRIG_SCORED_CARD = 0, /* per scoring card */
    SIG_TRIG_HAND_PLAYED,     /* once per played hand */
    SIG_TRIG_HELD_CARD,       /* per held card */
    SIG_TRIG_ROUND_END,       /* economy phase */
    SIG_TRIG_DISCARD,         /* per discard action */
    SIG_TRIG_REROLL,          /* per shop reroll */
    SIG_TRIG_SKIP_PACK,       /* per skipped booster pack */
} SigTrigger;

typedef enum SigCond {
    SIG_COND_NONE = 0,
    SIG_COND_SUIT,        /* cond_value = suit (HEARTS=0..SPADES=3) */
    SIG_COND_RANK,        /* cond_value = rank 2..14 */
    SIG_COND_RANK_SET,    /* rank_mask bits */
    SIG_COND_HAND_TYPE,   /* cond_value = HandType */
    SIG_COND_ENHANCEMENT, /* cond_value = Enhancement */
    SIG_COND_FACE,        /* rank 11-13 */
    SIG_COND_LAST_HAND,   /* final hand of the blind */
    SIG_COND_NO_DISCARDS, /* 0 discards remaining */
    SIG_COND_MAX3,        /* <=3 cards played */
    SIG_COND_REPEAT_HAND, /* hand type already played this round */
} SigCond;

typedef enum SigScale {
    SIG_SCALE_NONE = 0,
    SIG_SCALE_JOKERS,      /* per owned joker */
    SIG_SCALE_UNCOMMON,    /* per uncommon-rarity joker */
    SIG_SCALE_DECK,        /* per card in deck+hand+discard */
    SIG_SCALE_DECK_LOST,   /* per card below starting count (40/52) */
    SIG_SCALE_DISCARD_LEFT,
    SIG_SCALE_DOLLARS,
    SIG_SCALE_DOLLARS_5,   /* per $5 held */
    SIG_SCALE_STONES,
    SIG_SCALE_STEELS,
    SIG_SCALE_NINES,       /* per 9 in deck+hand+discard */
    SIG_SCALE_SELL_TOTAL,  /* per $1 of other jokers' sell value */
    SIG_SCALE_PLANETS,     /* per planet used */
    SIG_SCALE_TAROTS,      /* per tarot used */
    SIG_SCALE_HAND_PLAYS,  /* per total plays of the current hand type */
    SIG_SCALE_SKIPS,       /* per skipped blind */
    SIG_SCALE_INTEREST,    /* per interest bracket */
    SIG_SCALE_FREE_SLOTS,  /* per empty joker slot */
} SigScale;

typedef enum SigRule {
    SIG_RULE_NONE = 0,
    SIG_RULE_FOUR_FINGERS, /* flushes/straights with 4 cards */
    SIG_RULE_SHORTCUT,     /* straights may skip ranks */
    SIG_RULE_SMEARED,      /* suits count as same color */
    SIG_RULE_PAREIDOLIA,   /* all cards count as face */
    SIG_RULE_SPLASH,       /* every played card scores */
} SigRule;

typedef struct JokerSignature {
    uint8_t trigger;
    uint8_t condition;
    uint8_t cond_value;   /* suit / rank / hand type / enhancement */
    uint16_t rank_mask;   /* SIG_COND_RANK_SET */
    int16_t chips;
    int16_t mult;
    uint16_t xmult_pct;   /* 100 = x1.0 */
    int16_t dollars;
    uint8_t scale;
    int8_t growth;
    uint8_t chance_pct;
    uint8_t retriggers;
    int8_t levels;
    int8_t hand_size;
    int8_t discards;
    int8_t hands;
    int8_t sell_growth;
    uint8_t rule;
    uint8_t opaque;
    uint8_t nondeterministic;
} JokerSignature;

#ifdef __cplusplus
extern "C" const JokerSignature JOKER_SIGNATURES[CENTER_COUNT];
#else
const JokerSignature JOKER_SIGNATURES[CENTER_COUNT] = {
    /* Per-scored-card, suit-conditioned */
    [CENTER_J_GREEDY_JOKER]    = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_SUIT, .cond_value=DIAMONDS, .mult=3},
    [CENTER_J_LUSTY_JOKER]     = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_SUIT, .cond_value=HEARTS, .mult=3},
    [CENTER_J_WRATHFUL_JOKER]  = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_SUIT, .cond_value=SPADES, .mult=3},
    [CENTER_J_GLUTTENOUS_JOKER]= {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_SUIT, .cond_value=CLUBS, .mult=3},
    [CENTER_J_ARROWHEAD]       = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_SUIT, .cond_value=SPADES, .chips=50},
    [CENTER_J_ONYX_AGATE]      = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_SUIT, .cond_value=CLUBS, .mult=7},
    [CENTER_J_ROUGH_GEM]       = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_SUIT, .cond_value=DIAMONDS, .dollars=1},
    [CENTER_J_BLOODSTONE]      = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_SUIT, .cond_value=HEARTS, .xmult_pct=150, .chance_pct=50, .nondeterministic=1},

    /* Per-scored-card, rank-conditioned */
    [CENTER_J_SCHOLAR]         = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_RANK, .cond_value=14, .chips=20, .mult=4},
    [CENTER_J_WALKIE_TALKIE]   = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_RANK_SET, .rank_mask=(1u<<2)|(1u<<8), .chips=10, .mult=4}, /* 10,4 */
    [CENTER_J_EVEN_STEVEN]     = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_RANK_SET, .rank_mask=0x155, .mult=4}, /* 2,4,6,8,10 */
    [CENTER_J_ODD_TODD]        = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_RANK_SET, .rank_mask=0x10AA, .chips=31}, /* 3,5,7,9,A */
    [CENTER_J_FIBONACCI]       = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_RANK_SET, .rank_mask=0x104B, .mult=8}, /* 2,3,5,8,A */
    [CENTER_J_TRIBOULET]       = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_RANK_SET, .rank_mask=0xC00, .xmult_pct=200}, /* K,Q */
    [CENTER_J_WEE]             = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_RANK, .cond_value=2, .chips=8, .growth=8},
    [CENTER_J_HACK]            = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_RANK_SET, .rank_mask=0xF, .retriggers=1}, /* 2-5 */
    [CENTER_J_SCARY_FACE]      = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_FACE, .chips=30},
    [CENTER_J_SMILEY]          = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_FACE, .mult=5},
    [CENTER_J_SOCK_AND_BUSKIN] = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_FACE, .retriggers=1},
    [CENTER_J_BUSINESS]        = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_FACE, .dollars=2, .chance_pct=50, .nondeterministic=1},
    [CENTER_J_HIKER]           = {.trigger=SIG_TRIG_SCORED_CARD, .chips=5}, /* permanent perma_bonus */
    [CENTER_J_TICKET]          = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_ENHANCEMENT, .cond_value=ENHANCEMENT_GOLD, .dollars=4},

    /* Retriggers */
    [CENTER_J_HANGING_CHAD]    = {.trigger=SIG_TRIG_SCORED_CARD, .retriggers=2}, /* first card only */
    [CENTER_J_DUSK]            = {.trigger=SIG_TRIG_SCORED_CARD, .condition=SIG_COND_LAST_HAND, .retriggers=1},
    [CENTER_J_SELZER]          = {.trigger=SIG_TRIG_SCORED_CARD, .retriggers=1},

    /* Held card */
    [CENTER_J_BARON]           = {.trigger=SIG_TRIG_HELD_CARD, .condition=SIG_COND_RANK, .cond_value=13, .xmult_pct=150},
    [CENTER_J_SHOOT_THE_MOON]  = {.trigger=SIG_TRIG_HELD_CARD, .condition=SIG_COND_RANK, .cond_value=12, .mult=13},
    [CENTER_J_RESERVED_PARKING]= {.trigger=SIG_TRIG_HELD_CARD, .condition=SIG_COND_FACE, .dollars=1, .chance_pct=50, .nondeterministic=1},
    [CENTER_J_RAISED_FIST]     = {.trigger=SIG_TRIG_HELD_CARD, .mult=11}, /* 2x lowest held rank, 4..22 */
    [CENTER_J_MIME]            = {.trigger=SIG_TRIG_HELD_CARD, .retriggers=1},

    /* Flat / per-hand */
    [CENTER_J_JOKER]           = {.trigger=SIG_TRIG_HAND_PLAYED, .mult=4},
    [CENTER_J_STUNTMAN]        = {.trigger=SIG_TRIG_HAND_PLAYED, .chips=250, .hand_size=-2},
    [CENTER_J_GROS_MICHEL]     = {.trigger=SIG_TRIG_HAND_PLAYED, .mult=15}, /* 1/6 destroy */
    [CENTER_J_CAVENDISH]       = {.trigger=SIG_TRIG_HAND_PLAYED, .xmult_pct=300}, /* 1/1000 destroy */
    [CENTER_J_HALF]            = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_MAX3, .mult=20},
    [CENTER_J_MYSTIC_SUMMIT]   = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_NO_DISCARDS, .mult=15},
    [CENTER_J_ACROBAT]         = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_LAST_HAND, .xmult_pct=300},
    [CENTER_J_CARD_SHARP]      = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_REPEAT_HAND, .xmult_pct=300},
    [CENTER_J_PHOTOGRAPH]      = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_FACE, .xmult_pct=200}, /* first face only */
    [CENTER_J_SPACE]           = {.trigger=SIG_TRIG_HAND_PLAYED, .chance_pct=25, .levels=1, .nondeterministic=1},

    /* Hand-type-conditioned */
    [CENTER_J_JOLLY]           = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_HAND_TYPE, .cond_value=PAIR, .mult=8},
    [CENTER_J_ZANY]            = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_HAND_TYPE, .cond_value=THREE_OF_A_KIND, .mult=12},
    [CENTER_J_MAD]             = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_HAND_TYPE, .cond_value=TWO_PAIR, .mult=10},
    [CENTER_J_CRAZY]           = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_HAND_TYPE, .cond_value=STRAIGHT, .mult=12},
    [CENTER_J_DROLL]           = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_HAND_TYPE, .cond_value=FLUSH, .mult=10},
    [CENTER_J_SLY]             = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_HAND_TYPE, .cond_value=PAIR, .chips=50},
    [CENTER_J_WILY]            = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_HAND_TYPE, .cond_value=THREE_OF_A_KIND, .chips=100},
    [CENTER_J_CLEVER]          = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_HAND_TYPE, .cond_value=TWO_PAIR, .chips=80},
    [CENTER_J_DEVIOUS]         = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_HAND_TYPE, .cond_value=STRAIGHT, .chips=100},
    [CENTER_J_CRAFTY]          = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_HAND_TYPE, .cond_value=FLUSH, .chips=80},
    [CENTER_J_DUO]             = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_HAND_TYPE, .cond_value=PAIR, .xmult_pct=200},
    [CENTER_J_TRIO]            = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_HAND_TYPE, .cond_value=THREE_OF_A_KIND, .xmult_pct=300},
    [CENTER_J_FAMILY]          = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_HAND_TYPE, .cond_value=FOUR_OF_A_KIND, .xmult_pct=400},
    [CENTER_J_ORDER]           = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_HAND_TYPE, .cond_value=STRAIGHT, .xmult_pct=300},
    [CENTER_J_TRIBE]           = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_HAND_TYPE, .cond_value=FLUSH, .xmult_pct=200},

    /* State-scaled (flat trigger, magnitude per state unit) */
    [CENTER_J_ABSTRACT]        = {.trigger=SIG_TRIG_HAND_PLAYED, .mult=3, .scale=SIG_SCALE_JOKERS},
    [CENTER_J_BANNER]          = {.trigger=SIG_TRIG_HAND_PLAYED, .chips=30, .scale=SIG_SCALE_DISCARD_LEFT},
    [CENTER_J_BLUE_JOKER]      = {.trigger=SIG_TRIG_HAND_PLAYED, .chips=2, .scale=SIG_SCALE_DECK},
    [CENTER_J_BULL]            = {.trigger=SIG_TRIG_HAND_PLAYED, .chips=2, .scale=SIG_SCALE_DOLLARS},
    [CENTER_J_BOOTSTRAPS]      = {.trigger=SIG_TRIG_HAND_PLAYED, .mult=2, .scale=SIG_SCALE_DOLLARS_5},
    [CENTER_J_EROSION]         = {.trigger=SIG_TRIG_HAND_PLAYED, .mult=4, .scale=SIG_SCALE_DECK_LOST},
    [CENTER_J_STONE]           = {.trigger=SIG_TRIG_HAND_PLAYED, .chips=25, .scale=SIG_SCALE_STONES},
    [CENTER_J_STEEL_JOKER]     = {.trigger=SIG_TRIG_HAND_PLAYED, .xmult_pct=120, .scale=SIG_SCALE_STEELS},
    [CENTER_J_SUPERNOVA]       = {.trigger=SIG_TRIG_HAND_PLAYED, .mult=1, .scale=SIG_SCALE_HAND_PLAYS},
    [CENTER_J_FORTUNE_TELLER]  = {.trigger=SIG_TRIG_HAND_PLAYED, .mult=1, .scale=SIG_SCALE_TAROTS},
    [CENTER_J_SWASHBUCKLER]    = {.trigger=SIG_TRIG_HAND_PLAYED, .mult=1, .scale=SIG_SCALE_SELL_TOTAL},
    [CENTER_J_THROWBACK]       = {.trigger=SIG_TRIG_HAND_PLAYED, .xmult_pct=125, .scale=SIG_SCALE_SKIPS},
    [CENTER_J_STENCIL]         = {.trigger=SIG_TRIG_HAND_PLAYED, .xmult_pct=200, .scale=SIG_SCALE_FREE_SLOTS},

    /* Stateful growth (magnitude in state[0], listed = start, growth = per trigger) */
    [CENTER_J_GREEN_JOKER]     = {.trigger=SIG_TRIG_HAND_PLAYED, .mult=1, .growth=1}, /* -1 per discard */
    [CENTER_J_RIDE_THE_BUS]    = {.trigger=SIG_TRIG_HAND_PLAYED, .mult=1, .growth=1}, /* reset on face */
    [CENTER_J_TROUSERS]        = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_HAND_TYPE, .cond_value=TWO_PAIR, .mult=2, .growth=2},
    [CENTER_J_SQUARE]          = {.trigger=SIG_TRIG_HAND_PLAYED, .chips=4, .growth=4}, /* 4-card hands */
    [CENTER_J_RUNNER]          = {.trigger=SIG_TRIG_HAND_PLAYED, .condition=SIG_COND_HAND_TYPE, .cond_value=STRAIGHT, .chips=15, .growth=15},
    [CENTER_J_FLASH]           = {.trigger=SIG_TRIG_REROLL, .mult=2, .growth=2},
    [CENTER_J_RED_CARD]        = {.trigger=SIG_TRIG_SKIP_PACK, .mult=3, .growth=3},
    [CENTER_J_POPCORN]         = {.trigger=SIG_TRIG_HAND_PLAYED, .mult=20, .growth=-4},
    [CENTER_J_ICE_CREAM]       = {.trigger=SIG_TRIG_HAND_PLAYED, .chips=100, .growth=-5},
    [CENTER_J_RAMEN]           = {.trigger=SIG_TRIG_HAND_PLAYED, .xmult_pct=200, .growth=-1}, /* decays per discard */

    /* Economy (round end) */
    [CENTER_J_GOLDEN]          = {.trigger=SIG_TRIG_ROUND_END, .dollars=4},
    [CENTER_J_ROCKET]          = {.trigger=SIG_TRIG_ROUND_END, .dollars=1, .growth=2},
    [CENTER_J_CLOUD_9]         = {.trigger=SIG_TRIG_ROUND_END, .dollars=1, .scale=SIG_SCALE_NINES},
    [CENTER_J_DELAYED_GRAT]    = {.trigger=SIG_TRIG_ROUND_END, .dollars=2, .scale=SIG_SCALE_DISCARD_LEFT}, /* only if none used */
    [CENTER_J_SATELLITE]       = {.trigger=SIG_TRIG_ROUND_END, .dollars=1, .scale=SIG_SCALE_PLANETS},
    [CENTER_J_TO_THE_MOON]     = {.trigger=SIG_TRIG_ROUND_END, .dollars=1, .scale=SIG_SCALE_INTEREST},
    [CENTER_J_EGG]             = {.trigger=SIG_TRIG_ROUND_END, .sell_growth=3},
    [CENTER_J_GIFT]            = {.trigger=SIG_TRIG_ROUND_END, .sell_growth=1}, /* all cards */

    /* Discard-triggered */
    [CENTER_J_BURNT]           = {.trigger=SIG_TRIG_DISCARD, .levels=1}, /* first discard only */
    [CENTER_J_TRADING]         = {.trigger=SIG_TRIG_DISCARD, .dollars=3}, /* first single discard, destroys card */
    [CENTER_J_FACELESS]        = {.trigger=SIG_TRIG_DISCARD, .condition=SIG_COND_FACE, .dollars=5}, /* >=3 faces */
    [CENTER_J_HIT_THE_ROAD]    = {.trigger=SIG_TRIG_DISCARD, .condition=SIG_COND_RANK, .cond_value=11, .xmult_pct=150}, /* +0.5x per J */

    /* Logistics */
    [CENTER_J_JUGGLER]         = {.hand_size=1},
    [CENTER_J_MERRY_ANDY]      = {.hand_size=-1, .discards=3},
    [CENTER_J_DRUNKARD]        = {.discards=1},
    [CENTER_J_TROUBADOUR]      = {.hand_size=2, .hands=-1},
    [CENTER_J_BURGLAR]         = {.hands=3},
    [CENTER_J_TURTLE_BEAN]     = {.growth=-1, .hand_size=5},

    /* Hand rules */
    [CENTER_J_FOUR_FINGERS]    = {.rule=SIG_RULE_FOUR_FINGERS},
    [CENTER_J_SHORTCUT]        = {.rule=SIG_RULE_SHORTCUT},
    [CENTER_J_SMEARED]         = {.rule=SIG_RULE_SMEARED},
    [CENTER_J_PAREIDOLIA]      = {.rule=SIG_RULE_PAREIDOLIA},
    [CENTER_J_SPLASH]          = {.rule=SIG_RULE_SPLASH},

    /* Opaque: build-dependent or complex */
    [CENTER_J_8_BALL]          = {.opaque=1}, /* 25% tarot on 8 */
    [CENTER_J_ANCIENT]         = {.opaque=1}, /* x1.5 rotating suit */
    [CENTER_J_ASTRONOMER]      = {.opaque=1}, /* free planets */
    [CENTER_J_BASEBALL]        = {.opaque=1}, /* x1.5 per uncommon */
    [CENTER_J_BLACKBOARD]      = {.opaque=1}, /* x3 all held black */
    [CENTER_J_BLUEPRINT]       = {.opaque=1}, /* copies right joker */
    [CENTER_J_BRAINSTORM]      = {.opaque=1}, /* copies leftmost joker */
    [CENTER_J_CAINO]           = {.opaque=1}, /* +1x per face destroyed */
    [CENTER_J_CAMPFIRE]        = {.opaque=1}, /* +0.5x per sold, resets at boss */
    [CENTER_J_CARTOMANCER]     = {.opaque=1}, /* tarot per round */
    [CENTER_J_CASTLE]          = {.opaque=1}, /* +3 chips per suited discard */
    [CENTER_J_CEREMONIAL]      = {.opaque=1}, /* destroys right joker */
    [CENTER_J_CERTIFICATE]     = {.opaque=1}, /* card per round */
    [CENTER_J_CHAOS]           = {.opaque=1}, /* free rerolls */
    [CENTER_J_CHICOT]          = {.opaque=1}, /* disables boss */
    [CENTER_J_CONSTELLATION]   = {.opaque=1}, /* +0.1x per planet */
    [CENTER_J_CREDIT_CARD]     = {.opaque=1}, /* -$20 floor */
    [CENTER_J_DIET_COLA]       = {.opaque=1}, /* sell: double tag */
    [CENTER_J_DNA]             = {.opaque=1}, /* copy first single card */
    [CENTER_J_DRIVERS_LICENSE] = {.opaque=1}, /* x3 at 16 enhanced */
    [CENTER_J_FLOWER_POT]      = {.opaque=1}, /* x3 all suits */
    [CENTER_J_GLASS]           = {.opaque=1}, /* +0.75x per glass */
    [CENTER_J_HALLUCINATION]   = {.opaque=1}, /* tarot on pack open */
    [CENTER_J_HOLOGRAM]        = {.opaque=1}, /* +0.25x per card added */
    [CENTER_J_IDOL]            = {.opaque=1}, /* copies another idol */
    [CENTER_J_INVISIBLE]       = {.opaque=1}, /* sell: duplicate */
    [CENTER_J_LOYALTY_CARD]    = {.opaque=1}, /* x4 every 5th hand */
    [CENTER_J_LUCHADOR]        = {.opaque=1}, /* sell: disable boss */
    [CENTER_J_LUCKY_CAT]       = {.opaque=1}, /* +0.25x per lucky proc */
    [CENTER_J_MADNESS]         = {.opaque=1}, /* +0.5x per blind, destroys */
    [CENTER_J_MAIL]            = {.opaque=1}, /* $5 per rank discard */
    [CENTER_J_MARBLE]          = {.opaque=1}, /* stone per round */
    [CENTER_J_MATADOR]         = {.opaque=1}, /* $8 per boss debuff */
    [CENTER_J_MIDAS_MASK]      = {.opaque=1}, /* faces become gold */
    [CENTER_J_MISPRINT]        = {.opaque=1}, /* random +0..23 mult */
    [CENTER_J_MR_BONES]        = {.opaque=1}, /* survive at 25% */
    [CENTER_J_OBELISK]         = {.opaque=1}, /* +0.2x per other hand */
    [CENTER_J_OOPS]            = {.opaque=1}, /* 2x probabilities */
    [CENTER_J_PERKEO]          = {.opaque=1}, /* negative consumable copy */
    [CENTER_J_RIFF_RAFF]       = {.opaque=1}, /* 2 commons per round */
    [CENTER_J_RING_MASTER]     = {.opaque=1}, /* showman rerolls */
    [CENTER_J_SEANCE]          = {.opaque=1}, /* spectral per straight flush */
    [CENTER_J_SEEING_DOUBLE]   = {.opaque=1}, /* x2 club+other */
    [CENTER_J_SIXTH_SENSE]     = {.opaque=1}, /* spectral per 6 */
    [CENTER_J_SUPERPOSITION]   = {.opaque=1}, /* tarot per A+straight */
    [CENTER_J_TODO_LIST]       = {.opaque=1}, /* $4 per target hand */
    [CENTER_J_VAGABOND]        = {.opaque=1}, /* tarot when <=$4 */
    [CENTER_J_VAMPIRE]         = {.opaque=1}, /* eats enhancements */
    [CENTER_J_YORICK]          = {.opaque=1}, /* +1x per 23 discards */
};
#endif

#endif
