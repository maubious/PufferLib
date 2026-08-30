#ifndef CONSUMABLE_SIGNATURES_H
#define CONSUMABLE_SIGNATURES_H

#include <stdint.h>
#include "balatro_core.h"

typedef enum ConsumableKind {
    CONS_NONE = 0,
    CONS_PLANET,          /* Level up planet_hand */
    CONS_BLACK_HOLE,      /* Level up all hands */
    CONS_TARGET,          /* Target effect (enhancement, suit, rank up, seal) on selected cards */
    CONS_SPAWN_POOL,      /* Emperor (Tarots), High Priestess (Planets) */
    CONS_FOOL,            /* Recreate last used tarot/planet */
    CONS_HERMIT,          /* Gain min(dollars, 20) */
    CONS_TEMPERANCE,      /* Gain sum(joker sell cost) up to 50 */
    CONS_WHEEL,           /* 1 in 4 chance for foil/holo/poly on editionless joker */
    CONS_AURA,            /* Roll foil/holo/poly on 1 selected card */
    CONS_HANGED_MAN,      /* Destroy up to 2 selected cards */
    CONS_DEATH,           /* Copy card 1 to card 0 */
    CONS_JUDGEMENT,       /* Spawn random joker */
    CONS_WRAITH,          /* Spawn rare joker, set dollars = 0 */
    CONS_SOUL,            /* Spawn legendary joker */
    CONS_SIGIL,           /* Convert all hand cards to 1 random suit */
    CONS_OUIJA,           /* Convert all hand cards to 1 random rank, hand size - 1 */
    CONS_ECTOPLASM,       /* Add negative to editionless joker, reduce hand size */
    CONS_HEX,             /* Add polychrome to editionless joker, destroy others */
    CONS_IMMOLATE,        /* Destroy 5 random cards, gain $20 */
    CONS_ANKH,            /* Duplicate 1 random joker, destroy others */
    CONS_SPECTRAL_SPAWN,  /* Familiar, Grim, Incantation: destroy 1 random, spawn cards */
    CONS_CRYPTID,         /* Copy 1 card twice */
} ConsumableKind;

typedef struct ConsumableSignature {
    uint8_t kind;
    uint8_t target_effect;  /* TARGET_ENHANCEMENT, TARGET_SUIT, TARGET_RANK_UP, TARGET_SEAL */
    uint8_t target_val;     /* ENHANCEMENT_*, SUIT_*, SEAL_* */
    uint8_t min_select;
    uint8_t max_select;
    uint8_t planet_hand;    /* HandType for planets */
    uint8_t spawn_set;      /* SET_TAROT, SET_PLANET */
    uint8_t spawn_count;
    uint8_t spectral_ranks; /* 1 = face, 2 = ace, 3 = numbers */
    const char *stream;
} ConsumableSignature;

static const ConsumableSignature CONSUMABLE_SIGNATURES[CENTER_COUNT] = {
    /* Planets */
    [CENTER_C_MERCURY]        = {.kind = CONS_PLANET, .planet_hand = PAIR},
    [CENTER_C_VENUS]          = {.kind = CONS_PLANET, .planet_hand = THREE_OF_A_KIND},
    [CENTER_C_EARTH]          = {.kind = CONS_PLANET, .planet_hand = FULL_HOUSE},
    [CENTER_C_MARS]           = {.kind = CONS_PLANET, .planet_hand = FOUR_OF_A_KIND},
    [CENTER_C_JUPITER]        = {.kind = CONS_PLANET, .planet_hand = FLUSH},
    [CENTER_C_SATURN]         = {.kind = CONS_PLANET, .planet_hand = STRAIGHT},
    [CENTER_C_URANUS]         = {.kind = CONS_PLANET, .planet_hand = TWO_PAIR},
    [CENTER_C_NEPTUNE]        = {.kind = CONS_PLANET, .planet_hand = STRAIGHT_FLUSH},
    [CENTER_C_PLUTO]          = {.kind = CONS_PLANET, .planet_hand = HIGH_CARD},
    [CENTER_C_PLANET_X]       = {.kind = CONS_PLANET, .planet_hand = FIVE_OF_A_KIND},
    [CENTER_C_CERES]          = {.kind = CONS_PLANET, .planet_hand = FLUSH_HOUSE},
    [CENTER_C_ERIS]           = {.kind = CONS_PLANET, .planet_hand = FLUSH_FIVE},
    [CENTER_C_BLACK_HOLE]     = {.kind = CONS_BLACK_HOLE},

    /* Tarots */
    [CENTER_C_FOOL]           = {.kind = CONS_FOOL},
    [CENTER_C_MAGICIAN]       = {.kind = CONS_TARGET, .target_effect = TARGET_ENHANCEMENT, .target_val = ENHANCEMENT_LUCKY, .min_select = 1, .max_select = 2},
    [CENTER_C_HIGH_PRIESTESS] = {.kind = CONS_SPAWN_POOL, .spawn_set = SET_PLANET, .spawn_count = 2, .stream = "pri"},
    [CENTER_C_EMPRESS]        = {.kind = CONS_TARGET, .target_effect = TARGET_ENHANCEMENT, .target_val = ENHANCEMENT_MULT, .min_select = 1, .max_select = 2},
    [CENTER_C_EMPEROR]        = {.kind = CONS_SPAWN_POOL, .spawn_set = SET_TAROT, .spawn_count = 2, .stream = "emp"},
    [CENTER_C_HEIROPHANT]     = {.kind = CONS_TARGET, .target_effect = TARGET_ENHANCEMENT, .target_val = ENHANCEMENT_BONUS, .min_select = 1, .max_select = 2},
    [CENTER_C_LOVERS]         = {.kind = CONS_TARGET, .target_effect = TARGET_ENHANCEMENT, .target_val = ENHANCEMENT_WILD, .min_select = 1, .max_select = 1},
    [CENTER_C_CHARIOT]        = {.kind = CONS_TARGET, .target_effect = TARGET_ENHANCEMENT, .target_val = ENHANCEMENT_STEEL, .min_select = 1, .max_select = 1},
    [CENTER_C_JUSTICE]        = {.kind = CONS_TARGET, .target_effect = TARGET_ENHANCEMENT, .target_val = ENHANCEMENT_GLASS, .min_select = 1, .max_select = 1},
    [CENTER_C_HERMIT]         = {.kind = CONS_HERMIT},
    [CENTER_C_WHEEL_OF_FORTUNE] = {.kind = CONS_WHEEL},
    [CENTER_C_STRENGTH]       = {.kind = CONS_TARGET, .target_effect = TARGET_RANK_UP, .min_select = 1, .max_select = 2},
    [CENTER_C_HANGED_MAN]     = {.kind = CONS_HANGED_MAN, .min_select = 1, .max_select = 2},
    [CENTER_C_DEATH]          = {.kind = CONS_DEATH, .min_select = 2, .max_select = 2},
    [CENTER_C_TEMPERANCE]     = {.kind = CONS_TEMPERANCE},
    [CENTER_C_DEVIL]          = {.kind = CONS_TARGET, .target_effect = TARGET_ENHANCEMENT, .target_val = ENHANCEMENT_GOLD, .min_select = 1, .max_select = 1},
    [CENTER_C_TOWER]          = {.kind = CONS_TARGET, .target_effect = TARGET_ENHANCEMENT, .target_val = ENHANCEMENT_STONE, .min_select = 1, .max_select = 1},
    [CENTER_C_STAR]           = {.kind = CONS_TARGET, .target_effect = TARGET_SUIT, .target_val = DIAMONDS, .min_select = 1, .max_select = 3},
    [CENTER_C_MOON]           = {.kind = CONS_TARGET, .target_effect = TARGET_SUIT, .target_val = CLUBS, .min_select = 1, .max_select = 3},
    [CENTER_C_SUN]            = {.kind = CONS_TARGET, .target_effect = TARGET_SUIT, .target_val = HEARTS, .min_select = 1, .max_select = 3},
    [CENTER_C_JUDGEMENT]      = {.kind = CONS_JUDGEMENT},
    [CENTER_C_WORLD]          = {.kind = CONS_TARGET, .target_effect = TARGET_SUIT, .target_val = SPADES, .min_select = 1, .max_select = 3},

    /* Spectrals */
    [CENTER_C_FAMILIAR]       = {.kind = CONS_SPECTRAL_SPAWN, .spectral_ranks = 1, .spawn_count = 3, .stream = "familiar_create"},
    [CENTER_C_GRIM]           = {.kind = CONS_SPECTRAL_SPAWN, .spectral_ranks = 2, .spawn_count = 2, .stream = "grim_create"},
    [CENTER_C_INCANTATION]    = {.kind = CONS_SPECTRAL_SPAWN, .spectral_ranks = 3, .spawn_count = 4, .stream = "incantation_create"},
    [CENTER_C_TALISMAN]       = {.kind = CONS_TARGET, .target_effect = TARGET_SEAL, .target_val = SEAL_GOLD, .min_select = 1, .max_select = 1},
    [CENTER_C_AURA]           = {.kind = CONS_AURA, .min_select = 1, .max_select = 1},
    [CENTER_C_WRAITH]         = {.kind = CONS_WRAITH},
    [CENTER_C_SIGIL]          = {.kind = CONS_SIGIL},
    [CENTER_C_OUIJA]          = {.kind = CONS_OUIJA},
    [CENTER_C_ECTOPLASM]      = {.kind = CONS_ECTOPLASM},
    [CENTER_C_IMMOLATE]       = {.kind = CONS_IMMOLATE},
    [CENTER_C_ANKH]           = {.kind = CONS_ANKH},
    [CENTER_C_DEJA_VU]        = {.kind = CONS_TARGET, .target_effect = TARGET_SEAL, .target_val = SEAL_RED, .min_select = 1, .max_select = 1},
    [CENTER_C_HEX]            = {.kind = CONS_HEX},
    [CENTER_C_TRANCE]         = {.kind = CONS_TARGET, .target_effect = TARGET_SEAL, .target_val = SEAL_BLUE, .min_select = 1, .max_select = 1},
    [CENTER_C_MEDIUM]         = {.kind = CONS_TARGET, .target_effect = TARGET_SEAL, .target_val = SEAL_PURPLE, .min_select = 1, .max_select = 1},
    [CENTER_C_CRYPTID]        = {.kind = CONS_CRYPTID, .min_select = 1, .max_select = 1},
    [CENTER_C_SOUL]           = {.kind = CONS_SOUL},
};

#endif /* CONSUMABLE_SIGNATURES_H */
