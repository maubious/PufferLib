#ifndef CONSUMABLE_SIGNATURES_H
#define CONSUMABLE_SIGNATURES_H

#include <stdint.h>

#include "balatro_core.h"

typedef enum ConsumableKind {
    CONS_NONE,
    CONS_PLANET,
    CONS_BLACK_HOLE,
    CONS_TARGET,
    CONS_SPAWN_POOL,
    CONS_FOOL,
    CONS_HERMIT,
    CONS_TEMPERANCE,
    CONS_JOKER_EDITION,
    CONS_AURA,
    CONS_HANGED_MAN,
    CONS_DEATH,
    CONS_JUDGEMENT,
    CONS_JOKER_RARITY,
    CONS_SIGIL,
    CONS_OUIJA,
    CONS_IMMOLATE,
    CONS_ANKH,
    CONS_SPECTRAL_SPAWN,
    CONS_CRYPTID,
} ConsumableKind;

typedef enum ConsumableUseRule {
    CONSUMABLE_USE_ALWAYS,
    CONSUMABLE_USE_HAND_GT_ONE,
    CONSUMABLE_USE_JOKER_ROOM,
    CONSUMABLE_USE_JOKER_PRESENT_AND_ROOM,
    CONSUMABLE_USE_EDITIONLESS_JOKER,
    CONSUMABLE_USE_ACTIVE_EDITIONLESS_JOKER,
    CONSUMABLE_USE_LAST_TAROT,
} ConsumableUseRule;

typedef enum ConsumableTargetFilter {
    CONSUMABLE_TARGET_ALL_HAND,
    CONSUMABLE_TARGET_EDITIONLESS,
} ConsumableTargetFilter;

typedef enum ConsumableJokerEffect {
    JOKER_ROLL_EDITION,
    JOKER_NEGATIVE_SHRINK_HAND,
    JOKER_POLYCHROME_DESTROY,
} ConsumableJokerEffect;

typedef struct ConsumableSignature {
    uint8_t kind;
    uint8_t use_rule;
    uint8_t target_effect;
    uint8_t target_value;
    uint8_t target_min;
    uint8_t target_max;
    uint8_t target_filter;
    uint8_t planet_hand;
    uint8_t spawn_set;
    uint8_t spawn_count;
    uint8_t spectral_ranks;
    uint8_t joker_effect;
    uint8_t chance_percent;
    uint8_t rarity;
    uint8_t legendary;
    uint8_t reset_dollars;
    const char *stream;
} ConsumableSignature;

#ifdef __cplusplus
extern "C" const ConsumableSignature CONSUMABLE_SIGNATURES[CENTER_COUNT];
#else
const ConsumableSignature CONSUMABLE_SIGNATURES[CENTER_COUNT] = {
    [CENTER_C_MERCURY] = {
        .kind = CONS_PLANET,
        .planet_hand = PAIR,
    },
    [CENTER_C_VENUS] = {
        .kind = CONS_PLANET,
        .planet_hand = THREE_OF_A_KIND,
    },
    [CENTER_C_EARTH] = {
        .kind = CONS_PLANET,
        .planet_hand = FULL_HOUSE,
    },
    [CENTER_C_MARS] = {
        .kind = CONS_PLANET,
        .planet_hand = FOUR_OF_A_KIND,
    },
    [CENTER_C_JUPITER] = {
        .kind = CONS_PLANET,
        .planet_hand = FLUSH,
    },
    [CENTER_C_SATURN] = {
        .kind = CONS_PLANET,
        .planet_hand = STRAIGHT,
    },
    [CENTER_C_URANUS] = {
        .kind = CONS_PLANET,
        .planet_hand = TWO_PAIR,
    },
    [CENTER_C_NEPTUNE] = {
        .kind = CONS_PLANET,
        .planet_hand = STRAIGHT_FLUSH,
    },
    [CENTER_C_PLUTO] = {
        .kind = CONS_PLANET,
        .planet_hand = HIGH_CARD,
    },
    [CENTER_C_PLANET_X] = {
        .kind = CONS_PLANET,
        .planet_hand = FIVE_OF_A_KIND,
    },
    [CENTER_C_CERES] = {
        .kind = CONS_PLANET,
        .planet_hand = FLUSH_HOUSE,
    },
    [CENTER_C_ERIS] = {
        .kind = CONS_PLANET,
        .planet_hand = FLUSH_FIVE,
    },
    [CENTER_C_BLACK_HOLE] = {
        .kind = CONS_BLACK_HOLE,
    },

    [CENTER_C_FOOL] = {
        .kind = CONS_FOOL,
        .use_rule = CONSUMABLE_USE_LAST_TAROT,
    },
    [CENTER_C_MAGICIAN] = {
        .kind = CONS_TARGET,
        .target_effect = TARGET_ENHANCEMENT,
        .target_value = ENHANCEMENT_LUCKY,
        .target_min = 1,
        .target_max = 2,
    },
    [CENTER_C_HIGH_PRIESTESS] = {
        .kind = CONS_SPAWN_POOL,
        .spawn_set = SET_PLANET,
        .spawn_count = 2,
        .stream = "pri",
    },
    [CENTER_C_EMPRESS] = {
        .kind = CONS_TARGET,
        .target_effect = TARGET_ENHANCEMENT,
        .target_value = ENHANCEMENT_MULT,
        .target_min = 1,
        .target_max = 2,
    },
    [CENTER_C_EMPEROR] = {
        .kind = CONS_SPAWN_POOL,
        .spawn_set = SET_TAROT,
        .spawn_count = 2,
        .stream = "emp",
    },
    [CENTER_C_HEIROPHANT] = {
        .kind = CONS_TARGET,
        .target_effect = TARGET_ENHANCEMENT,
        .target_value = ENHANCEMENT_BONUS,
        .target_min = 1,
        .target_max = 2,
    },
    [CENTER_C_LOVERS] = {
        .kind = CONS_TARGET,
        .target_effect = TARGET_ENHANCEMENT,
        .target_value = ENHANCEMENT_WILD,
        .target_min = 1,
        .target_max = 1,
    },
    [CENTER_C_CHARIOT] = {
        .kind = CONS_TARGET,
        .target_effect = TARGET_ENHANCEMENT,
        .target_value = ENHANCEMENT_STEEL,
        .target_min = 1,
        .target_max = 1,
    },
    [CENTER_C_JUSTICE] = {
        .kind = CONS_TARGET,
        .target_effect = TARGET_ENHANCEMENT,
        .target_value = ENHANCEMENT_GLASS,
        .target_min = 1,
        .target_max = 1,
    },
    [CENTER_C_HERMIT] = {
        .kind = CONS_HERMIT,
    },
    [CENTER_C_WHEEL_OF_FORTUNE] = {
        .kind = CONS_JOKER_EDITION,
        .use_rule = CONSUMABLE_USE_ACTIVE_EDITIONLESS_JOKER,
        .joker_effect = JOKER_ROLL_EDITION,
        .chance_percent = 25,
        .stream = "wheel_of_fortune",
    },
    [CENTER_C_STRENGTH] = {
        .kind = CONS_TARGET,
        .target_effect = TARGET_RANK_UP,
        .target_min = 1,
        .target_max = 2,
    },
    [CENTER_C_HANGED_MAN] = {
        .kind = CONS_HANGED_MAN,
        .target_min = 1,
        .target_max = 2,
    },
    [CENTER_C_DEATH] = {
        .kind = CONS_DEATH,
        .target_min = 2,
        .target_max = 2,
    },
    [CENTER_C_TEMPERANCE] = {
        .kind = CONS_TEMPERANCE,
    },
    [CENTER_C_DEVIL] = {
        .kind = CONS_TARGET,
        .target_effect = TARGET_ENHANCEMENT,
        .target_value = ENHANCEMENT_GOLD,
        .target_min = 1,
        .target_max = 1,
    },
    [CENTER_C_TOWER] = {
        .kind = CONS_TARGET,
        .target_effect = TARGET_ENHANCEMENT,
        .target_value = ENHANCEMENT_STONE,
        .target_min = 1,
        .target_max = 1,
    },
    [CENTER_C_STAR] = {
        .kind = CONS_TARGET,
        .target_effect = TARGET_SUIT,
        .target_value = DIAMONDS,
        .target_min = 1,
        .target_max = 3,
    },
    [CENTER_C_MOON] = {
        .kind = CONS_TARGET,
        .target_effect = TARGET_SUIT,
        .target_value = CLUBS,
        .target_min = 1,
        .target_max = 3,
    },
    [CENTER_C_SUN] = {
        .kind = CONS_TARGET,
        .target_effect = TARGET_SUIT,
        .target_value = HEARTS,
        .target_min = 1,
        .target_max = 3,
    },
    [CENTER_C_JUDGEMENT] = {
        .kind = CONS_JUDGEMENT,
        .use_rule = CONSUMABLE_USE_JOKER_ROOM,
        .stream = "jud",
    },
    [CENTER_C_WORLD] = {
        .kind = CONS_TARGET,
        .target_effect = TARGET_SUIT,
        .target_value = SPADES,
        .target_min = 1,
        .target_max = 3,
    },

    [CENTER_C_FAMILIAR] = {
        .kind = CONS_SPECTRAL_SPAWN,
        .use_rule = CONSUMABLE_USE_HAND_GT_ONE,
        .spawn_count = 3,
        .spectral_ranks = 1,
        .stream = "familiar_create",
    },
    [CENTER_C_GRIM] = {
        .kind = CONS_SPECTRAL_SPAWN,
        .use_rule = CONSUMABLE_USE_HAND_GT_ONE,
        .spawn_count = 2,
        .spectral_ranks = 2,
        .stream = "grim_create",
    },
    [CENTER_C_INCANTATION] = {
        .kind = CONS_SPECTRAL_SPAWN,
        .use_rule = CONSUMABLE_USE_HAND_GT_ONE,
        .spawn_count = 4,
        .spectral_ranks = 3,
        .stream = "incantation_create",
    },
    [CENTER_C_TALISMAN] = {
        .kind = CONS_TARGET,
        .target_effect = TARGET_SEAL,
        .target_value = SEAL_GOLD,
        .target_min = 1,
        .target_max = 1,
    },
    [CENTER_C_AURA] = {
        .kind = CONS_AURA,
        .target_min = 1,
        .target_max = 1,
        .target_filter = CONSUMABLE_TARGET_EDITIONLESS,
        .stream = "aura",
    },
    [CENTER_C_WRAITH] = {
        .kind = CONS_JOKER_RARITY,
        .use_rule = CONSUMABLE_USE_JOKER_ROOM,
        .rarity = 3,
        .reset_dollars = 1,
        .stream = "wra",
    },
    [CENTER_C_SIGIL] = {
        .kind = CONS_SIGIL,
        .use_rule = CONSUMABLE_USE_HAND_GT_ONE,
        .stream = "sigil",
    },
    [CENTER_C_OUIJA] = {
        .kind = CONS_OUIJA,
        .use_rule = CONSUMABLE_USE_HAND_GT_ONE,
        .stream = "ouija",
    },
    [CENTER_C_ECTOPLASM] = {
        .kind = CONS_JOKER_EDITION,
        .use_rule = CONSUMABLE_USE_EDITIONLESS_JOKER,
        .joker_effect = JOKER_NEGATIVE_SHRINK_HAND,
        .stream = "ectoplasm",
    },
    [CENTER_C_IMMOLATE] = {
        .kind = CONS_IMMOLATE,
        .use_rule = CONSUMABLE_USE_HAND_GT_ONE,
        .stream = "immolate",
    },
    [CENTER_C_ANKH] = {
        .kind = CONS_ANKH,
        .use_rule = CONSUMABLE_USE_JOKER_PRESENT_AND_ROOM,
        .stream = "ankh_choice",
    },
    [CENTER_C_DEJA_VU] = {
        .kind = CONS_TARGET,
        .target_effect = TARGET_SEAL,
        .target_value = SEAL_RED,
        .target_min = 1,
        .target_max = 1,
    },
    [CENTER_C_HEX] = {
        .kind = CONS_JOKER_EDITION,
        .use_rule = CONSUMABLE_USE_EDITIONLESS_JOKER,
        .joker_effect = JOKER_POLYCHROME_DESTROY,
        .stream = "hex",
    },
    [CENTER_C_TRANCE] = {
        .kind = CONS_TARGET,
        .target_effect = TARGET_SEAL,
        .target_value = SEAL_BLUE,
        .target_min = 1,
        .target_max = 1,
    },
    [CENTER_C_MEDIUM] = {
        .kind = CONS_TARGET,
        .target_effect = TARGET_SEAL,
        .target_value = SEAL_PURPLE,
        .target_min = 1,
        .target_max = 1,
    },
    [CENTER_C_CRYPTID] = {
        .kind = CONS_CRYPTID,
        .target_min = 1,
        .target_max = 1,
    },
    [CENTER_C_SOUL] = {
        .kind = CONS_JOKER_RARITY,
        .use_rule = CONSUMABLE_USE_JOKER_ROOM,
        .rarity = 4,
        .legendary = 1,
        .stream = "sou",
    },
};
#endif

#endif
