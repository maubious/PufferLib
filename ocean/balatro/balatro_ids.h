/* Generated content metadata. Do not hand-edit. */
#pragma once

typedef enum CenterId {
#define CENTER(name, id, ...) name = id,
#include "balatro_content.def"
#undef CENTER
    CENTER_COUNT = 300
} CenterId;

typedef enum BlindId {
    BLIND_NONE = 0,
    BLIND_BL_ARM = 1,
    BLIND_BL_BIG = 2,
    BLIND_BL_CLUB = 3,
    BLIND_BL_EYE = 4,
    BLIND_BL_FINAL_ACORN = 5,
    BLIND_BL_FINAL_BELL = 6,
    BLIND_BL_FINAL_HEART = 7,
    BLIND_BL_FINAL_LEAF = 8,
    BLIND_BL_FINAL_VESSEL = 9,
    BLIND_BL_FISH = 10,
    BLIND_BL_FLINT = 11,
    BLIND_BL_GOAD = 12,
    BLIND_BL_HEAD = 13,
    BLIND_BL_HOOK = 14,
    BLIND_BL_HOUSE = 15,
    BLIND_BL_MANACLE = 16,
    BLIND_BL_MARK = 17,
    BLIND_BL_MOUTH = 18,
    BLIND_BL_NEEDLE = 19,
    BLIND_BL_OX = 20,
    BLIND_BL_PILLAR = 21,
    BLIND_BL_PLANT = 22,
    BLIND_BL_PSYCHIC = 23,
    BLIND_BL_SERPENT = 24,
    BLIND_BL_SMALL = 25,
    BLIND_BL_TOOTH = 26,
    BLIND_BL_WALL = 27,
    BLIND_BL_WATER = 28,
    BLIND_BL_WHEEL = 29,
    BLIND_BL_WINDOW = 30,
    BLIND_COUNT = 31
} BlindId;

typedef enum TagId {
    TAG_NONE = 0,
    TAG_TAG_BOSS = 1,
    TAG_TAG_BUFFOON = 2,
    TAG_TAG_CHARM = 3,
    TAG_TAG_COUPON = 4,
    TAG_TAG_D_SIX = 5,
    TAG_TAG_DOUBLE = 6,
    TAG_TAG_ECONOMY = 7,
    TAG_TAG_ETHEREAL = 8,
    TAG_TAG_FOIL = 9,
    TAG_TAG_GARBAGE = 10,
    TAG_TAG_HANDY = 11,
    TAG_TAG_HOLO = 12,
    TAG_TAG_INVESTMENT = 13,
    TAG_TAG_JUGGLE = 14,
    TAG_TAG_METEOR = 15,
    TAG_TAG_NEGATIVE = 16,
    TAG_TAG_ORBITAL = 17,
    TAG_TAG_POLYCHROME = 18,
    TAG_TAG_RARE = 19,
    TAG_TAG_SKIP = 20,
    TAG_TAG_STANDARD = 21,
    TAG_TAG_TOP_UP = 22,
    TAG_TAG_UNCOMMON = 23,
    TAG_TAG_VOUCHER = 24,
    TAG_COUNT = 25
} TagId;
