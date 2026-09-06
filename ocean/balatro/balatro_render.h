#pragma once
#ifndef BALATRO_RENDER_H
#define BALATRO_RENDER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "raylib.h"
#include "rlgl.h"
#include "balatro_core.h"
#include "balatro_ids.h"
#include "joker_signatures.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Palette definitions */
static const Color BALATRO_BG           = (Color){14, 28, 24, 255};
static const Color BALATRO_PANEL_BG     = (Color){20, 40, 35, 255};
static const Color BALATRO_PANEL_BORDER = (Color){35, 68, 60, 255};
static const Color BALATRO_HEADER_BG    = (Color){28, 55, 48, 255};
static const Color BALATRO_BLUE_CHIPS   = (Color){0, 150, 255, 255};
static const Color BALATRO_RED_MULT     = (Color){255, 60, 60, 255};
static const Color BALATRO_GOLD         = (Color){255, 205, 35, 255};
static const Color BALATRO_WHITE        = (Color){245, 245, 245, 255};
static const Color BALATRO_DARK_TEXT    = (Color){30, 30, 35, 255};
static const Color BALATRO_CARD_WHITE   = (Color){248, 246, 240, 255};
static const Color BALATRO_HEART_RED    = (Color){230, 45, 45, 255};
static const Color BALATRO_DIAMOND_AMB  = (Color){240, 125, 30, 255};
static const Color BALATRO_CLUB_GRN     = (Color){35, 165, 95, 255};
static const Color BALATRO_SPADE_DARK   = (Color){45, 55, 75, 255};

/* Rarity Colors */
static const Color BALATRO_RARITY_COMMON    = (Color){185, 195, 210, 255};
static const Color BALATRO_RARITY_UNCOMMON  = (Color){45, 210, 115, 255};
static const Color BALATRO_RARITY_RARE      = (Color){35, 155, 255, 255};
static const Color BALATRO_RARITY_LEGENDARY = (Color){195, 75, 255, 255};

/* Consumable Colors */
static const Color BALATRO_COLOR_TAROT    = (Color){95, 45, 155, 255};
static const Color BALATRO_COLOR_PLANET   = (Color){25, 45, 120, 255};
static const Color BALATRO_COLOR_PLANET_TEXT = (Color){130, 195, 255, 255};
static const Color BALATRO_COLOR_SPECTRAL = (Color){20, 125, 135, 255};

/* Cosmetic wall-clock (card shimmer/pulse). Advanced once per live frame in
   balatro_render so it freezes while paused. File-static like the rest of
   this header's state: exactly one translation unit's copy ever runs. */
static float g_balatro_ui_time = 0.0f;
static const Color BALATRO_COLOR_VOUCHER  = (Color){180, 135, 60, 255};
static const Color BALATRO_COLOR_BOOSTER  = (Color){140, 60, 180, 255};

/* String and metadata lookup routines */
static const char *get_hand_type_name(HandType hand) {
    switch (hand) {
        case FLUSH_FIVE:      return "Flush Five";
        case FLUSH_HOUSE:     return "Flush House";
        case FIVE_OF_A_KIND:  return "Five of a Kind";
        case STRAIGHT_FLUSH:  return "Straight Flush";
        case FOUR_OF_A_KIND:  return "Four of a Kind";
        case FULL_HOUSE:      return "Full House";
        case FLUSH:           return "Flush";
        case STRAIGHT:        return "Straight";
        case THREE_OF_A_KIND: return "Three of a Kind";
        case TWO_PAIR:        return "Two Pair";
        case PAIR:            return "Pair";
        case HIGH_CARD:       return "High Card";
        default:              return "Unknown Hand";
    }
}

static const char *get_suit_name(uint8_t suit) {
    switch (suit) {
        case HEARTS:   return "Hearts";
        case DIAMONDS: return "Diamonds";
        case CLUBS:    return "Clubs";
        case SPADES:   return "Spades";
        default:       return "Suit";
    }
}

static const char *get_rank_str(uint8_t rank) {
    switch (rank) {
        case 2:  return "2";
        case 3:  return "3";
        case 4:  return "4";
        case 5:  return "5";
        case 6:  return "6";
        case 7:  return "7";
        case 8:  return "8";
        case 9:  return "9";
        case 10: return "10";
        case 11: return "J";
        case 12: return "Q";
        case 13: return "K";
        case 14: return "A";
        default: return "?";
    }
}

static void get_joker_value(const Card *card, const State *state, char *out, size_t max_len) {
    assert(card);
    assert(state);
    assert(out);
    assert(max_len);

    int value = card->state[0];
    switch (card->center_id) {
        case CENTER_J_CONSTELLATION:
        case CENTER_J_CAINO:
        case CENTER_J_CAMPFIRE:
        case CENTER_J_MADNESS:
        case CENTER_J_VAMPIRE:
        case CENTER_J_HOLOGRAM:
        case CENTER_J_LUCKY_CAT:
        case CENTER_J_OBELISK:
        case CENTER_J_GLASS:
        case CENTER_J_HIT_THE_ROAD:
            snprintf(out, max_len, "x%g MULT", (value > 100 ? value : 100) / 100.0);
            return;
        case CENTER_J_YORICK:
            snprintf(out, max_len, "x%g | %d LEFT", (value > 100 ? value : 100) / 100.0,
                     card->state[1] > 0 ? card->state[1] : 23);
            return;
        case CENTER_J_RAMEN:
            snprintf(out, max_len, "x%g MULT", (value > 100 ? value : 200) / 100.0);
            return;
        case CENTER_J_CEREMONIAL:
        case CENTER_J_GREEN_JOKER:
        case CENTER_J_RIDE_THE_BUS:
        case CENTER_J_FLASH:
        case CENTER_J_RED_CARD:
        case CENTER_J_TROUSERS:
            snprintf(out, max_len, "+%d MULT", value);
            return;
        case CENTER_J_SQUARE:
        case CENTER_J_RUNNER:
        case CENTER_J_WEE:
            snprintf(out, max_len, "+%d CHIPS", value);
            return;
        case CENTER_J_ICE_CREAM:
            snprintf(out, max_len, "+%d CHIPS", value > 0 ? value : 100);
            return;
        case CENTER_J_POPCORN:
            snprintf(out, max_len, "+%d MULT", value > 0 ? value : 20);
            return;
        case CENTER_J_TURTLE_BEAN:
            snprintf(out, max_len, "+%d HAND", value > 0 ? value : 5);
            return;
        case CENTER_J_SELZER:
            snprintf(out, max_len, "%d HANDS", value > 0 ? value : 10);
            return;
        case CENTER_J_ROCKET:
            snprintf(out, max_len, "$%d / ROUND", value > 0 ? value : 1);
            return;
        case CENTER_J_INVISIBLE:
            snprintf(out, max_len, "%d / 2 ROUNDS", value);
            return;
        case CENTER_J_LOYALTY_CARD: {
            int played = (int)state->run_hands_played - (card->state[2] ? card->state[1] : 0);
            int remaining = 5 - played % 6;
            if (remaining == 0)
                snprintf(out, max_len, "x4 ACTIVE");
            else
                snprintf(out, max_len, "x4 IN %d", remaining);
            return;
        }
        case CENTER_J_TODO_LIST:
            snprintf(out, max_len, "$4: %s", get_hand_type_name((HandType)card->state[1]));
            return;
        case CENTER_J_MAIL:
            snprintf(out, max_len, "$5: DISCARD %s", get_rank_str(state->mail_rank));
            return;
        case CENTER_J_CASTLE:
            snprintf(out, max_len, "+%d | %s", value, get_suit_name(state->castle_suit));
            return;
        case CENTER_J_IDOL:
            snprintf(out, max_len, "x2: %s %s", get_rank_str(state->idol_rank), get_suit_name(state->idol_suit));
            return;
        case CENTER_J_ANCIENT:
            snprintf(out, max_len, "x1.5: %s", get_suit_name(state->ancient_suit));
            return;
        case CENTER_J_FORTUNE_TELLER:
            snprintf(out, max_len, "+%d MULT", state->tarots_used);
            return;
        case CENTER_J_SATELLITE: {
            int planets = 0;
            for (uint16_t mask = state->planet_usage_mask; mask; mask &= (uint16_t)(mask - 1)) planets++;
            snprintf(out, max_len, "$%d / ROUND", planets);
            return;
        }
        case CENTER_J_THROWBACK:
            snprintf(out, max_len, "x%g MULT", 1.0 + state->skips * 0.25);
            return;
        case CENTER_J_ABSTRACT:
            snprintf(out, max_len, "+%d MULT", state->joker_count * 3);
            return;
        case CENTER_J_BANNER:
            snprintf(out, max_len, "+%d CHIPS", state->discards_left * 30);
            return;
        case CENTER_J_BLUE_JOKER:
            snprintf(out, max_len, "+%d CHIPS", state->deck_count * 2);
            return;
        case CENTER_J_BULL:
            snprintf(out, max_len, "+%d CHIPS", (state->dollars > 0 ? state->dollars : 0) * 2);
            return;
        case CENTER_J_BOOTSTRAPS:
            snprintf(out, max_len, "+%d MULT", (state->dollars > 0 ? state->dollars / 5 : 0) * 2);
            return;
        case CENTER_J_EROSION: {
            int starting = state->config.deck == CENTER_B_ABANDONED ? 40 : 52;
            int count = state->deck_count + state->hand_count + state->discard_count;
            snprintf(out, max_len, "+%d MULT", count < starting ? (starting - count) * 4 : 0);
            return;
        }
        case CENTER_J_STONE:
        case CENTER_J_STEEL_JOKER:
        case CENTER_J_CLOUD_9: {
            int stones = 0;
            int steels = 0;
            int nines = 0;
            const Card *zones[] = {state->deck, state->hand, state->discard};
            const uint16_t counts[] = {state->deck_count, state->hand_count, state->discard_count};
            for (int zone = 0; zone < 3; ++zone)
                for (uint16_t i = 0; i < counts[zone]; ++i) {
                    stones += zones[zone][i].enhancement == ENHANCEMENT_STONE;
                    steels += zones[zone][i].enhancement == ENHANCEMENT_STEEL;
                    nines += zones[zone][i].rank == 9;
                }
            if (card->center_id == CENTER_J_STONE)
                snprintf(out, max_len, "+%d CHIPS", stones * 25);
            else if (card->center_id == CENTER_J_STEEL_JOKER)
                snprintf(out, max_len, "x%g MULT", 1.0 + steels * 0.2);
            else
                snprintf(out, max_len, "$%d / ROUND", nines);
            return;
        }
        case CENTER_J_SWASHBUCKLER: {
            int total = 0;
            for (uint8_t i = 0; i < state->joker_count; ++i)
                if (!(state->jokers[i].flags & CARD_DEBUFFED)) total += state->jokers[i].sell_cost;
            for (uint8_t i = 0; i < state->joker_count; ++i)
                if (card == &state->jokers[i] && !(card->flags & CARD_DEBUFFED)) total -= card->sell_cost;
            snprintf(out, max_len, "+%d MULT", total);
            return;
        }
        case CENTER_J_STENCIL: {
            int stencils = 0;
            for (uint8_t i = 0; i < state->joker_count; ++i)
                stencils += state->jokers[i].center_id == CENTER_J_STENCIL && !(state->jokers[i].flags & CARD_DEBUFFED);
            snprintf(out, max_len, "x%d MULT", state->joker_slots - state->joker_count + stencils);
            return;
        }
        case CENTER_J_CHAOS: snprintf(out, max_len, "FREE REROLL"); return;
        case CENTER_J_BLUEPRINT: snprintf(out, max_len, "COPY RIGHT"); return;
        case CENTER_J_BRAINSTORM: snprintf(out, max_len, "COPY LEFT"); return;
        case CENTER_J_CHICOT: snprintf(out, max_len, "BOSS DISABLED"); return;
        case CENTER_J_LUCHADOR: snprintf(out, max_len, "SELL: BOSS OFF"); return;
        case CENTER_J_DIET_COLA: snprintf(out, max_len, "SELL: DOUBLE TAG"); return;
        case CENTER_J_CREDIT_CARD: snprintf(out, max_len, "-$20 CREDIT"); return;
        case CENTER_J_MARBLE: snprintf(out, max_len, "+1 STONE / BLIND"); return;
        case CENTER_J_CARTOMANCER: snprintf(out, max_len, "+1 TAROT / BLIND"); return;
        case CENTER_J_CERTIFICATE: snprintf(out, max_len, "+1 SEALED CARD"); return;
        case CENTER_J_RIFF_RAFF: snprintf(out, max_len, "+2 COMMONS"); return;
        case CENTER_J_MIDAS_MASK: snprintf(out, max_len, "FACES -> GOLD"); return;
        case CENTER_J_DNA: snprintf(out, max_len, "COPY FIRST CARD"); return;
        case CENTER_J_PERKEO: snprintf(out, max_len, "COPY CONSUMABLE"); return;
        case CENTER_J_OOPS: snprintf(out, max_len, "2x ODDS"); return;
        case CENTER_J_MR_BONES: snprintf(out, max_len, "SAVE @ 25%%"); return;
        case CENTER_J_MISPRINT: snprintf(out, max_len, "+0..23 MULT"); return;
        case CENTER_J_RING_MASTER: snprintf(out, max_len, "DUPLICATES OK"); return;
        case CENTER_J_SIXTH_SENSE: snprintf(out, max_len, "6 -> SPECTRAL"); return;
        case CENTER_J_SEANCE: snprintf(out, max_len, "SF -> SPECTRAL"); return;
        case CENTER_J_SUPERPOSITION: snprintf(out, max_len, "A+STRAIGHT TAROT"); return;
        case CENTER_J_VAGABOND: snprintf(out, max_len, "<= $4: TAROT"); return;
        case CENTER_J_DRIVERS_LICENSE: snprintf(out, max_len, "x3 AT 16 ENH"); return;
        case CENTER_J_FLOWER_POT: snprintf(out, max_len, "x3 ALL SUITS"); return;
        case CENTER_J_BLACKBOARD: snprintf(out, max_len, "x3 ALL BLACK"); return;
        case CENTER_J_BASEBALL: snprintf(out, max_len, "x1.5 / UNCOMMON"); return;
        case CENTER_J_MATADOR: snprintf(out, max_len, "$8 BOSS TRIGGER"); return;
        case CENTER_J_HALLUCINATION: snprintf(out, max_len, "1/2 PACK TAROT"); return;
        case CENTER_J_ASTRONOMER: snprintf(out, max_len, "FREE PLANETS"); return;
        case CENTER_J_8_BALL: snprintf(out, max_len, "1/4: TAROT"); return;
        case CENTER_J_SEEING_DOUBLE: snprintf(out, max_len, "x2 CLUB+OTHER"); return;
        case CENTER_J_RESERVED_PARKING: snprintf(out, max_len, "1/2: $1 HELD FACE"); return;
        case CENTER_J_BUSINESS: snprintf(out, max_len, "1/2: $2 FACE"); return;
        case CENTER_J_BLOODSTONE: snprintf(out, max_len, "1/2: x1.5 HEART"); return;
        case CENTER_J_FOUR_FINGERS: snprintf(out, max_len, "4-CARD STR/FLUSH"); return;
        case CENTER_J_SHORTCUT: snprintf(out, max_len, "GAPPED STRAIGHTS"); return;
        case CENTER_J_SMEARED: snprintf(out, max_len, "SUITS BY COLOR"); return;
        case CENTER_J_PAREIDOLIA: snprintf(out, max_len, "ALL CARDS = FACE"); return;
        case CENTER_J_SPLASH: snprintf(out, max_len, "ALL CARDS SCORE"); return;
        default:
            break;
    }

    const JokerSignature *effect = &JOKER_SIGNATURES[card->center_id];
    if (effect->chips && effect->mult)
        snprintf(out, max_len, "+%dC +%dM", effect->chips, effect->mult);
    else if (effect->chips)
        snprintf(out, max_len, "+%d CHIPS", effect->chips);
    else if (effect->mult)
        snprintf(out, max_len, "+%d MULT", effect->mult);
    else if (effect->xmult_pct)
        snprintf(out, max_len, "x%g MULT", effect->xmult_pct / 100.0);
    else if (effect->dollars)
        snprintf(out, max_len, "$%d", effect->dollars);
    else if (effect->retriggers)
        snprintf(out, max_len, "+%d RETRIGGER", effect->retriggers);
    else if (effect->levels)
        snprintf(out, max_len, "+%d LEVEL", effect->levels);
    else if (effect->hand_size || effect->discards || effect->hands) {
        if (effect->hand_size)
            snprintf(out, max_len, "%+d HAND SIZE", effect->hand_size);
        else if (effect->discards)
            snprintf(out, max_len, "%+d DISCARDS", effect->discards);
        else
            snprintf(out, max_len, "%+d HANDS", effect->hands);
    } else if (effect->sell_growth)
        snprintf(out, max_len, "SELL +$%d / RND", effect->sell_growth);
    else if (effect->rule)
        snprintf(out, max_len, "HAND RULE");
    else
        snprintf(out, max_len, "SPECIAL EFFECT");
}

static void get_raised_fist_value(const State *state, int selected_count, const uint8_t *selected_indices,
                                  char *out, size_t max_len) {
    assert(state);
    assert(selected_count == 0 || selected_indices);
    assert(out);
    assert(max_len);

    uint16_t lowest = state->hand_count;
    for (uint16_t i = 0; i < state->hand_count; ++i) {
        int selected = 0;
        for (int j = 0; j < selected_count; ++j) selected |= selected_indices[j] == i;
        if (selected || state->hand[i].enhancement == ENHANCEMENT_STONE) continue;
        if (lowest == state->hand_count || state->hand[i].rank <= state->hand[lowest].rank) lowest = i;
    }

    int mult = 0;
    if (lowest < state->hand_count && !(state->hand[lowest].flags & CARD_DEBUFFED)) {
        uint8_t rank = state->hand[lowest].rank;
        mult = 2 * (rank == 14 ? 11 : rank >= 10 ? 10 : rank);
    }
    snprintf(out, max_len, "+%d MULT", mult);
}

static const char *get_enhancement_name(uint8_t enh) {
    switch (enh) {
        case ENHANCEMENT_BONUS: return "Bonus (+30 Chips)";
        case ENHANCEMENT_MULT:  return "Mult (+4 Mult)";
        case ENHANCEMENT_WILD:  return "Wild (Any Suit)";
        case ENHANCEMENT_GLASS: return "Glass (x2 Mult, 1/4 Break)";
        case ENHANCEMENT_STEEL: return "Steel (x1.5 Mult in Hand)";
        case ENHANCEMENT_STONE: return "Stone (+50 Chips, No Rank)";
        case ENHANCEMENT_GOLD:  return "Gold (+$3 if Held)";
        case ENHANCEMENT_LUCKY: return "Lucky (1/5 +$20, 1/15 +20 Mult)";
        default: return "Standard";
    }
}

static const char *get_edition_name(uint8_t edition) {
    switch (edition) {
        case EDITION_FOIL:       return "Foil (+50 Chips)";
        case EDITION_HOLO:       return "Holographic (+10 Mult)";
        case EDITION_POLYCHROME: return "Polychrome (x1.5 Mult)";
        case EDITION_NEGATIVE:   return "Negative (+1 Slot)";
        default: return "Base";
    }
}

static const char *get_seal_name(uint8_t seal) {
    switch (seal) {
        case SEAL_GOLD:   return "Gold Seal (+$3 Scored)";
        case SEAL_RED:    return "Red Seal (Retrigger)";
        case SEAL_BLUE:   return "Blue Seal (Planet on Final Hand)";
        case SEAL_PURPLE: return "Purple Seal (Tarot on Discard)";
        default: return "None";
    }
}

static const char *get_blind_name(uint16_t id) {
    switch (id) {
        case BLIND_BL_SMALL: return "Small Blind";
        case BLIND_BL_BIG:   return "Big Blind";
        case BLIND_BL_HOOK:  return "The Hook";
        case BLIND_BL_OX:    return "The Ox";
        case BLIND_BL_HOUSE: return "The House";
        case BLIND_BL_WALL:  return "The Wall";
        case BLIND_BL_WHEEL: return "The Wheel";
        case BLIND_BL_ARM:   return "The Arm";
        case BLIND_BL_CLUB:  return "The Club";
        case BLIND_BL_FISH:  return "The Fish";
        case BLIND_BL_PSYCHIC: return "The Psychic";
        case BLIND_BL_GOAD:  return "The Goad";
        case BLIND_BL_WATER: return "The Water";
        case BLIND_BL_EYE:   return "The Eye";
        case BLIND_BL_MOUTH: return "The Mouth";
        case BLIND_BL_PLANT: return "The Plant";
        case BLIND_BL_NEEDLE: return "The Needle";
        case BLIND_BL_HEAD:  return "The Head";
        case BLIND_BL_TOOTH: return "The Tooth";
        case BLIND_BL_FLINT: return "The Flint";
        case BLIND_BL_MARK:  return "The Mark";
        case BLIND_BL_PILLAR: return "The Pillar";
        case BLIND_BL_MANACLE: return "The Manacle";
        case BLIND_BL_SERPENT: return "The Serpent";
        case BLIND_BL_WINDOW: return "The Window";
        case BLIND_BL_FINAL_ACORN: return "Amber Acorn";
        case BLIND_BL_FINAL_LEAF: return "Verdant Leaf";
        case BLIND_BL_FINAL_VESSEL: return "Violet Vessel";
        case BLIND_BL_FINAL_HEART: return "Crimson Heart";
        case BLIND_BL_FINAL_BELL: return "Cerulean Bell";
        default: return "Blind";
    }
}

static const char *get_blind_desc(uint16_t id) {
    switch (id) {
        case BLIND_BL_SMALL: return "Score required chips to beat blind";
        case BLIND_BL_BIG:   return "Score required chips to beat blind";
        case BLIND_BL_HOOK:  return "Discards 2 random cards per hand played";
        case BLIND_BL_OX:    return "Sets money to $0 if most played hand is played";
        case BLIND_BL_HOUSE: return "First hand is drawn face down";
        case BLIND_BL_WALL:  return "Extra large blind (4x base chips)";
        case BLIND_BL_WHEEL: return "1 in 7 cards drawn face down";
        case BLIND_BL_ARM:   return "Decreases level of played poker hand";
        case BLIND_BL_CLUB:  return "All Clubs cards are debuffed";
        case BLIND_BL_FISH:  return "Cards drawn face down after each play";
        case BLIND_BL_PSYCHIC: return "Must play 5 cards";
        case BLIND_BL_GOAD:  return "All Spades cards are debuffed";
        case BLIND_BL_WATER: return "Start with 0 discards";
        case BLIND_BL_EYE:   return "No repeat hand types this round";
        case BLIND_BL_MOUTH: return "Only one hand type allowed this round";
        case BLIND_BL_PLANT: return "All Face cards are debuffed";
        case BLIND_BL_NEEDLE: return "Play only 1 hand";
        case BLIND_BL_HEAD:  return "All Hearts cards are debuffed";
        case BLIND_BL_TOOTH: return "Lose $1 per card played";
        case BLIND_BL_FLINT: return "Base Chips and Mult are halved";
        case BLIND_BL_MARK:  return "All face cards drawn face down";
        case BLIND_BL_PILLAR: return "Cards played previously are debuffed";
        case BLIND_BL_MANACLE: return "-1 Hand Size";
        case BLIND_BL_SERPENT: return "Always draw 3 cards after play/discard";
        case BLIND_BL_WINDOW: return "All Diamonds cards are debuffed";
        case BLIND_BL_FINAL_ACORN: return "Flips and shuffles all Jokers";
        case BLIND_BL_FINAL_LEAF: return "All cards debuffed until 1 Joker sold";
        case BLIND_BL_FINAL_VESSEL: return "Very large blind (6x base chips)";
        case BLIND_BL_FINAL_HEART: return "One random Joker disabled every hand";
        case BLIND_BL_FINAL_BELL: return "Forces 1 card to always be selected";
        default: return "Defeat this blind to advance";
    }
}

static const char *get_tag_name(uint8_t tag) {
    switch (tag) {
        case TAG_TAG_UNCOMMON:   return "Uncommon Tag";
        case TAG_TAG_RARE:       return "Rare Tag";
        case TAG_TAG_NEGATIVE:   return "Negative Tag";
        case TAG_TAG_FOIL:       return "Foil Tag";
        case TAG_TAG_HOLO:       return "Holo Tag";
        case TAG_TAG_POLYCHROME: return "Polychrome Tag";
        case TAG_TAG_INVESTMENT: return "Investment Tag";
        case TAG_TAG_VOUCHER:    return "Voucher Tag";
        case TAG_TAG_BOSS:       return "Boss Tag";
        case TAG_TAG_STANDARD:   return "Standard Tag";
        case TAG_TAG_CHARM:      return "Charm Tag";
        case TAG_TAG_METEOR:     return "Meteor Tag";
        case TAG_TAG_BUFFOON:    return "Buffoon Tag";
        case TAG_TAG_HANDY:      return "Handy Tag";
        case TAG_TAG_GARBAGE:    return "Garbage Tag";
        case TAG_TAG_ETHEREAL:   return "Ethereal Tag";
        case TAG_TAG_COUPON:     return "Coupon Tag";
        case TAG_TAG_DOUBLE:     return "Double Tag";
        case TAG_TAG_JUGGLE:     return "Juggle Tag";
        case TAG_TAG_D_SIX:      return "D6 Tag";
        case TAG_TAG_TOP_UP:     return "Top-up Tag";
        case TAG_TAG_SKIP:       return "Skip Tag";
        case TAG_TAG_ORBITAL:    return "Orbital Tag";
        case TAG_TAG_ECONOMY:    return "Economy Tag";
        default: return "Skip Tag";
    }
}

static const char *get_tag_description(uint8_t tag) {
    switch (tag) {
        case TAG_TAG_UNCOMMON: return "The next shop has an Uncommon Joker";
        case TAG_TAG_RARE: return "The next shop has a Rare Joker";
        case TAG_TAG_NEGATIVE: return "The next shop Joker is Negative";
        case TAG_TAG_FOIL: return "The next shop Joker is Foil";
        case TAG_TAG_HOLO: return "The next shop Joker is Holographic";
        case TAG_TAG_POLYCHROME: return "The next shop Joker is Polychrome";
        case TAG_TAG_INVESTMENT: return "Earn $25 after defeating the Boss Blind";
        case TAG_TAG_VOUCHER: return "Adds an extra Voucher to the next shop";
        case TAG_TAG_BOSS: return "Rerolls the upcoming Boss Blind";
        case TAG_TAG_STANDARD: return "Immediately opens a free Mega Standard Pack";
        case TAG_TAG_CHARM: return "Immediately opens a free Mega Arcana Pack";
        case TAG_TAG_METEOR: return "Immediately opens a free Mega Celestial Pack";
        case TAG_TAG_BUFFOON: return "Immediately opens a free Mega Buffoon Pack";
        case TAG_TAG_HANDY: return "Earn $1 for every hand played this run";
        case TAG_TAG_GARBAGE: return "Earn $1 for every unused discard this run";
        case TAG_TAG_ETHEREAL: return "Immediately opens a free Spectral Pack";
        case TAG_TAG_COUPON: return "Cards and Booster Packs are free in the next shop";
        case TAG_TAG_DOUBLE: return "Copies the next selected Skip Tag";
        case TAG_TAG_JUGGLE: return "+3 Hand Size for the next round";
        case TAG_TAG_D_SIX: return "Rerolls start at $0 in the next shop";
        case TAG_TAG_TOP_UP: return "Creates up to 2 Common Jokers";
        case TAG_TAG_SKIP: return "Earn $5 for each Blind skipped this run";
        case TAG_TAG_ORBITAL: return "Upgrades a specified poker hand by 3 levels";
        case TAG_TAG_ECONOMY: return "Doubles current money, up to a $40 gain";
        default: assert(0); return "";
    }
}

static const char *get_center_name(uint16_t id) {
    switch (id) {
        /* Jokers */
        case CENTER_J_8_BALL: return "8 Ball";
        case CENTER_J_ABSTRACT: return "Abstract Joker";
        case CENTER_J_ACROBAT: return "Acrobat";
        case CENTER_J_ANCIENT: return "Ancient Joker";
        case CENTER_J_ARROWHEAD: return "Arrowhead";
        case CENTER_J_ASTRONOMER: return "Astronomer";
        case CENTER_J_BANNER: return "Banner";
        case CENTER_J_BARON: return "Baron";
        case CENTER_J_BASEBALL: return "Baseball Card";
        case CENTER_J_BLACKBOARD: return "Blackboard";
        case CENTER_J_BLOODSTONE: return "Bloodstone";
        case CENTER_J_BLUE_JOKER: return "Blue Joker";
        case CENTER_J_BLUEPRINT: return "Blueprint";
        case CENTER_J_BOOTSTRAPS: return "Bootstraps";
        case CENTER_J_BRAINSTORM: return "Brainstorm";
        case CENTER_J_BULL: return "Bull";
        case CENTER_J_BURGLAR: return "Burglar";
        case CENTER_J_BURNT: return "Burnt Joker";
        case CENTER_J_BUSINESS: return "Business Card";
        case CENTER_J_CAINO: return "Canio";
        case CENTER_J_CAMPFIRE: return "Campfire";
        case CENTER_J_CARD_SHARP: return "Card Sharp";
        case CENTER_J_CARTOMANCER: return "Cartomancer";
        case CENTER_J_CASTLE: return "Castle";
        case CENTER_J_CAVENDISH: return "Cavendish";
        case CENTER_J_CEREMONIAL: return "Ceremonial Dagger";
        case CENTER_J_CERTIFICATE: return "Certificate";
        case CENTER_J_CHAOS: return "Chaos the Clown";
        case CENTER_J_CHICOT: return "Chicot";
        case CENTER_J_CLEVER: return "Clever Joker";
        case CENTER_J_CLOUD_9: return "Cloud 9";
        case CENTER_J_CONSTELLATION: return "Constellation";
        case CENTER_J_CRAFTY: return "Crafty Joker";
        case CENTER_J_CRAZY: return "Crazy Joker";
        case CENTER_J_CREDIT_CARD: return "Credit Card";
        case CENTER_J_DELAYED_GRAT: return "Delayed Gratification";
        case CENTER_J_DEVIOUS: return "Devious Joker";
        case CENTER_J_DIET_COLA: return "Diet Cola";
        case CENTER_J_DNA: return "DNA";
        case CENTER_J_DRIVERS_LICENSE: return "Driver's License";
        case CENTER_J_DROLL: return "Droll Joker";
        case CENTER_J_DRUNKARD: return "Drunkard";
        case CENTER_J_DUO: return "The Duo";
        case CENTER_J_DUSK: return "Dusk";
        case CENTER_J_EGG: return "Egg";
        case CENTER_J_EROSION: return "Erosion";
        case CENTER_J_EVEN_STEVEN: return "Even Steven";
        case CENTER_J_FACELESS: return "Faceless Joker";
        case CENTER_J_FAMILY: return "The Family";
        case CENTER_J_FIBONACCI: return "Fibonacci";
        case CENTER_J_FLASH: return "Flash Card";
        case CENTER_J_FLOWER_POT: return "Flower Pot";
        case CENTER_J_FORTUNE_TELLER: return "Fortune Teller";
        case CENTER_J_FOUR_FINGERS: return "Four Fingers";
        case CENTER_J_GIFT: return "Gift Card";
        case CENTER_J_GLASS: return "Glass Joker";
        case CENTER_J_GLUTTENOUS_JOKER: return "Gluttonous Joker";
        case CENTER_J_GOLDEN: return "Golden Joker";
        case CENTER_J_GREEDY_JOKER: return "Greedy Joker";
        case CENTER_J_GREEN_JOKER: return "Green Joker";
        case CENTER_J_GROS_MICHEL: return "Gros Michel";
        case CENTER_J_HACK: return "Hack";
        case CENTER_J_HALF: return "Half Joker";
        case CENTER_J_HALLUCINATION: return "Hallucination";
        case CENTER_J_HANGING_CHAD: return "Hanging Chad";
        case CENTER_J_HIKER: return "Hiker";
        case CENTER_J_HIT_THE_ROAD: return "Hit the Road";
        case CENTER_J_HOLOGRAM: return "Hologram";
        case CENTER_J_ICE_CREAM: return "Ice Cream";
        case CENTER_J_IDOL: return "The Idol";
        case CENTER_J_INVISIBLE: return "Invisible Joker";
        case CENTER_J_JOKER: return "Joker";
        case CENTER_J_JOLLY: return "Jolly Joker";
        case CENTER_J_JUGGLER: return "Juggler";
        case CENTER_J_LOYALTY_CARD: return "Loyalty Card";
        case CENTER_J_LUCHADOR: return "Luchador";
        case CENTER_J_LUCKY_CAT: return "Lucky Cat";
        case CENTER_J_LUSTY_JOKER: return "Lusty Joker";
        case CENTER_J_MAD: return "Mad Joker";
        case CENTER_J_MADNESS: return "Madness";
        case CENTER_J_MAIL: return "Mail-In Rebate";
        case CENTER_J_MARBLE: return "Marble Joker";
        case CENTER_J_MATADOR: return "Matador";
        case CENTER_J_MERRY_ANDY: return "Merry Andy";
        case CENTER_J_MIDAS_MASK: return "Midas Mask";
        case CENTER_J_MIME: return "Mime";
        case CENTER_J_MISPRINT: return "Misprint";
        case CENTER_J_MR_BONES: return "Mr. Bones";
        case CENTER_J_MYSTIC_SUMMIT: return "Mystic Summit";
        case CENTER_J_OBELISK: return "Obelisk";
        case CENTER_J_ODD_TODD: return "Odd Todd";
        case CENTER_J_ONYX_AGATE: return "Onyx Agate";
        case CENTER_J_OOPS: return "Oops! All 6s";
        case CENTER_J_ORDER: return "The Order";
        case CENTER_J_PAREIDOLIA: return "Pareidolia";
        case CENTER_J_PERKEO: return "Perkeo";
        case CENTER_J_PHOTOGRAPH: return "Photograph";
        case CENTER_J_POPCORN: return "Popcorn";
        case CENTER_J_RAISED_FIST: return "Raised Fist";
        case CENTER_J_RAMEN: return "Ramen";
        case CENTER_J_RED_CARD: return "Red Card";
        case CENTER_J_RESERVED_PARKING: return "Reserved Parking";
        case CENTER_J_RIDE_THE_BUS: return "Ride the Bus";
        case CENTER_J_RIFF_RAFF: return "Riff-raff";
        case CENTER_J_RING_MASTER: return "Showman";
        case CENTER_J_ROCKET: return "Rocket";
        case CENTER_J_ROUGH_GEM: return "Rough Gem";
        case CENTER_J_RUNNER: return "Runner";
        case CENTER_J_SATELLITE: return "Satellite";
        case CENTER_J_SCARY_FACE: return "Scary Face";
        case CENTER_J_SCHOLAR: return "Scholar";
        case CENTER_J_SEANCE: return "Séance";
        case CENTER_J_SEEING_DOUBLE: return "Seeing Double";
        case CENTER_J_SELZER: return "Seltzer";
        case CENTER_J_SHOOT_THE_MOON: return "Shoot the Moon";
        case CENTER_J_SHORTCUT: return "Shortcut";
        case CENTER_J_SIXTH_SENSE: return "Sixth Sense";
        case CENTER_J_SLY: return "Sly Joker";
        case CENTER_J_SMEARED: return "Smeared Joker";
        case CENTER_J_SMILEY: return "Smiley Face";
        case CENTER_J_SOCK_AND_BUSKIN: return "Sock and Buskin";
        case CENTER_J_SPACE: return "Space Joker";
        case CENTER_J_TROUSERS: return "Spare Trousers";
        case CENTER_J_SPLASH: return "Splash";
        case CENTER_J_SQUARE: return "Square Joker";
        case CENTER_J_STEEL_JOKER: return "Steel Joker";
        case CENTER_J_STENCIL: return "Joker Stencil";
        case CENTER_J_STONE: return "Stone Joker";
        case CENTER_J_STUNTMAN: return "Stuntman";
        case CENTER_J_SUPERNOVA: return "Supernova";
        case CENTER_J_SUPERPOSITION: return "Superposition";
        case CENTER_J_SWASHBUCKLER: return "Swashbuckler";
        case CENTER_J_THROWBACK: return "Throwback";
        case CENTER_J_TICKET: return "Ticket";
        case CENTER_J_TODO_LIST: return "To Do List";
        case CENTER_J_TO_THE_MOON: return "To the Moon";
        case CENTER_J_TRADING: return "Trading Card";
        case CENTER_J_TRIBE: return "The Tribe";
        case CENTER_J_TRIBOULET: return "Triboulet";
        case CENTER_J_TRIO: return "The Trio";
        case CENTER_J_TROUBADOUR: return "Troubadour";
        case CENTER_J_TURTLE_BEAN: return "Turtle Bean";
        case CENTER_J_VAGABOND: return "Vagabond";
        case CENTER_J_VAMPIRE: return "Vampire";
        case CENTER_J_WALKIE_TALKIE: return "Walkie Talkie";
        case CENTER_J_WEE: return "Wee Joker";
        case CENTER_J_WILY: return "Wily Joker";
        case CENTER_J_WRATHFUL_JOKER: return "Wrathful Joker";
        case CENTER_J_YORICK: return "Yorick";
        case CENTER_J_ZANY: return "Zany Joker";

        /* Tarots */
        case CENTER_C_FOOL: return "The Fool";
        case CENTER_C_MAGICIAN: return "The Magician";
        case CENTER_C_HIGH_PRIESTESS: return "The High Priestess";
        case CENTER_C_EMPRESS: return "The Empress";
        case CENTER_C_EMPEROR: return "The Emperor";
        case CENTER_C_HEIROPHANT: return "The Hierophant";
        case CENTER_C_LOVERS: return "The Lovers";
        case CENTER_C_CHARIOT: return "The Chariot";
        case CENTER_C_JUSTICE: return "Justice";
        case CENTER_C_HERMIT: return "The Hermit";
        case CENTER_C_WHEEL_OF_FORTUNE: return "Wheel of Fortune";
        case CENTER_C_STRENGTH: return "Strength";
        case CENTER_C_HANGED_MAN: return "The Hanged Man";
        case CENTER_C_DEATH: return "Death";
        case CENTER_C_TEMPERANCE: return "Temperance";
        case CENTER_C_DEVIL: return "The Devil";
        case CENTER_C_TOWER: return "The Tower";
        case CENTER_C_STAR: return "The Star";
        case CENTER_C_MOON: return "The Moon";
        case CENTER_C_SUN: return "The Sun";
        case CENTER_C_JUDGEMENT: return "Judgement";
        case CENTER_C_WORLD: return "The World";

        /* Planets */
        case CENTER_C_MERCURY: return "Mercury";
        case CENTER_C_VENUS: return "Venus";
        case CENTER_C_EARTH: return "Earth";
        case CENTER_C_MARS: return "Mars";
        case CENTER_C_JUPITER: return "Jupiter";
        case CENTER_C_SATURN: return "Saturn";
        case CENTER_C_URANUS: return "Uranus";
        case CENTER_C_NEPTUNE: return "Neptune";
        case CENTER_C_PLUTO: return "Pluto";
        case CENTER_C_PLANET_X: return "Planet X";
        case CENTER_C_CERES: return "Ceres";
        case CENTER_C_ERIS: return "Eris";

        /* Spectrals */
        case CENTER_C_FAMILIAR: return "Familiar";
        case CENTER_C_GRIM: return "Grim";
        case CENTER_C_INCANTATION: return "Incantation";
        case CENTER_C_TALISMAN: return "Talisman";
        case CENTER_C_AURA: return "Aura";
        case CENTER_C_WRAITH: return "Wraith";
        case CENTER_C_SIGIL: return "Sigil";
        case CENTER_C_OUIJA: return "Ouija";
        case CENTER_C_ECTOPLASM: return "Ectoplasm";
        case CENTER_C_IMMOLATE: return "Immolate";
        case CENTER_C_ANKH: return "Ankh";
        case CENTER_C_DEJA_VU: return "Déjà Vu";
        case CENTER_C_HEX: return "Hex";
        case CENTER_C_TRANCE: return "Trance";
        case CENTER_C_MEDIUM: return "Medium";
        case CENTER_C_CRYPTID: return "Cryptid";
        case CENTER_C_SOUL: return "The Soul";
        case CENTER_C_BLACK_HOLE: return "Black Hole";

        /* Vouchers */
        case CENTER_V_OVERSTOCK_NORM: return "Overstock";
        case CENTER_V_OVERSTOCK_PLUS: return "Overstock Plus";
        case CENTER_V_CLEARANCE_SALE: return "Clearance Sale";
        case CENTER_V_LIQUIDATION: return "Liquidation";
        case CENTER_V_HONE: return "Hone";
        case CENTER_V_GLOW_UP: return "Glow Up";
        case CENTER_V_REROLL_SURPLUS: return "Reroll Surplus";
        case CENTER_V_REROLL_GLUT: return "Reroll Glut";
        case CENTER_V_CRYSTAL_BALL: return "Crystal Ball";
        case CENTER_V_OMEN_GLOBE: return "Omen Globe";
        case CENTER_V_TELESCOPE: return "Telescope";
        case CENTER_V_OBSERVATORY: return "Observatory";
        case CENTER_V_GRABBER: return "Grabber";
        case CENTER_V_NACHO_TONG: return "Nacho Tong";
        case CENTER_V_WASTEFUL: return "Wasteful";
        case CENTER_V_RECYCLOMANCY: return "Recyclomancy";
        case CENTER_V_TAROT_MERCHANT: return "Tarot Merchant";
        case CENTER_V_TAROT_TYCOON: return "Tarot Tycoon";
        case CENTER_V_PLANET_MERCHANT: return "Planet Merchant";
        case CENTER_V_PLANET_TYCOON: return "Planet Tycoon";
        case CENTER_V_SEED_MONEY: return "Seed Money";
        case CENTER_V_MONEY_TREE: return "Money Tree";
        case CENTER_V_BLANK: return "Blank";
        case CENTER_V_ANTIMATTER: return "Antimatter";
        case CENTER_V_MAGIC_TRICK: return "Magic Trick";
        case CENTER_V_ILLUSION: return "Illusion";
        case CENTER_V_HIEROGLYPH: return "Hieroglyph";
        case CENTER_V_PETROGLYPH: return "Petroglyph";
        case CENTER_V_DIRECTORS_CUT: return "Director's Cut";
        case CENTER_V_RETCON: return "Retcon";
        case CENTER_V_PAINT_BRUSH: return "Paint Brush";
        case CENTER_V_PALETTE: return "Palette";

        /* Booster packs */
        case CENTER_P_ARCANA_NORMAL_1:
        case CENTER_P_ARCANA_NORMAL_2:
        case CENTER_P_ARCANA_NORMAL_3:
        case CENTER_P_ARCANA_NORMAL_4: return "Arcane Pack";
        case CENTER_P_ARCANA_JUMBO_1:
        case CENTER_P_ARCANA_JUMBO_2:  return "Jumbo Arcane Pack";
        case CENTER_P_ARCANA_MEGA_1:
        case CENTER_P_ARCANA_MEGA_2:   return "Mega Arcane Pack";
        case CENTER_P_CELESTIAL_NORMAL_1:
        case CENTER_P_CELESTIAL_NORMAL_2:
        case CENTER_P_CELESTIAL_NORMAL_3:
        case CENTER_P_CELESTIAL_NORMAL_4: return "Celestial Pack";
        case CENTER_P_CELESTIAL_JUMBO_1:
        case CENTER_P_CELESTIAL_JUMBO_2:  return "Jumbo Celestial Pack";
        case CENTER_P_CELESTIAL_MEGA_1:
        case CENTER_P_CELESTIAL_MEGA_2:   return "Mega Celestial Pack";
        case CENTER_P_STANDARD_NORMAL_1:
        case CENTER_P_STANDARD_NORMAL_2:
        case CENTER_P_STANDARD_NORMAL_3:
        case CENTER_P_STANDARD_NORMAL_4: return "Standard Pack";
        case CENTER_P_STANDARD_JUMBO_1:
        case CENTER_P_STANDARD_JUMBO_2:  return "Jumbo Standard Pack";
        case CENTER_P_STANDARD_MEGA_1:
        case CENTER_P_STANDARD_MEGA_2:   return "Mega Standard Pack";
        case CENTER_P_BUFFOON_NORMAL_1:
        case CENTER_P_BUFFOON_NORMAL_2:  return "Buffoon Pack";
        case CENTER_P_BUFFOON_JUMBO_1:   return "Jumbo Buffoon Pack";
        case CENTER_P_BUFFOON_MEGA_1:    return "Mega Buffoon Pack";
        case CENTER_P_SPECTRAL_NORMAL_1:
        case CENTER_P_SPECTRAL_NORMAL_2: return "Spectral Pack";
        case CENTER_P_SPECTRAL_JUMBO_1:  return "Jumbo Spectral Pack";
        case CENTER_P_SPECTRAL_MEGA_1:   return "Mega Spectral Pack";

        /* Playing cards carry CENTER_C_BASE; only the card-capable sets are
           named here - decks/editions/enhancements are never card centers. */
        case CENTER_C_BASE: return "Playing Card";
        /* Zero value: zeroed/empty card slots can reach the name path. */
        case CENTER_NONE: return "None";

        default:
            assert(0);
            return "";
    }
}

static void get_center_description(const Card *card, const State *state, char *out, size_t max_len) {
    if (!card || !out || max_len == 0) return;
    int st = card->state[0];
    uint16_t cid = card->center_id;

    switch (cid) {
        /* Jokers */
        case CENTER_J_JOKER: snprintf(out, max_len, "+4 Mult"); break;
        case CENTER_J_GREEDY_JOKER: snprintf(out, max_len, "+3 Mult for each scored Diamond"); break;
        case CENTER_J_LUSTY_JOKER: snprintf(out, max_len, "+3 Mult for each scored Heart"); break;
        case CENTER_J_WRATHFUL_JOKER: snprintf(out, max_len, "+3 Mult for each scored Spade"); break;
        case CENTER_J_GLUTTENOUS_JOKER: snprintf(out, max_len, "+3 Mult for each scored Club"); break;
        case CENTER_J_CHAOS: snprintf(out, max_len, "+1 Free Reroll per shop"); break;
        case CENTER_J_JOLLY: snprintf(out, max_len, "+8 Mult if played hand contains a Pair"); break;
        case CENTER_J_ZANY: snprintf(out, max_len, "+12 Mult if played hand contains a Three of a Kind"); break;
        case CENTER_J_MAD: snprintf(out, max_len, "+10 Mult if played hand contains a Two Pair"); break;
        case CENTER_J_CRAZY: snprintf(out, max_len, "+12 Mult if played hand contains a Straight"); break;
        case CENTER_J_DROLL: snprintf(out, max_len, "+10 Mult if played hand contains a Flush"); break;
        case CENTER_J_SLY: snprintf(out, max_len, "+50 Chips if played hand contains a Pair"); break;
        case CENTER_J_WILY: snprintf(out, max_len, "+100 Chips if played hand contains a Three of a Kind"); break;
        case CENTER_J_CLEVER: snprintf(out, max_len, "+80 Chips if played hand contains a Two Pair"); break;
        case CENTER_J_DEVIOUS: snprintf(out, max_len, "+100 Chips if played hand contains a Straight"); break;
        case CENTER_J_CRAFTY: snprintf(out, max_len, "+80 Chips if played hand contains a Flush"); break;
        case CENTER_J_HALF: snprintf(out, max_len, "+20 Mult if played hand contains 3 or fewer cards"); break;
        case CENTER_J_STENCIL: {
            char value[32];
            get_joker_value(card, state, value, sizeof(value));
            snprintf(out, max_len, "x1 Mult for each empty Joker slot, including Stencil (Currently %s)", value);
            break;
        }
        case CENTER_J_FOUR_FINGERS: snprintf(out, max_len, "All Flushes and Straights can be made with 4 cards"); break;
        case CENTER_J_MIME: snprintf(out, max_len, "Retrigger all card held in hand abilities"); break;
        case CENTER_J_FLOWER_POT: snprintf(out, max_len, "x3 Mult if scoring cards contain all 4 suits"); break;
        case CENTER_J_CREDIT_CARD: snprintf(out, max_len, "Go up to -$20 in debt"); break;
        case CENTER_J_CEREMONIAL: snprintf(out, max_len, "Destroys joker to right on blind select, adds 2x sell value to Mult (Currently +%d)", st); break;
        case CENTER_J_BANNER: snprintf(out, max_len, "+30 Chips for each remaining discard (Currently +%d)", state ? state->discards_left * 30 : 30); break;
        case CENTER_J_MYSTIC_SUMMIT: snprintf(out, max_len, "+15 Mult when 0 discards remaining"); break;
        case CENTER_J_MARBLE: snprintf(out, max_len, "Adds one Stone card to deck when Blind is selected"); break;
        case CENTER_J_LOYALTY_CARD: {
            int played = state ? (int)state->run_hands_played - (card->state[2] ? card->state[1] : 0) : 0;
            int remaining = 5 - played % 6;
            if (remaining == 0)
                snprintf(out, max_len, "x4 Mult every 6 hands played (Active!)");
            else
                snprintf(out, max_len, "x4 Mult every 6 hands played (%d remaining)", remaining);
            break;
        }
        case CENTER_J_8_BALL: snprintf(out, max_len, "1 in 4 chance to create Tarot on scoring 8"); break;
        case CENTER_J_MISPRINT: snprintf(out, max_len, "+0 to +23 Mult (Random each hand)"); break;
        case CENTER_J_DUSK: snprintf(out, max_len, "Retrigger all played cards in final hand of round"); break;
        case CENTER_J_RAISED_FIST: snprintf(out, max_len, "Adds double the rank of lowest card in hand to Mult"); break;
        case CENTER_J_FIBONACCI: snprintf(out, max_len, "+8 Mult per scored Ace, 2, 3, 5, or 8"); break;
        case CENTER_J_STEEL_JOKER: {
            char value[32];
            get_joker_value(card, state, value, sizeof(value));
            snprintf(out, max_len, "Gives x0.2 Mult for each Steel Card in full deck (Currently %s)", value);
            break;
        }
        case CENTER_J_SCARY_FACE: snprintf(out, max_len, "+30 Chips per scored Face card"); break;
        case CENTER_J_ABSTRACT: snprintf(out, max_len, "+3 Mult for each Joker card owned (Currently +%d)", state ? state->joker_count * 3 : 3); break;
        case CENTER_J_DELAYED_GRAT: snprintf(out, max_len, "Earn $2 per discard if no discards used by end of round"); break;
        case CENTER_J_HACK: snprintf(out, max_len, "Retrigger each played 2, 3, 4, or 5"); break;
        case CENTER_J_PAREIDOLIA: snprintf(out, max_len, "All cards are considered Face cards"); break;
        case CENTER_J_GROS_MICHEL: snprintf(out, max_len, "+15 Mult. 1 in 6 chance to go extinct at end of round"); break;
        case CENTER_J_EVEN_STEVEN: snprintf(out, max_len, "+4 Mult per scored even card (10, 8, 6, 4, 2)"); break;
        case CENTER_J_ODD_TODD: snprintf(out, max_len, "+31 Chips per scored odd card (A, 9, 7, 5, 3)"); break;
        case CENTER_J_SCHOLAR: snprintf(out, max_len, "+20 Chips and +4 Mult per scored Ace"); break;
        case CENTER_J_BUSINESS: snprintf(out, max_len, "1 in 2 chance for scored face card to give $2"); break;
        case CENTER_J_SUPERNOVA: snprintf(out, max_len, "Adds Mult equal to number of times poker hand played"); break;
        case CENTER_J_RIDE_THE_BUS: snprintf(out, max_len, "+1 Mult per consecutive hand without a scoring face card (Currently +%d)", st); break;
        case CENTER_J_SPACE: snprintf(out, max_len, "1 in 4 chance to upgrade level of played poker hand"); break;
        case CENTER_J_EGG: snprintf(out, max_len, "Gains $3 sell value at end of round (Sell: $%d)", card->sell_cost); break;
        case CENTER_J_BURGLAR: snprintf(out, max_len, "+3 Hands, 0 Discards when Blind is selected"); break;
        case CENTER_J_BLACKBOARD: snprintf(out, max_len, "x3 Mult if all cards held in hand are Spades or Clubs"); break;
        case CENTER_J_ICE_CREAM: snprintf(out, max_len, "+100 Chips, -5 Chips for every hand played (Currently +%d)", st ? st : 100); break;
        case CENTER_J_DNA: snprintf(out, max_len, "First hand of round with 1 card copies it to deck permanently"); break;
        case CENTER_J_HANGING_CHAD: snprintf(out, max_len, "Retrigger first scored card 2 additional times"); break;
        case CENTER_J_SPLASH: snprintf(out, max_len, "Every played card counts in scoring"); break;
        case CENTER_J_BLUE_JOKER: snprintf(out, max_len, "+2 Chips for each remaining card in deck (Currently +%d)", state ? state->deck_count * 2 : 0); break;
        case CENTER_J_SIXTH_SENSE: snprintf(out, max_len, "In first hand of round, if single 6 scored, destroy to create Spectral"); break;
        case CENTER_J_CONSTELLATION: snprintf(out, max_len, "Gains x0.1 Mult per Planet card used (Currently x%g)", (st > 100 ? st : 100) / 100.0); break;
        case CENTER_J_HIKER: snprintf(out, max_len, "+5 Chips permanently to each played scoring card"); break;
        case CENTER_J_FACELESS: snprintf(out, max_len, "Earn $5 if 3 or more face cards discarded at same time"); break;
        case CENTER_J_GREEN_JOKER: snprintf(out, max_len, "+1 Mult per hand played, -1 Mult per discard (Currently +%d)", st); break;
        case CENTER_J_SUPERPOSITION: snprintf(out, max_len, "Create Tarot card if poker hand contains Ace and Straight"); break;
        case CENTER_J_TODO_LIST: snprintf(out, max_len, "Earn $4 if poker hand is %s", get_hand_type_name((HandType)card->state[1])); break;
        case CENTER_J_CERTIFICATE: snprintf(out, max_len, "Adds a random playing card with a random Seal to hand when round begins"); break;
        case CENTER_J_CAVENDISH: snprintf(out, max_len, "x3 Mult. 1 in 1000 chance to go extinct at end of round"); break;
        case CENTER_J_CARD_SHARP: snprintf(out, max_len, "x3 Mult if played poker hand has already been played this round"); break;
        case CENTER_J_RED_CARD: snprintf(out, max_len, "Gains +3 Mult when any Booster Pack is skipped (Currently +%d)", st); break;
        case CENTER_J_MADNESS: snprintf(out, max_len, "Gains x0.5 Mult when Small/Big Blind selected, destroys a random Joker (Currently x%g)", (st > 100 ? st : 100) / 100.0); break;
        case CENTER_J_SQUARE: snprintf(out, max_len, "Gains +4 Chips if played hand has exactly 4 cards (Currently +%d)", st); break;
        case CENTER_J_SEANCE: snprintf(out, max_len, "Create Spectral card if played hand is a Straight Flush"); break;
        case CENTER_J_RIFF_RAFF: snprintf(out, max_len, "Creates 2 Common Jokers when Blind is selected"); break;
        case CENTER_J_VAMPIRE: snprintf(out, max_len, "Gains x0.1 Mult per Enhanced card played, removes enhancement (Currently x%g)", (st > 100 ? st : 100) / 100.0); break;
        case CENTER_J_SHORTCUT: snprintf(out, max_len, "Allows Straights to be made with gaps of 1 rank"); break;
        case CENTER_J_HOLOGRAM: snprintf(out, max_len, "Gains x0.25 Mult whenever a playing card is added to deck (Currently x%g)", (st > 100 ? st : 100) / 100.0); break;
        case CENTER_J_VAGABOND: snprintf(out, max_len, "Creates Tarot card if hand is played with $4 or less"); break;
        case CENTER_J_BARON: snprintf(out, max_len, "Each King held in hand gives x1.5 Mult"); break;
        case CENTER_J_CLOUD_9: {
            char value[32];
            get_joker_value(card, state, value, sizeof(value));
            snprintf(out, max_len, "Earn $1 for each 9 in full deck at end of round (Currently %s)", value);
            break;
        }
        case CENTER_J_ROCKET: snprintf(out, max_len, "Earn $1 at end of round, +$2 when Boss Blind defeated (Currently $%d)", st ? st : 1); break;
        case CENTER_J_OBELISK: snprintf(out, max_len, "Gains x0.2 Mult per consecutive hand without playing most played poker hand (Currently x%g)", (st > 100 ? st : 100) / 100.0); break;
        case CENTER_J_MIDAS_MASK: snprintf(out, max_len, "All played face cards become Gold cards when scored"); break;
        case CENTER_J_LUCHADOR: snprintf(out, max_len, "Sell to disable current Boss Blind"); break;
        case CENTER_J_PHOTOGRAPH: snprintf(out, max_len, "First played face card gives x2 Mult when scored"); break;
        case CENTER_J_GIFT: snprintf(out, max_len, "Adds $1 sell value to every Joker and Consumable at end of round"); break;
        case CENTER_J_TURTLE_BEAN: snprintf(out, max_len, "+5 Hand size, decreases by 1 each round (Currently +%d)", st ? st : 5); break;
        case CENTER_J_EROSION: {
            char value[32];
            get_joker_value(card, state, value, sizeof(value));
            snprintf(out, max_len, "+4 Mult for each card below the deck's starting size (Currently %s)", value);
            break;
        }
        case CENTER_J_RESERVED_PARKING: snprintf(out, max_len, "1 in 2 chance for face cards in hand to give $1"); break;
        case CENTER_J_MAIL: snprintf(out, max_len, "Earn $5 per discarded %s", state ? get_rank_str(state->mail_rank) : "target rank"); break;
        case CENTER_J_TO_THE_MOON: snprintf(out, max_len, "Earn an extra $1 interest for every $5 you have at end of round"); break;
        case CENTER_J_HALLUCINATION: snprintf(out, max_len, "1 in 2 chance to create Tarot card when any Booster opened"); break;
        case CENTER_J_FORTUNE_TELLER: snprintf(out, max_len, "+1 Mult per Tarot card used this run (Currently +%d)", state ? state->tarots_used : 0); break;
        case CENTER_J_JUGGLER: snprintf(out, max_len, "+1 Hand Size"); break;
        case CENTER_J_TROUBADOUR: snprintf(out, max_len, "+2 Hand Size, -1 Hand per round"); break;
        case CENTER_J_DRUNKARD: snprintf(out, max_len, "+1 Discard per round"); break;
        case CENTER_J_STONE: {
            char value[32];
            get_joker_value(card, state, value, sizeof(value));
            snprintf(out, max_len, "+25 Chips for each Stone Card in full deck (Currently %s)", value);
            break;
        }
        case CENTER_J_GOLDEN: snprintf(out, max_len, "Earn $4 at end of round"); break;
        case CENTER_J_LUCKY_CAT: snprintf(out, max_len, "Gains x0.25 Mult each time a Lucky Card successfully triggers (Currently x%g)", (st > 100 ? st : 100) / 100.0); break;
        case CENTER_J_BULL: snprintf(out, max_len, "+2 Chips for each dollar you have (Currently +%d)", state ? state->dollars * 2 : 0); break;
        case CENTER_J_DIET_COLA: snprintf(out, max_len, "Sell this card to create a free Double Tag"); break;
        case CENTER_J_MERRY_ANDY: snprintf(out, max_len, "+3 Discards per round, -1 Hand Size"); break;
        case CENTER_J_TRADING: snprintf(out, max_len, "First discard of round with 1 card destroys it and gives $3"); break;
        case CENTER_J_FLASH: snprintf(out, max_len, "Gains +2 Mult per reroll in the shop (Currently +%d)", st); break;
        case CENTER_J_POPCORN: snprintf(out, max_len, "+20 Mult, loses -4 Mult each round (Currently +%d)", st ? st : 20); break;
        case CENTER_J_TROUSERS: snprintf(out, max_len, "Gains +2 Mult if played hand contains Two Pair (Currently +%d)", st); break;
        case CENTER_J_ANCIENT: snprintf(out, max_len, "x1.5 Mult per scored %s card; suit changes each round", state ? get_suit_name(state->ancient_suit) : "target suit"); break;
        case CENTER_J_RAMEN: snprintf(out, max_len, "Loses x0.01 Mult per card discarded (Currently x%g)", (st > 100 ? st : 200) / 100.0); break;
        case CENTER_J_WALKIE_TALKIE: snprintf(out, max_len, "+10 Chips and +4 Mult per scored 10 or 4"); break;
        case CENTER_J_SELZER: snprintf(out, max_len, "Retrigger all played cards for next %d hands", st ? st : 10); break;
        case CENTER_J_CASTLE: snprintf(out, max_len, "Gains +3 Chips per discarded %s card; suit changes each round (Currently +%d)",
                                      state ? get_suit_name(state->castle_suit) : "target suit", st); break;
        case CENTER_J_SMILEY: snprintf(out, max_len, "+5 Mult per scored Face card"); break;
        case CENTER_J_CAMPFIRE: snprintf(out, max_len, "Gains x0.25 Mult per card sold, resets on Boss defeated (Currently x%g)", (st > 100 ? st : 100) / 100.0); break;
        case CENTER_J_BLUEPRINT: snprintf(out, max_len, "Copies ability of Joker to the right"); break;
        case CENTER_J_OOPS: snprintf(out, max_len, "Doubles all listed probabilities"); break;
        case CENTER_J_WEE: snprintf(out, max_len, "Gains +8 Chips when each 2 is scored (Currently +%d)", st); break;
        case CENTER_J_HIT_THE_ROAD: snprintf(out, max_len, "Gains x0.5 Mult per Jack discarded this round (Currently x%g)", (st > 100 ? st : 100) / 100.0); break;
        case CENTER_J_IDOL: snprintf(out, max_len, "Each scored %s of %s gives x2 Mult; target changes each round",
                                    state ? get_rank_str(state->idol_rank) : "target rank",
                                    state ? get_suit_name(state->idol_suit) : "target suit"); break;
        case CENTER_J_DUO: snprintf(out, max_len, "x2 Mult if played hand contains a Pair"); break;
        case CENTER_J_TRIO: snprintf(out, max_len, "x3 Mult if played hand contains a Three of a Kind"); break;
        case CENTER_J_FAMILY: snprintf(out, max_len, "x4 Mult if played hand contains a Four of a Kind"); break;
        case CENTER_J_ORDER: snprintf(out, max_len, "x3 Mult if played hand contains a Straight"); break;
        case CENTER_J_TRIBE: snprintf(out, max_len, "x2 Mult if played hand contains a Flush"); break;
        case CENTER_J_STUNTMAN: snprintf(out, max_len, "+250 Chips, -2 Hand Size"); break;
        case CENTER_J_MR_BONES: snprintf(out, max_len, "Prevents defeat at 25%% of required chips, then destroys itself"); break;
        case CENTER_J_TICKET: snprintf(out, max_len, "Scored Gold cards earn $4"); break;
        case CENTER_J_INVISIBLE: snprintf(out, max_len, "Sell after 2 rounds to duplicate a random Joker"); break;
        case CENTER_J_BRAINSTORM: snprintf(out, max_len, "Copies ability of leftmost Joker"); break;
        case CENTER_J_SATELLITE: {
            char value[32];
            get_joker_value(card, state, value, sizeof(value));
            snprintf(out, max_len, "Earn $1 at end of round per unique Planet card used (Currently %s)", value);
            break;
        }
        case CENTER_J_SHOOT_THE_MOON: snprintf(out, max_len, "+13 Mult for each Queen held in hand"); break;
        case CENTER_J_DRIVERS_LICENSE: snprintf(out, max_len, "x3 Mult if you have at least 16 Enhanced cards in deck"); break;
        case CENTER_J_BOOTSTRAPS: snprintf(out, max_len, "+2 Mult for every $5 you have (Currently +%d)", state ? (state->dollars / 5) * 2 : 0); break;
        case CENTER_J_CAINO: snprintf(out, max_len, "Gains x1 Mult whenever a Face card is destroyed (Currently x%g)", (st > 100 ? st : 100) / 100.0); break;
        case CENTER_J_TRIBOULET: snprintf(out, max_len, "x2 Mult per scored King and Queen"); break;
        case CENTER_J_YORICK: snprintf(out, max_len, "Gains x1 Mult every 23 cards discarded (Currently x%g, %d remaining)",
                                      (st > 100 ? st : 100) / 100.0, card->state[1] > 0 ? card->state[1] : 23); break;
        case CENTER_J_CHICOT: snprintf(out, max_len, "Disables effect of every Boss Blind"); break;
        case CENTER_J_PERKEO: snprintf(out, max_len, "Creates a Negative copy of 1 random Consumable at shop exit"); break;
        case CENTER_J_BLOODSTONE: snprintf(out, max_len, "1 in 2 chance for scored Hearts to give x1.5 Mult"); break;
        case CENTER_J_ARROWHEAD: snprintf(out, max_len, "+50 Chips per scored Spade"); break;
        case CENTER_J_ONYX_AGATE: snprintf(out, max_len, "+7 Mult per scored Club"); break;
        case CENTER_J_ROUGH_GEM: snprintf(out, max_len, "Earn $1 per scored Diamond"); break;
        case CENTER_J_SMEARED: snprintf(out, max_len, "Hearts and Diamonds count as same suit; Spades and Clubs count as same suit"); break;
        case CENTER_J_ACROBAT: snprintf(out, max_len, "x3 Mult on final hand of round"); break;
        case CENTER_J_SOCK_AND_BUSKIN: snprintf(out, max_len, "Retrigger all played Face cards"); break;
        case CENTER_J_SWASHBUCKLER: {
            /* Engine (core.c main_jokers): mult += sum of sell values of all
               non-debuffed jokers, minus this one's own sell value when it is
               an owned, non-debuffed joker. state[0] is never written, so the
               current bonus must be computed from the state. */
            int sell_total = 0;
            if (state)
                for (uint8_t i = 0; i < state->joker_count; ++i)
                    if (!(state->jokers[i].flags & CARD_DEBUFFED)) sell_total += state->jokers[i].sell_cost;
            int owned = 0;
            if (state)
                for (uint8_t i = 0; i < state->joker_count; ++i)
                    owned |= card == &state->jokers[i];
            if (owned && !(card->flags & CARD_DEBUFFED)) sell_total -= card->sell_cost;
            snprintf(out, max_len, "Adds sell value of all other Jokers to Mult (Currently +%d)", sell_total);
            break;
        }
        case CENTER_J_THROWBACK: snprintf(out, max_len, "x0.25 Mult per Blind skipped this run (Currently x%.2f)", 1.0f + (state ? state->skips : 0) * 0.25f); break;
        case CENTER_J_MATADOR: snprintf(out, max_len, "Earn $8 if played hand triggers Boss Blind ability"); break;
        case CENTER_J_BURNT: snprintf(out, max_len, "Upgrade level of first discarded poker hand each round"); break;
        case CENTER_J_CARTOMANCER: snprintf(out, max_len, "Create Tarot card when Blind is selected"); break;
        case CENTER_J_ASTRONOMER: snprintf(out, max_len, "All Planet cards and Celestial Packs in shop are free"); break;
        case CENTER_J_BASEBALL: snprintf(out, max_len, "Uncommon Jokers each give x1.5 Mult"); break;
        case CENTER_J_GLASS: snprintf(out, max_len, "Gains x0.75 Mult for each Glass Card destroyed (Currently x%g)", (st > 100 ? st : 100) / 100.0); break;
        case CENTER_J_RING_MASTER: snprintf(out, max_len, "Joker, Tarot, Planet, and Spectral cards may appear multiple times"); break;
        case CENTER_J_RUNNER: snprintf(out, max_len, "Gains +15 Chips if played hand contains a Straight (Currently +%d)", st); break;
        case CENTER_J_SEEING_DOUBLE: snprintf(out, max_len, "x2 Mult if played hand has scoring Club card and scoring card of another suit"); break;

        /* Tarots */
        case CENTER_C_FOOL: snprintf(out, max_len, "Creates copy of last Tarot or Planet card used this run"); break;
        case CENTER_C_MAGICIAN: snprintf(out, max_len, "Enhances 2 selected cards into Lucky Cards"); break;
        case CENTER_C_HIGH_PRIESTESS: snprintf(out, max_len, "Creates up to 2 random Planet cards"); break;
        case CENTER_C_EMPRESS: snprintf(out, max_len, "Enhances 2 selected cards into Mult Cards (+4 Mult)"); break;
        case CENTER_C_EMPEROR: snprintf(out, max_len, "Creates up to 2 random Tarot cards"); break;
        case CENTER_C_HEIROPHANT: snprintf(out, max_len, "Enhances 2 selected cards into Bonus Cards (+30 Chips)"); break;
        case CENTER_C_LOVERS: snprintf(out, max_len, "Enhances 1 selected card into a Wild Card"); break;
        case CENTER_C_CHARIOT: snprintf(out, max_len, "Enhances 1 selected card into a Steel Card (x1.5 Mult in hand)"); break;
        case CENTER_C_JUSTICE: snprintf(out, max_len, "Enhances 1 selected card into a Glass Card (x2 Mult, 1 in 4 breaks)"); break;
        case CENTER_C_HERMIT: snprintf(out, max_len, "Doubles money (Max $20)"); break;
        case CENTER_C_WHEEL_OF_FORTUNE: snprintf(out, max_len, "1 in 4 chance to add Foil, Holographic, or Polychrome to random Joker"); break;
        case CENTER_C_STRENGTH: snprintf(out, max_len, "Increases rank of up to 2 selected cards by 1"); break;
        case CENTER_C_HANGED_MAN: snprintf(out, max_len, "Destroys up to 2 selected cards"); break;
        case CENTER_C_DEATH: snprintf(out, max_len, "Converts left selected card into copy of right selected card"); break;
        case CENTER_C_TEMPERANCE: snprintf(out, max_len, "Gives total sell value of all current Jokers (Max $50)"); break;
        case CENTER_C_DEVIL: snprintf(out, max_len, "Enhances 1 selected card into a Gold Card (+$3 at end of round)"); break;
        case CENTER_C_TOWER: snprintf(out, max_len, "Enhances 1 selected card into a Stone Card (+50 Chips, no suit/rank)"); break;
        case CENTER_C_STAR: snprintf(out, max_len, "Converts up to 3 selected cards to Diamonds"); break;
        case CENTER_C_MOON: snprintf(out, max_len, "Converts up to 3 selected cards to Clubs"); break;
        case CENTER_C_SUN: snprintf(out, max_len, "Converts up to 3 selected cards to Hearts"); break;
        case CENTER_C_WORLD: snprintf(out, max_len, "Converts up to 3 selected cards to Spades"); break;
        case CENTER_C_JUDGEMENT: snprintf(out, max_len, "Creates a random Joker card"); break;

        /* Planets */
        case CENTER_C_MERCURY: snprintf(out, max_len, "Level up Pair (+15 Chips, +1 Mult)"); break;
        case CENTER_C_VENUS: snprintf(out, max_len, "Level up Three of a Kind (+30 Chips, +2 Mult)"); break;
        case CENTER_C_EARTH: snprintf(out, max_len, "Level up Full House (+25 Chips, +2 Mult)"); break;
        case CENTER_C_MARS: snprintf(out, max_len, "Level up Four of a Kind (+30 Chips, +3 Mult)"); break;
        case CENTER_C_JUPITER: snprintf(out, max_len, "Level up Flush (+15 Chips, +2 Mult)"); break;
        case CENTER_C_SATURN: snprintf(out, max_len, "Level up Straight (+30 Chips, +3 Mult)"); break;
        case CENTER_C_URANUS: snprintf(out, max_len, "Level up Two Pair (+20 Chips, +1 Mult)"); break;
        case CENTER_C_NEPTUNE: snprintf(out, max_len, "Level up Straight Flush (+40 Chips, +4 Mult)"); break;
        case CENTER_C_PLUTO: snprintf(out, max_len, "Level up High Card (+10 Chips, +1 Mult)"); break;
        case CENTER_C_PLANET_X: snprintf(out, max_len, "Level up Five of a Kind (+35 Chips, +3 Mult)"); break;
        case CENTER_C_CERES: snprintf(out, max_len, "Level up Flush House (+40 Chips, +4 Mult)"); break;
        case CENTER_C_ERIS: snprintf(out, max_len, "Level up Flush Five (+50 Chips, +3 Mult)"); break;

        /* Spectrals */
        case CENTER_C_FAMILIAR: snprintf(out, max_len, "Destroy 1 random card in hand, add 3 random Enhanced face cards"); break;
        case CENTER_C_GRIM: snprintf(out, max_len, "Destroy 1 random card in hand, add 2 random Enhanced Aces"); break;
        case CENTER_C_INCANTATION: snprintf(out, max_len, "Destroy 1 random card in hand, add 4 random Enhanced numbered cards"); break;
        case CENTER_C_TALISMAN: snprintf(out, max_len, "Add Gold Seal to 1 selected card in hand"); break;
        case CENTER_C_AURA: snprintf(out, max_len, "Add Foil, Holographic, or Polychrome to 1 selected card"); break;
        case CENTER_C_WRAITH: snprintf(out, max_len, "Creates a random Rare Joker, sets money to $0"); break;
        case CENTER_C_SIGIL: snprintf(out, max_len, "Converts all cards in hand to a single random suit"); break;
        case CENTER_C_OUIJA: snprintf(out, max_len, "Converts all cards in hand to a single random rank, -1 Hand Size"); break;
        case CENTER_C_ECTOPLASM: snprintf(out, max_len, "Add Negative to a random Joker, -1 Hand Size"); break;
        case CENTER_C_IMMOLATE: snprintf(out, max_len, "Destroys 5 random cards in hand, gain $20"); break;
        case CENTER_C_ANKH: snprintf(out, max_len, "Create copy of random Joker, destroy all other Jokers"); break;
        case CENTER_C_DEJA_VU: snprintf(out, max_len, "Add Red Seal to 1 selected card in hand"); break;
        case CENTER_C_HEX: snprintf(out, max_len, "Add Polychrome to a random Joker, destroy all other Jokers"); break;
        case CENTER_C_TRANCE: snprintf(out, max_len, "Add Blue Seal to 1 selected card in hand"); break;
        case CENTER_C_MEDIUM: snprintf(out, max_len, "Add Purple Seal to 1 selected card in hand"); break;
        case CENTER_C_CRYPTID: snprintf(out, max_len, "Create 2 copies of 1 selected card in hand"); break;
        case CENTER_C_SOUL: snprintf(out, max_len, "Creates a Legendary Joker"); break;
        case CENTER_C_BLACK_HOLE: snprintf(out, max_len, "Upgrade every poker hand by 1 level"); break;

        /* Vouchers */
        case CENTER_V_OVERSTOCK_NORM: snprintf(out, max_len, "+1 Card slot available in shop"); break;
        case CENTER_V_OVERSTOCK_PLUS: snprintf(out, max_len, "+1 Card slot available in shop (+2 total)"); break;
        case CENTER_V_CLEARANCE_SALE: snprintf(out, max_len, "All cards and packs in shop are 25%% off"); break;
        case CENTER_V_LIQUIDATION: snprintf(out, max_len, "All cards and packs in shop are 50%% off"); break;
        case CENTER_V_HONE: snprintf(out, max_len, "Foil, Holographic, and Polychrome cards appear 2x more often"); break;
        case CENTER_V_GLOW_UP: snprintf(out, max_len, "Foil, Holographic, and Polychrome cards appear 4x more often"); break;
        case CENTER_V_REROLL_SURPLUS: snprintf(out, max_len, "Rerolls cost $2 less"); break;
        case CENTER_V_REROLL_GLUT: snprintf(out, max_len, "Rerolls cost an additional $2 less"); break;
        case CENTER_V_CRYSTAL_BALL: snprintf(out, max_len, "+1 Consumable slot"); break;
        case CENTER_V_OMEN_GLOBE: snprintf(out, max_len, "Spectral cards may appear in any Arcane Pack"); break;
        case CENTER_V_TELESCOPE: snprintf(out, max_len, "Celestial Packs always contain Planet for most played hand"); break;
        case CENTER_V_OBSERVATORY: snprintf(out, max_len, "Planet cards in Consumable area give x1.5 Mult for their hand"); break;
        case CENTER_V_GRABBER: snprintf(out, max_len, "+1 Hand per round"); break;
        case CENTER_V_NACHO_TONG: snprintf(out, max_len, "+1 Hand per round (+2 total)"); break;
        case CENTER_V_WASTEFUL: snprintf(out, max_len, "+1 Discard per round"); break;
        case CENTER_V_RECYCLOMANCY: snprintf(out, max_len, "+1 Discard per round (+2 total)"); break;
        case CENTER_V_TAROT_MERCHANT: snprintf(out, max_len, "Tarot cards appear 2x more often in shop"); break;
        case CENTER_V_TAROT_TYCOON: snprintf(out, max_len, "Tarot cards appear 4x more often in shop"); break;
        case CENTER_V_PLANET_MERCHANT: snprintf(out, max_len, "Planet cards appear 2x more often in shop"); break;
        case CENTER_V_PLANET_TYCOON: snprintf(out, max_len, "Planet cards appear 4x more often in shop"); break;
        case CENTER_V_SEED_MONEY: snprintf(out, max_len, "Raise cap on interest earned per round to $10"); break;
        case CENTER_V_MONEY_TREE: snprintf(out, max_len, "Raise cap on interest earned per round to $20"); break;
        case CENTER_V_BLANK: snprintf(out, max_len, "Does nothing?"); break;
        case CENTER_V_ANTIMATTER: snprintf(out, max_len, "+1 Joker Slot"); break;
        case CENTER_V_MAGIC_TRICK: snprintf(out, max_len, "Playing cards can be purchased from the shop"); break;
        case CENTER_V_ILLUSION: snprintf(out, max_len, "Playing cards in shop may have Enhancement, Edition, or Seal"); break;
        case CENTER_V_HIEROGLYPH: snprintf(out, max_len, "-1 Ante, -1 Hand each round"); break;
        case CENTER_V_PETROGLYPH: snprintf(out, max_len, "-1 Ante, -1 Discard each round"); break;
        case CENTER_V_DIRECTORS_CUT: snprintf(out, max_len, "Reroll Boss Blind 1 time per Ante, $10 per roll"); break;
        case CENTER_V_RETCON: snprintf(out, max_len, "Reroll Boss Blind unlimited times, $10 per roll"); break;
        case CENTER_V_PAINT_BRUSH: snprintf(out, max_len, "+1 Hand Size"); break;
        case CENTER_V_PALETTE: snprintf(out, max_len, "+1 Hand Size (+2 total)"); break;

        /* Packs */
        case CENTER_P_ARCANA_NORMAL_1:
        case CENTER_P_ARCANA_NORMAL_2:
        case CENTER_P_ARCANA_NORMAL_3:
        case CENTER_P_ARCANA_NORMAL_4: snprintf(out, max_len, "Choose 1 of 3 Tarot cards to use immediately"); break;
        case CENTER_P_ARCANA_JUMBO_1:
        case CENTER_P_ARCANA_JUMBO_2: snprintf(out, max_len, "Choose 1 of 5 Tarot cards to use immediately"); break;
        case CENTER_P_ARCANA_MEGA_1:
        case CENTER_P_ARCANA_MEGA_2: snprintf(out, max_len, "Choose 2 of 5 Tarot cards to use immediately"); break;
        case CENTER_P_CELESTIAL_NORMAL_1:
        case CENTER_P_CELESTIAL_NORMAL_2:
        case CENTER_P_CELESTIAL_NORMAL_3:
        case CENTER_P_CELESTIAL_NORMAL_4: snprintf(out, max_len, "Choose 1 of 3 Planet cards to use immediately"); break;
        case CENTER_P_CELESTIAL_JUMBO_1:
        case CENTER_P_CELESTIAL_JUMBO_2: snprintf(out, max_len, "Choose 1 of 5 Planet cards to use immediately"); break;
        case CENTER_P_CELESTIAL_MEGA_1:
        case CENTER_P_CELESTIAL_MEGA_2: snprintf(out, max_len, "Choose 2 of 5 Planet cards to use immediately"); break;
        case CENTER_P_STANDARD_NORMAL_1:
        case CENTER_P_STANDARD_NORMAL_2:
        case CENTER_P_STANDARD_NORMAL_3:
        case CENTER_P_STANDARD_NORMAL_4: snprintf(out, max_len, "Choose 1 of 3 Playing cards to add to deck"); break;
        case CENTER_P_STANDARD_JUMBO_1:
        case CENTER_P_STANDARD_JUMBO_2: snprintf(out, max_len, "Choose 1 of 5 Playing cards to add to deck"); break;
        case CENTER_P_STANDARD_MEGA_1:
        case CENTER_P_STANDARD_MEGA_2: snprintf(out, max_len, "Choose 2 of 5 Playing cards to add to deck"); break;
        case CENTER_P_BUFFOON_NORMAL_1:
        case CENTER_P_BUFFOON_NORMAL_2: snprintf(out, max_len, "Choose 1 of 2 Joker cards"); break;
        case CENTER_P_BUFFOON_JUMBO_1: snprintf(out, max_len, "Choose 1 of 4 Joker cards"); break;
        case CENTER_P_BUFFOON_MEGA_1: snprintf(out, max_len, "Choose 2 of 4 Joker cards"); break;
        case CENTER_P_SPECTRAL_NORMAL_1:
        case CENTER_P_SPECTRAL_NORMAL_2: snprintf(out, max_len, "Choose 1 of 2 Spectral cards to use immediately"); break;
        case CENTER_P_SPECTRAL_JUMBO_1: snprintf(out, max_len, "Choose 1 of 4 Spectral cards to use immediately"); break;
        case CENTER_P_SPECTRAL_MEGA_1: snprintf(out, max_len, "Choose 2 of 4 Spectral cards to use immediately"); break;

        default: assert(0); break;
    }
}

/* ========================================================================= */
/* Procedural Vector Drawing Primitives                                      */
/* ========================================================================= */

/* Draws a suit glyph. Shapes are built from rotated local coordinates so the
   whole glyph can be mirrored (180 degrees) for the bottom-right card pip. */
static void draw_suit_icon_rot(Suit suit, float cx, float cy, float size, float angle, Color color) {
    float ca = cosf(angle), sa = sinf(angle);
#define ST(x, y) ((Vector2){cx + (x) * ca - (y) * sa, cy + (x) * sa + (y) * ca})
    switch (suit) {
        case HEARTS: {
            float radius = size * 0.245f;
            DrawCircleV(ST(-size * 0.225f, -size * 0.17f), radius, color);
            DrawCircleV(ST(size * 0.225f, -size * 0.17f), radius, color);
            DrawTriangle(ST(-size * 0.45f, -size * 0.10f), ST(0, size * 0.54f),
                         ST(size * 0.45f, -size * 0.10f), color);
            break;
        }
        case DIAMONDS: {
            float hw = size * 0.39f, hh = size * 0.55f;
            DrawTriangle(ST(0, -hh), ST(-hw, 0), ST(hw, 0), color);
            DrawTriangle(ST(0, hh), ST(hw, 0), ST(-hw, 0), color);
            if (size >= 20.0f) {
                DrawTriangle(ST(0, -hh * 0.72f), ST(0, hh * 0.18f), ST(hw * 0.62f, 0), Fade(WHITE, 0.24f));
            }
            break;
        }
        case CLUBS: {
            float radius = size * 0.255f;
            DrawCircleV(ST(0, -size * 0.265f), radius, color);
            DrawCircleV(ST(-size * 0.285f, size * 0.02f), radius, color);
            DrawCircleV(ST(size * 0.285f, size * 0.02f), radius, color);
            DrawCircleV(ST(0, -size * 0.01f), size * 0.20f, color);
            DrawTriangle(ST(-size * 0.10f, size * 0.10f), ST(-size * 0.13f, size * 0.48f),
                         ST(size * 0.10f, size * 0.10f), color);
            DrawTriangle(ST(size * 0.10f, size * 0.10f), ST(-size * 0.13f, size * 0.48f),
                         ST(size * 0.13f, size * 0.48f), color);
            DrawLineEx(ST(-size * 0.22f, size * 0.48f), ST(size * 0.22f, size * 0.48f), size * 0.10f, color);
            break;
        }
        case SPADES: {
            float radius = size * 0.245f;
            DrawTriangle(ST(0, -size * 0.57f), ST(-size * 0.44f, size * 0.08f),
                         ST(size * 0.44f, size * 0.08f), color);
            DrawCircleV(ST(-size * 0.225f, size * 0.08f), radius, color);
            DrawCircleV(ST(size * 0.225f, size * 0.08f), radius, color);
            DrawCircleV(ST(0, size * 0.06f), size * 0.20f, color);
            DrawTriangle(ST(-size * 0.09f, size * 0.12f), ST(-size * 0.13f, size * 0.49f),
                         ST(size * 0.09f, size * 0.12f), color);
            DrawTriangle(ST(size * 0.09f, size * 0.12f), ST(-size * 0.13f, size * 0.49f),
                         ST(size * 0.13f, size * 0.49f), color);
            DrawLineEx(ST(-size * 0.21f, size * 0.49f), ST(size * 0.21f, size * 0.49f), size * 0.09f, color);
            break;
        }
    }
#undef ST
}

static Color get_suit_color(uint8_t suit) {
    switch (suit) {
        case HEARTS:   return BALATRO_HEART_RED;
        case DIAMONDS: return BALATRO_DIAMOND_AMB;
        case CLUBS:    return BALATRO_CLUB_GRN;
        case SPADES:   return BALATRO_SPADE_DARK;
        default:       return BALATRO_DARK_TEXT;
    }
}

/* Draws a procedural Playing Card */
static void draw_playing_card(const Card *card, Rectangle r, bool selected, bool hovered, bool in_shop, int cost, Texture2D puffer) {
    if (selected) {
        r.y -= 22.0f;
    } else if (hovered) {
        r.y -= 8.0f;
    }

    if (card->flags & CARD_FACEDOWN) {
        /* Card back: dark red with a puffer silhouette - distinct from the
           full-color mascot portraits on joker cards. */
        DrawRectangleRounded(r, 0.08f, 4, (Color){158, 32, 40, 255});
        DrawRectangleRoundedLinesEx((Rectangle){r.x + 5, r.y + 5, r.width - 10, r.height - 10}, 0.06f, 3, 1.5f, (Color){232, 200, 172, 255});
        float bcx = r.x + r.width * 0.5f, bcy = r.y + r.height * 0.5f;
        DrawPoly((Vector2){bcx, bcy}, 4, 20.0f, 45.0f, Fade((Color){232, 200, 172, 255}, 0.22f));
        assert(puffer.id > 0);
        DrawTexturePro(puffer, (Rectangle){0, 0, 128, 128},
                       (Rectangle){bcx - 17.0f, bcy - 17.0f, 34.0f, 34.0f},
                       (Vector2){0, 0}, 0, (Color){60, 12, 14, 255});
        if (selected) DrawRectangleRoundedLinesEx(r, 0.08f, 4, 3.5f, BALATRO_GOLD);
        else if (hovered) DrawRectangleRoundedLinesEx(r, 0.08f, 4, 2.5f, BALATRO_BLUE_CHIPS);
        return; /* hide rank, suit, enhancement, seal, edition, debuff */
    }

    Color face_color = BALATRO_CARD_WHITE;
    if (card->edition == EDITION_NEGATIVE) {
        face_color = (Color){20, 22, 28, 255};
    } else if (card->enhancement == ENHANCEMENT_STONE) {
        face_color = (Color){150, 155, 160, 255};
    } else if (card->enhancement == ENHANCEMENT_STEEL) {
        face_color = (Color){210, 215, 220, 255};
    } else if (card->enhancement == ENHANCEMENT_GOLD) {
        face_color = (Color){255, 248, 210, 255};
    } else if (card->enhancement == ENHANCEMENT_GLASS) {
        face_color = (Color){225, 245, 255, 255};
    }

    /* Card body and offset shadow give the hand a physical stack. */
    DrawRectangleRounded((Rectangle){r.x + 4, r.y + 6, r.width, r.height}, 0.08f, 4, Fade(BLACK, 0.36f));
    DrawRectangleRounded(r, 0.08f, 4, face_color);
    DrawRectangleRounded((Rectangle){r.x + 3, r.y + 3, r.width - 6, 23}, 0.16f, 3, Fade(get_suit_color(card->suit), 0.09f));

    /* Edition glowing border */
    Color border_color = (Color){180, 180, 180, 255};
    if (selected) {
        border_color = BALATRO_GOLD;
        DrawRectangleRoundedLinesEx(r, 0.08f, 4, 3.5f, BALATRO_GOLD);
    } else if (hovered) {
        border_color = BALATRO_BLUE_CHIPS;
        DrawRectangleRoundedLinesEx(r, 0.08f, 4, 2.5f, BALATRO_BLUE_CHIPS);
    } else if (card->edition == EDITION_FOIL) {
        border_color = BALATRO_BLUE_CHIPS;
        DrawRectangleRoundedLinesEx(r, 0.08f, 4, 2.0f, border_color);
    } else if (card->edition == EDITION_HOLO) {
        border_color = (Color){240, 60, 200, 255};
        DrawRectangleRoundedLinesEx(r, 0.08f, 4, 2.0f, border_color);
    } else if (card->edition == EDITION_POLYCHROME) {
        float t = g_balatro_ui_time * 3.0f;
        border_color = (Color){(unsigned char)(128 + 127*sinf(t)), (unsigned char)(128 + 127*sinf(t + 2.0f)), (unsigned char)(128 + 127*sinf(t + 4.0f)), 255};
        DrawRectangleRoundedLinesEx(r, 0.08f, 4, 2.5f, border_color);
    } else {
        DrawRectangleRoundedLinesEx(r, 0.08f, 4, 1.2f, border_color);
    }

    /* Stone card: No rank / suit */
    if (card->enhancement == ENHANCEMENT_STONE) {
        DrawText("+50", (int)(r.x + r.width * 0.5f - 18), (int)(r.y + r.height * 0.5f - 10), 20, (Color){60, 60, 60, 255});
        DrawText("STONE", (int)(r.x + 8), (int)(r.y + r.height - 18), 10, (Color){80, 80, 80, 255});
        return;
    }

    Color suit_color = get_suit_color(card->suit);
    if (card->edition == EDITION_NEGATIVE) {
        suit_color = BALATRO_WHITE;
    }

    /* Corner pips: rank + suit top-left, mirrored bottom-right */
    const char *rank_str = get_rank_str(card->rank);
    DrawText(rank_str, (int)(r.x + 6), (int)(r.y + 4), 20, suit_color);
    draw_suit_icon_rot((Suit)card->suit, r.x + 13, r.y + 34, 15, 0.0f, suit_color);

    float rr_x = r.x + r.width - 15.0f, rr_y = r.y + r.height - 13.0f;
    DrawTextPro(GetFontDefault(), rank_str, (Vector2){rr_x, rr_y},
                (Vector2){(float)MeasureText(rank_str, 20) * 0.5f, 10.0f}, 180.0f, 20, 1.0f, suit_color);
    draw_suit_icon_rot((Suit)card->suit, r.x + r.width - 13.0f, r.y + r.height - 34.0f, 15.0f, PI, suit_color);

    /* Inner frame line */
    DrawRectangleRoundedLinesEx((Rectangle){r.x + 2.5f, r.y + 2.5f, r.width - 5.0f, r.height - 5.0f},
                                0.06f, 3, 1.0f, Fade(suit_color, 0.22f));

    /* Center Artwork */
    if (card->rank >= 11 && card->rank <= 14) {
        /* Face card emblem: rounded medallion with flanking suit pips */
        Rectangle em = {r.x + r.width * 0.5f - 22.0f, r.y + r.height * 0.5f - 22.0f, 44.0f, 44.0f};
        DrawRectangleRounded(em, 0.22f, 3, Fade(suit_color, 0.14f));
        DrawRectangleRoundedLinesEx(em, 0.22f, 3, 1.6f, suit_color);
        DrawText(rank_str, (int)(r.x + r.width * 0.5f - MeasureText(rank_str, 24) * 0.5f),
                 (int)(r.y + r.height * 0.5f - 12), 24, suit_color);
        draw_suit_icon_rot((Suit)card->suit, em.x + 8.0f, em.y + 8.0f, 9.0f, 0.0f, suit_color);
        draw_suit_icon_rot((Suit)card->suit, em.x + em.width - 8.0f, em.y + em.height - 8.0f, 9.0f, PI, suit_color);
    } else {
        /* Clean oversized central pip. */
        draw_suit_icon_rot((Suit)card->suit, r.x + r.width * 0.5f, r.y + r.height * 0.5f, 30, 0.0f, suit_color);
    }

    /* Enhancement badge at top */
    if (card->enhancement != ENHANCEMENT_NONE) {
        Color enh_bg = BALATRO_BLUE_CHIPS;
        const char *enh_lbl = "ENH";
        switch (card->enhancement) {
            case ENHANCEMENT_BONUS: enh_bg = BALATRO_BLUE_CHIPS; enh_lbl = "+30"; break;
            case ENHANCEMENT_MULT:  enh_bg = BALATRO_RED_MULT;   enh_lbl = "+4M"; break;
            case ENHANCEMENT_WILD:  enh_bg = (Color){180, 70, 220, 255}; enh_lbl = "WILD"; break;
            case ENHANCEMENT_GLASS: enh_bg = (Color){80, 180, 220, 255}; enh_lbl = "x2"; break;
            case ENHANCEMENT_STEEL: enh_bg = (Color){100, 110, 120, 255}; enh_lbl = "1.5x"; break;
            case ENHANCEMENT_GOLD:  enh_bg = BALATRO_GOLD;       enh_lbl = "+$3"; break;
            case ENHANCEMENT_LUCKY: enh_bg = (Color){40, 180, 90, 255}; enh_lbl = "LCK"; break;
        }
        DrawRectangleRounded((Rectangle){r.x + r.width - 34, r.y + 4, 30, 14}, 0.3f, 2, enh_bg);
        DrawText(enh_lbl, (int)(r.x + r.width - 32), (int)(r.y + 5), 10, BALATRO_WHITE);
    }

    /* Seals */
    if (card->seal != SEAL_NONE) {
        Color seal_c = BALATRO_GOLD;
        const char *seal_s = "$";
        switch (card->seal) {
            case SEAL_GOLD:   seal_c = BALATRO_GOLD; seal_s = "$"; break;
            case SEAL_RED:    seal_c = BALATRO_RED_MULT; seal_s = "2x"; break;
            case SEAL_BLUE:   seal_c = BALATRO_BLUE_CHIPS; seal_s = "P"; break;
            case SEAL_PURPLE: seal_c = BALATRO_COLOR_TAROT; seal_s = "T"; break;
        }
        DrawCircle((int)(r.x + 12), (int)(r.y + r.height - 12), 9, seal_c);
        DrawText(seal_s, (int)(r.x + 8), (int)(r.y + r.height - 18), 11, BALATRO_WHITE);
    }

    /* Debuffed overlay */
    if (card->flags & CARD_DEBUFFED) {
        DrawRectangleRounded(r, 0.08f, 4, Fade(BLACK, 0.45f));
        DrawLineEx((Vector2){r.x + 5, r.y + 5}, (Vector2){r.x + r.width - 5, r.y + r.height - 5}, 3.0f, BALATRO_RED_MULT);
        DrawRectangle((int)r.x, (int)(r.y + r.height * 0.5f - 8), (int)r.width, 16, BALATRO_RED_MULT);
        DrawText("DEBUFFED", (int)(r.x + 8), (int)(r.y + r.height * 0.5f - 6), 10, BALATRO_WHITE);
    }

    /* Shop price badge */
    if (in_shop) {
        DrawRectangleRounded((Rectangle){r.x + r.width * 0.5f - 20, r.y + r.height - 18, 40, 16}, 0.3f, 2, BALATRO_GOLD);
        DrawText(TextFormat("$%d", cost), (int)(r.x + r.width * 0.5f - 12), (int)(r.y + r.height - 16), 12, BALATRO_DARK_TEXT);
    }
}

/* Centered multiline text for card titles */
static void draw_centered_multiline_text(const char *text, int cx, int y, int max_w, int max_lines, int font_size, Color color) {
    if (!text || !text[0]) return;

    /* If full text fits on a single line, render directly centered */
    if (MeasureText(text, font_size) <= max_w) {
        DrawText(text, cx - MeasureText(text, font_size) / 2, y, font_size, color);
        return;
    }

    char buffer[128];
    strncpy(buffer, text, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';

    char *start = buffer;
    char *word = strtok(start, " ");
    char line[64] = {0};
    int cur_y = y;
    int line_count = 0;

    while (word != NULL && line_count < max_lines) {
        char test_line[64];
        if (line[0] == '\0') {
            snprintf(test_line, sizeof(test_line), "%s", word);
        } else {
            snprintf(test_line, sizeof(test_line), "%s %s", line, word);
        }

        int width = MeasureText(test_line, font_size);
        if (width > max_w && line[0] != '\0') {
            DrawText(line, cx - MeasureText(line, font_size) / 2, cur_y, font_size, color);
            cur_y += font_size + 1;
            line_count++;
            snprintf(line, sizeof(line), "%s", word);
        } else {
            snprintf(line, sizeof(line), "%s", test_line);
        }
        word = strtok(NULL, " ");
    }
    if (line[0] != '\0' && line_count < max_lines) {
        DrawText(line, cx - MeasureText(line, font_size) / 2, cur_y, font_size, color);
    }
}

/* Draws a procedural Joker card */
static void draw_joker_card(const Card *card, Rectangle r, bool hovered, bool in_shop, int cost, Texture2D puffer) {
    if (hovered) {
        r.y -= 6.0f;
    }

    const CenterDefinition *def = &centers[card->center_id];
    Color rarity_color = BALATRO_RARITY_COMMON;
    const char *rarity_name = "Common";
    if (def->rarity == 2) {
        rarity_color = BALATRO_RARITY_UNCOMMON;
        rarity_name = "Uncommon";
    } else if (def->rarity == 3) {
        rarity_color = BALATRO_RARITY_RARE;
        rarity_name = "Rare";
    } else if (def->rarity == 4) {
        rarity_color = BALATRO_RARITY_LEGENDARY;
        rarity_name = "Legendary";
    }

    Color surface = card->edition == EDITION_NEGATIVE ? (Color){16, 17, 28, 255} : (Color){234, 235, 226, 255};
    uint32_t pattern = (uint32_t)card->center_id * UINT32_C(2654435761);
    Color accent = {
        (unsigned char)(70 + (pattern & 0x7f)),
        (unsigned char)(70 + ((pattern >> 8) & 0x7f)),
        (unsigned char)(70 + ((pattern >> 16) & 0x7f)),
        255,
    };

    DrawRectangleRounded((Rectangle){r.x + 4, r.y + 6, r.width, r.height}, 0.08f, 4, Fade(BLACK, 0.42f));
    DrawRectangleRounded(r, 0.08f, 4, surface);
    DrawRectangleRounded((Rectangle){r.x + 3, r.y + 3, r.width - 6, r.height - 6}, 0.07f, 4,
                         card->edition == EDITION_NEGATIVE ? (Color){28, 25, 45, 255} : Fade(rarity_color, 0.13f));

    Rectangle art = {r.x + 6, r.y + 42, r.width - 12, r.height - 48};
    Color art_bg = card->edition == EDITION_NEGATIVE
        ? (Color){22, 18, 39, 255}
        : (Color){(unsigned char)(28 + accent.r / 7), (unsigned char)(32 + accent.g / 7),
                  (unsigned char)(34 + accent.b / 7), 255};
    DrawRectangleRounded(art, 0.12f, 4, art_bg);

    /* Every Joker receives a stable, center-specific print motif. */
    switch (card->center_id & 3u) {
        case 0:
            for (int ray = 0; ray < 8; ++ray) {
                float angle = ray * PI * 0.25f + (float)(pattern & 15u) * 0.03f;
                float radius = fminf(art.width * 0.44f, art.height * 0.42f);
                DrawLineEx((Vector2){art.x + art.width * 0.5f, art.y + art.height * 0.53f},
                           (Vector2){art.x + art.width * 0.5f + cosf(angle) * radius,
                                     art.y + art.height * 0.53f + sinf(angle) * radius},
                           1.0f, Fade(accent, 0.28f));
            }
            break;
        case 1:
            for (int diamond = 0; diamond < 3; ++diamond)
                DrawPoly((Vector2){art.x + art.width * (0.25f + diamond * 0.25f),
                                   art.y + art.height * (diamond & 1 ? 0.68f : 0.30f)},
                         4, 6.0f + diamond * 2.0f, 45.0f, Fade(accent, 0.20f));
            break;
        case 2:
            for (int bubble = 0; bubble < 7; ++bubble) {
                float bx = art.x + 6.0f + fmodf((float)(bubble * 23 + (pattern & 31u)), art.width - 12.0f);
                float by = art.y + 6.0f + fmodf((float)(bubble * 17 + ((pattern >> 5) & 31u)), art.height - 12.0f);
                DrawCircleLines((int)bx, (int)by, 2.0f + (bubble % 3), Fade(accent, 0.30f));
            }
            break;
        default:
            for (int stripe = 0; stripe < 5; ++stripe) {
                float x = art.x + 3.0f + stripe * (art.width - 6.0f) * 0.18f;
                DrawLineEx((Vector2){x, art.y + art.height - 3},
                           (Vector2){fminf(x + art.width * 0.28f, art.x + art.width - 3.0f), art.y + 3},
                           2.0f, Fade(accent, 0.20f));
            }
            break;
    }

    float cx = art.x + art.width * 0.5f;
    float cy = art.y + art.height * 0.53f;
    float portrait_radius = fminf(art.width * 0.34f, art.height * 0.38f);
    DrawPoly((Vector2){cx + 2, cy + 3}, 6, portrait_radius * 1.12f, 30.0f, Fade(BLACK, 0.36f));
    DrawPoly((Vector2){cx, cy}, 6, portrait_radius * 1.12f, 30.0f, Fade(accent, 0.48f));
    DrawPolyLinesEx((Vector2){cx, cy}, 6, portrait_radius * 1.12f, 30.0f, 1.2f, Fade(rarity_color, 0.88f));
    assert(puffer.id > 0);
    float source_x = card->center_id & 1u ? 128.0f : 0.0f;
    DrawTexturePro(puffer, (Rectangle){source_x, 0, 128, 128},
                   (Rectangle){cx - portrait_radius, cy - portrait_radius,
                               portrait_radius * 2.0f, portrait_radius * 2.0f},
                   (Vector2){0, 0}, 0, WHITE);

    const JokerSignature *effect = &JOKER_SIGNATURES[card->center_id];
    Color effect_color = effect->chips ? BALATRO_BLUE_CHIPS
        : effect->mult ? BALATRO_RED_MULT
        : effect->xmult_pct ? BALATRO_RARITY_LEGENDARY
        : effect->dollars || effect->sell_growth ? BALATRO_GOLD
        : effect->rule ? BALATRO_RARITY_UNCOMMON : accent;
    DrawCircle((int)(art.x + 8), (int)(art.y + art.height - 8), 4.5f, Fade(BLACK, 0.55f));
    DrawCircle((int)(art.x + 8), (int)(art.y + art.height - 9), 3.2f, effect_color);

    if (card->edition == EDITION_FOIL) {
        DrawLineEx((Vector2){art.x + 5, art.y + art.height - 8},
                   (Vector2){art.x + art.width - 5, art.y + 8}, 4.0f, Fade((Color){150, 225, 255, 255}, 0.34f));
    } else if (card->edition == EDITION_HOLO) {
        DrawCircleLines((int)cx, (int)cy, portrait_radius * 1.14f, Fade((Color){255, 85, 205, 255}, 0.72f));
        DrawCircleLines((int)cx, (int)cy, portrait_radius * 1.28f, Fade((Color){70, 235, 255, 255}, 0.62f));
    } else if (card->edition == EDITION_POLYCHROME) {
        Color colors[] = {{255, 80, 120, 255}, {90, 225, 255, 255}, {230, 110, 255, 255}};
        for (int stripe = 0; stripe < 3; ++stripe)
            DrawLineEx((Vector2){art.x + 5, art.y + 9 + stripe * 7},
                       (Vector2){art.x + art.width - 5, art.y + 3 + stripe * 7}, 2.0f, Fade(colors[stripe], 0.62f));
    }
    DrawRectangleRoundedLinesEx(art, 0.12f, 4, 1.2f, Fade(accent, 0.82f));

    DrawRectangleRounded((Rectangle){r.x + 4, r.y + 3, r.width - 8, 16}, 0.2f, 2, rarity_color);
    if (in_shop) {
        DrawText(rarity_name, (int)(r.x + 6), (int)(r.y + 5), 10, BALATRO_WHITE);
        char price[16];
        snprintf(price, sizeof(price), "$%d", cost);
        Rectangle price_rect = {r.x + r.width - 27, r.y + 3, 24, 16};
        DrawRectangleRounded(price_rect, 0.3f, 2, BALATRO_GOLD);
        DrawText(price, (int)(price_rect.x + price_rect.width * 0.5f - MeasureText(price, 10) * 0.5f),
                 (int)(price_rect.y + 3), 10, BALATRO_DARK_TEXT);
    } else {
        DrawText(rarity_name, (int)(r.x + r.width * 0.5f - (MeasureText(rarity_name, 10) / 2)), (int)(r.y + 5), 10, BALATRO_WHITE);
    }

    DrawRectangleRounded((Rectangle){r.x + 5, r.y + 21, r.width - 10, 19}, 0.18f, 3,
                         card->edition == EDITION_NEGATIVE ? (Color){18, 17, 31, 235} : (Color){247, 246, 237, 235});
    const char *name = get_center_name(card->center_id);
    int font_size = 10;
    if (r.width < 55.0f) font_size = 8;
    else if (MeasureText(name, font_size) > (int)(r.width - 6) * 2) font_size = 8;
    draw_centered_multiline_text(name, (int)(r.x + r.width * 0.5f), (int)(r.y + 21), (int)(r.width - 6), 2, font_size, card->edition == EDITION_NEGATIVE ? BALATRO_WHITE : BALATRO_DARK_TEXT);

    DrawRectangleRoundedLinesEx((Rectangle){r.x + 2, r.y + 2, r.width - 4, r.height - 4},
                                0.07f, 4, 0.8f, Fade(rarity_color, 0.48f));
    DrawRectangleRoundedLinesEx(r, 0.08f, 4, hovered ? 3.0f : 2.0f, hovered ? BALATRO_GOLD : rarity_color);

    /* Debuffed overlay (Crimson Heart disables a joker each hand) */
    if (card->flags & CARD_DEBUFFED) {
        DrawRectangleRounded(r, 0.08f, 4, Fade(BLACK, 0.5f));
        DrawLineEx((Vector2){r.x + 4, r.y + 4}, (Vector2){r.x + r.width - 4, r.y + r.height - 4}, 2.5f, BALATRO_RED_MULT);
        DrawRectangle((int)r.x, (int)(r.y + r.height * 0.5f - 7), (int)r.width, 14, BALATRO_RED_MULT);
        const char *debuff_text = "DEBUFFED";
        DrawText(debuff_text, (int)(r.x + r.width * 0.5f - MeasureText(debuff_text, 9) * 0.5f),
                 (int)(r.y + r.height * 0.5f - 5), 9, BALATRO_WHITE);
    }
}

/* Draws a procedural Consumable (Tarot, Planet, Spectral) */
static void draw_consumable_card(const Card *card, Rectangle r, bool hovered, int index, bool in_shop, int cost) {
    if (hovered) {
        r.y -= 6.0f;
    }

    const CenterDefinition *def = &centers[card->center_id];
    Color body_color = BALATRO_COLOR_TAROT;
    const char *set_name = "TAROT";
    if (def->set == SET_PLANET) {
        body_color = BALATRO_COLOR_PLANET;
        set_name = "PLANET";
    } else if (def->set == SET_SPECTRAL) {
        body_color = BALATRO_COLOR_SPECTRAL;
        set_name = "SPECTRAL";
    }

    if (def->set == SET_SPECTRAL) {
        DrawRectangleRounded((Rectangle){r.x + 3, r.y + 4, r.width, r.height}, 0.08f, 4,
                             Fade(BLACK, 0.48f));
        DrawRectangleRounded(r, 0.08f, 4, (Color){13, 18, 48, 255});
        DrawRectangleRounded((Rectangle){r.x + 3, r.y + 3, r.width - 6, r.height - 6},
                             0.07f, 4, (Color){31, 36, 82, 255});
        for (int star = 0; star < 9; ++star) {
            float sx = r.x + 8.0f + fmodf((float)(star * 29 + card->center_id * 7), r.width - 16.0f);
            float sy = r.y + 23.0f + fmodf((float)(star * 37 + card->center_id * 11), r.height - 48.0f);
            float radius = star % 3 == 0 ? 1.2f : 0.7f;
            DrawCircleV((Vector2){sx, sy}, radius,
                        star & 1 ? Fade((Color){156, 255, 244, 255}, 0.58f)
                                 : Fade((Color){208, 151, 255, 255}, 0.62f));
        }
        Color border = hovered ? BALATRO_GOLD : (Color){126, 245, 232, 255};
        DrawRectangleRoundedLinesEx(r, 0.08f, 4, hovered ? 3.0f : 2.0f, border);
        DrawRectangleRoundedLinesEx((Rectangle){r.x + 3, r.y + 3, r.width - 6, r.height - 6},
                                    0.07f, 4, 1.0f, Fade((Color){206, 132, 255, 255}, 0.65f));
    } else {
        DrawRectangleRounded(r, 0.08f, 4, body_color);
        DrawRectangleRoundedLinesEx(r, 0.08f, 4, hovered ? 2.5f : 1.5f,
                                    hovered ? BALATRO_GOLD : BALATRO_WHITE);
    }

    /* Set Header */
    DrawRectangleRounded((Rectangle){r.x + 4, r.y + 4, r.width - 8, 14}, 0.2f, 2,
                         def->set == SET_SPECTRAL ? (Color){12, 54, 72, 225} : Fade(BLACK, 0.35f));
    DrawText(set_name, (int)(r.x + r.width * 0.5f - (MeasureText(set_name, 9) / 2)), (int)(r.y + 6), 9, BALATRO_WHITE);

    /* Central Motif first, so the name renders in front */
    float cx = r.x + r.width * 0.5f;
    float cy = r.y + r.height * 0.52f;
    if (def->set == SET_PLANET) {
        DrawCircle((int)cx, (int)cy, 12, (Color){180, 220, 255, 255});
        DrawEllipseLines((int)cx, (int)cy, 18, 6, BALATRO_GOLD);
    } else if (def->set == SET_TAROT) {
        DrawCircleV((Vector2){cx, cy}, 15, Fade((Color){28, 12, 54, 255}, 0.8f));
        for (int ray = 0; ray < 8; ++ray) {
            float angle = ray * PI * 0.25f;
            DrawLineEx((Vector2){cx + cosf(angle) * 11.0f, cy + sinf(angle) * 11.0f},
                       (Vector2){cx + cosf(angle) * 16.0f, cy + sinf(angle) * 16.0f}, 1.5f, BALATRO_GOLD);
        }
        DrawEllipse((int)cx, (int)cy, 12, 6, BALATRO_CARD_WHITE);
        DrawEllipseLines((int)cx, (int)cy, 12, 6, BALATRO_GOLD);
        DrawCircleV((Vector2){cx, cy}, 4, (Color){92, 45, 155, 255});
        DrawCircleV((Vector2){cx + 1, cy - 1}, 1.5f, BALATRO_WHITE);
    } else {
        float pulse = 0.88f + 0.12f * sinf(g_balatro_ui_time * 2.4f + card->center_id);
        Color cyan = (Color){113, 255, 237, 255};
        Color violet = (Color){205, 126, 255, 255};
        DrawCircleGradient((int)cx, (int)cy, 22.0f * pulse,
                           Fade(cyan, 0.24f), Fade((Color){20, 17, 61, 255}, 0.0f));
        switch (card->center_id) {
        case CENTER_C_SOUL: {
            for (int ray = 0; ray < 8; ++ray) {
                float angle = ray * PI * 0.25f + PI * 0.125f;
                DrawLineEx((Vector2){cx + cosf(angle) * 14.0f, cy + sinf(angle) * 14.0f},
                           (Vector2){cx + cosf(angle) * 21.0f, cy + sinf(angle) * 21.0f},
                           1.8f, ray & 1 ? violet : cyan);
            }
            DrawPoly((Vector2){cx, cy}, 4, 18.0f, 45.0f, Fade(violet, 0.72f));
            DrawPoly((Vector2){cx, cy}, 4, 14.0f, 45.0f, (Color){22, 30, 76, 255});
            DrawEllipse((int)cx, (int)cy, 12, 7, (Color){225, 255, 249, 255});
            DrawEllipseLines((int)cx, (int)cy, 12, 7, cyan);
            DrawCircleV((Vector2){cx, cy}, 5.0f, violet);
            DrawCircleV((Vector2){cx + 1.5f, cy - 1.5f}, 1.8f, BALATRO_WHITE);
            break;
        }
        case CENTER_C_BLACK_HOLE:
            DrawCircleV((Vector2){cx, cy}, 12.0f, (Color){2, 3, 9, 255});
            DrawEllipseLines((int)cx, (int)cy, 23, 8, cyan);
            DrawEllipseLines((int)cx, (int)cy, 18, 13, violet);
            DrawCircleV((Vector2){cx - 3, cy - 4}, 2.0f, Fade(BALATRO_WHITE, 0.7f));
            break;
        case CENTER_C_IMMOLATE:
            DrawPoly((Vector2){cx, cy + 4}, 3, 18.0f, 0, (Color){255, 102, 70, 255});
            DrawPoly((Vector2){cx, cy + 7}, 3, 12.0f, 0, BALATRO_GOLD);
            DrawCircleV((Vector2){cx, cy + 8}, 5.0f, (Color){255, 239, 158, 255});
            break;
        case CENTER_C_ANKH:
            DrawEllipseLines((int)cx, (int)(cy - 9), 8, 10, cyan);
            DrawLineEx((Vector2){cx, cy}, (Vector2){cx, cy + 18}, 3.0f, violet);
            DrawLineEx((Vector2){cx - 9, cy + 8}, (Vector2){cx + 9, cy + 8}, 3.0f, violet);
            break;
        case CENTER_C_FAMILIAR:
        case CENTER_C_GRIM:
        case CENTER_C_INCANTATION:
            DrawCircleV((Vector2){cx, cy - 2}, 13.0f, Fade(violet, 0.78f));
            DrawCircleV((Vector2){cx - 5, cy - 4}, 3.0f, (Color){10, 18, 43, 255});
            DrawCircleV((Vector2){cx + 5, cy - 4}, 3.0f, (Color){10, 18, 43, 255});
            DrawTriangle((Vector2){cx, cy}, (Vector2){cx - 3, cy + 5},
                         (Vector2){cx + 3, cy + 5}, (Color){10, 18, 43, 255});
            for (int tooth = -2; tooth <= 2; ++tooth)
                DrawLineEx((Vector2){cx + tooth * 3.0f, cy + 8},
                           (Vector2){cx + tooth * 3.0f, cy + 13}, 1.4f, cyan);
            break;
        case CENTER_C_TALISMAN:
        case CENTER_C_DEJA_VU:
        case CENTER_C_TRANCE:
        case CENTER_C_MEDIUM: {
            Color seal = card->center_id == CENTER_C_TALISMAN ? BALATRO_GOLD
                : card->center_id == CENTER_C_DEJA_VU ? BALATRO_HEART_RED
                : card->center_id == CENTER_C_TRANCE ? BALATRO_BLUE_CHIPS : violet;
            DrawCircleV((Vector2){cx, cy}, 15.0f, Fade(seal, 0.35f));
            DrawCircleLines((int)cx, (int)cy, 14.0f, seal);
            DrawPoly((Vector2){cx, cy}, 6, 9.0f, 30.0f, seal);
            DrawCircleV((Vector2){cx, cy}, 3.0f, (Color){20, 23, 57, 255});
            break;
        }
        case CENTER_C_WRAITH:
            DrawCircleV((Vector2){cx, cy - 7}, 9.0f, Fade(cyan, 0.72f));
            DrawTriangle((Vector2){cx - 14, cy + 17}, (Vector2){cx, cy - 8},
                         (Vector2){cx + 14, cy + 17}, Fade(violet, 0.78f));
            DrawCircleV((Vector2){cx - 3, cy - 7}, 1.5f, BALATRO_WHITE);
            DrawCircleV((Vector2){cx + 3, cy - 7}, 1.5f, BALATRO_WHITE);
            break;
        case CENTER_C_CRYPTID:
            DrawRectangleRounded((Rectangle){cx - 15, cy - 14, 17, 25}, 0.12f, 2,
                                 Fade(cyan, 0.62f));
            DrawRectangleRoundedLinesEx((Rectangle){cx - 15, cy - 14, 17, 25}, 0.12f, 2, 1.2f, cyan);
            DrawRectangleRounded((Rectangle){cx - 2, cy - 10, 17, 25}, 0.12f, 2,
                                 Fade(violet, 0.62f));
            DrawRectangleRoundedLinesEx((Rectangle){cx - 2, cy - 10, 17, 25}, 0.12f, 2, 1.2f, violet);
            break;
        default:
            DrawCircleV((Vector2){cx, cy}, 14.0f, Fade(violet, 0.42f));
            DrawCircleLines((int)cx, (int)cy, 13.0f, cyan);
            DrawPoly((Vector2){cx, cy}, 6, 8.0f, 30.0f, violet);
            DrawCircleV((Vector2){cx, cy}, 3.0f, BALATRO_WHITE);
            break;
        }
    }

    /* Name with Multi-line Wrapping, rendered over the motif */
    const char *name = get_center_name(card->center_id);
    int font_size = 9;
    if (r.width < 55.0f) font_size = 8;
    draw_centered_multiline_text(name, (int)(r.x + r.width * 0.5f), (int)(r.y + 20), (int)(r.width - 6), 2, font_size, BALATRO_GOLD);

    /* Price / Action label */
    char price_buf[16];
    snprintf(price_buf, sizeof(price_buf), "$%d", in_shop ? cost : card->sell_cost);
    DrawRectangleRounded((Rectangle){r.x + 8, r.y + r.height - 18, r.width - 16, 14}, 0.3f, 2, in_shop ? BALATRO_GOLD : Fade(BLACK, 0.4f));
    DrawText(price_buf, (int)(r.x + r.width * 0.5f - (MeasureText(price_buf, 10) / 2)), (int)(r.y + r.height - 16), 10, in_shop ? BALATRO_DARK_TEXT : BALATRO_WHITE);
}

/* ========================================================================= */
/* Screen & HUD Renderers                                                    */
/* ========================================================================= */

/* Draws the Left Sidebar with Ante, Blind Chips, Hands/Discards, Money, and Hand preview */
static void draw_hud_sidebar(const State *state, const LegalMasks *legal, Rectangle rect, int selected_count, const uint8_t *selected_indices, Action *out_action, Vector2 mouse, Client *client) {
    DrawRectangleRec(rect, BALATRO_PANEL_BG);
    DrawRectangleLinesEx(rect, 2.0f, BALATRO_PANEL_BORDER);

    float y = rect.y + 12.0f;
    float x = rect.x + 12.0f;
    float w = rect.width - 24.0f;

    /* Ante & Stake */
    DrawRectangleRounded((Rectangle){x, y, w, 28}, 0.2f, 2, BALATRO_HEADER_BG);
    DrawText(TextFormat("ANTE %d / %d", state->ante, state->config.win_ante ? state->config.win_ante : 8), (int)(x + 10), (int)(y + 7), 15, BALATRO_GOLD);
    static const char *stake_names[8] = {
        "WHITE STAKE", "RED STAKE", "GREEN STAKE", "BLACK STAKE",
        "BLUE STAKE", "PURPLE STAKE", "ORANGE STAKE", "GOLD STAKE"
    };
    assert(state->config.stake >= 1 && state->config.stake <= 8);
    const char *stake_name = stake_names[state->config.stake - 1];
    DrawText(stake_name, (int)(x + w - MeasureText(stake_name, 10) - 10), (int)(y + 9), 10, BALATRO_WHITE);
    y += 36.0f;

    /* Current Blind Box */
    DrawRectangleRounded((Rectangle){x, y, w, 82}, 0.15f, 2, BALATRO_HEADER_BG);
    DrawRectangleRoundedLinesEx((Rectangle){x, y, w, 82}, 0.15f, 2, 1.5f, BALATRO_PANEL_BORDER);

    uint16_t sidebar_blind_id = state->phase == PHASE_BLIND_SELECT
        ? (state->blind_on_deck == 0 ? BLIND_BL_SMALL
           : state->blind_on_deck == 1 ? BLIND_BL_BIG : state->next_boss_id)
        : state->blind_id;
    const char *blind_title = get_blind_name(sidebar_blind_id);
    DrawText(blind_title, (int)(x + 10), (int)(y + 8), 16, BALATRO_WHITE);

    /* Target score & Progress bar */
    if (state->phase == PHASE_BLIND_SELECT) {
        DrawText("CHOOSE NEXT BLIND", (int)(x + 10), (int)(y + 34), 12, BALATRO_BLUE_CHIPS);
    } else {
        char score_buf[64];
        snprintf(score_buf, sizeof(score_buf), "Score: %.0f / %.0f", state->chips, state->blind_chips);
        DrawText(score_buf, (int)(x + 10), (int)(y + 30), 13, BALATRO_BLUE_CHIPS);

        /* Progress bar fill */
        DrawRectangle((int)(x + 10), (int)(y + 50), (int)(w - 20), 12, (Color){10, 20, 18, 255});
        float progress = (state->blind_chips > 0) ? (float)(state->chips / state->blind_chips) : 0.0f;
        if (progress > 1.0f) progress = 1.0f;
        DrawRectangle((int)(x + 10), (int)(y + 50), (int)((w - 20) * progress), 12, BALATRO_BLUE_CHIPS);
        DrawRectangleLines((int)(x + 10), (int)(y + 50), (int)(w - 20), 12, BALATRO_PANEL_BORDER);
    }

    y += 92.0f;

    /* Hands Left & Discards Left boxes. In the shop (and pack opening) the
       round counters are spent; show the upcoming round's max instead, and
       keep money/deck live as purchases change them. */
    float box_w = (w - 10) * 0.5f;
    int shop_view = state->phase == PHASE_SHOP || state->phase == PHASE_PACK_OPENING;
    int hands_val = shop_view ? state->hands_per_round : state->hands_left;
    int discards_val = shop_view ? state->discards_per_round : state->discards_left;
    const char *hands_lbl = "HANDS";
    const char *disc_lbl = "DISCARDS";

    /* Hands */
    DrawRectangleRounded((Rectangle){x, y, box_w, 54}, 0.2f, 2, BALATRO_HEADER_BG);
    DrawText(hands_lbl, (int)(x + 10), (int)(y + 6), 11, (Color){180, 200, 220, 255});
    DrawText(TextFormat("%d", hands_val), (int)(x + 10), (int)(y + 20), 26, BALATRO_BLUE_CHIPS);

    /* Discards */
    DrawRectangleRounded((Rectangle){x + box_w + 10, y, box_w, 54}, 0.2f, 2, BALATRO_HEADER_BG);
    DrawText(disc_lbl, (int)(x + box_w + 20), (int)(y + 6), 11, (Color){220, 180, 180, 255});
    DrawText(TextFormat("%d", discards_val), (int)(x + box_w + 20), (int)(y + 20), 26, BALATRO_RED_MULT);

    y += 64.0f;

    /* Dollars ($) & Deck count */
    DrawRectangleRounded((Rectangle){x, y, box_w, 46}, 0.2f, 2, BALATRO_HEADER_BG);
    DrawText("MONEY", (int)(x + 10), (int)(y + 6), 10, (Color){200, 190, 150, 255});
    DrawText(TextFormat("$%d", state->dollars), (int)(x + 10), (int)(y + 20), 20, BALATRO_GOLD);

    DrawRectangleRounded((Rectangle){x + box_w + 10, y, box_w, 46}, 0.2f, 2, BALATRO_HEADER_BG);
    DrawText("DECK", (int)(x + box_w + 20), (int)(y + 6), 10, (Color){180, 190, 200, 255});
    /* Headline: total cards remaining (updates immediately on destroy/create);
       the draw-pile count is the sub-line. */
    DrawText(TextFormat("%d", state->deck_count + state->discard_count + state->hand_count),
             (int)(x + box_w + 20), (int)(y + 16), 18, BALATRO_WHITE);
    DrawText(TextFormat("in deck: %d", state->deck_count),
             (int)(x + box_w + 20), (int)(y + 35), 9, (Color){140, 155, 165, 255});

    y += 56.0f;

    /* Persistent hand evaluator: keep sidebar geometry fixed in every phase. */
    DrawRectangleRounded((Rectangle){x, y, w, 110}, 0.15f, 2, BALATRO_HEADER_BG);
    DrawRectangleRoundedLinesEx((Rectangle){x, y, w, 110}, 0.15f, 2, 1.5f, BALATRO_PANEL_BORDER);

    const char *hand_name = "";
    const char *status = "";
    Color hand_color = (Color){150, 160, 170, 255};
    int level = 0;
    int preview_chips = 0;
    int preview_mult = 0;
    char status_buf[64] = {0};
    if (client->score_anim.active) {
        const ScoreAnim *score = &client->score_anim;
        hand_name = get_hand_type_name(score->hand_type);
        level = score->level;
        preview_chips = score->chips;
        preview_mult = score->mult;
        if (score->hand_not_allowed) {
            snprintf(status_buf, sizeof(status_buf), "NOT ALLOWED - locked to %s",
                     get_hand_type_name(score->allowed_hand_type));
            hand_color = BALATRO_RED_MULT;
        } else {
            snprintf(status_buf, sizeof(status_buf), "Hand Score: %.0f", score->total);
            hand_color = BALATRO_GOLD;
        }
        status = status_buf;
    } else if (state->phase == PHASE_SELECTING_HAND && selected_count > 0) {
        Card played[MAX_SELECTION];
        int has_facedown = 0;
        for (int i = 0; i < selected_count; ++i) {
            assert(selected_indices[i] < state->hand_count);
            played[i] = state->hand[selected_indices[i]];
            has_facedown |= (played[i].flags & CARD_FACEDOWN) != 0;
        }
        if (has_facedown) {
            hand_name = "FACE-DOWN CARDS";
            status = "Cards reveal when played";
            hand_color = BALATRO_GOLD;
        } else {
            uint8_t scoring_mask = 0;
            HandType evaluated = classify_hand(
                played, (size_t)selected_count, &scoring_mask,
                joker_active(state, CENTER_J_FOUR_FINGERS),
                joker_active(state, CENTER_J_SHORTCUT),
                joker_active(state, CENTER_J_SMEARED));
            hand_name = get_hand_type_name(evaluated);
            level = state->hand_levels[evaluated] ? state->hand_levels[evaluated] : 1;
            int mouth_rejects = !state->blind_disabled && state->blind_id == BLIND_BL_MOUTH &&
                state->blind_only_hand != UINT8_MAX && state->blind_only_hand != (uint8_t)evaluated;
            if (mouth_rejects) {
                snprintf(status_buf, sizeof(status_buf), "NOT ALLOWED - locked to %s",
                         get_hand_type_name((HandType)state->blind_only_hand));
                status = status_buf;
                hand_color = BALATRO_RED_MULT;
            } else {
                hand_base_stats(evaluated, level, &preview_chips, &preview_mult);
                snprintf(status_buf, sizeof(status_buf), "Base Score: %d", preview_chips * preview_mult);
                status = status_buf;
                hand_color = BALATRO_GOLD;
            }
        }
    }

    if (hand_name[0])
        DrawText(hand_name, (int)(x + 10), (int)(y + 8), 16, hand_color);
    if (level)
        DrawText(TextFormat("Level %d", level), (int)(x + w - 60), (int)(y + 10), 11, BALATRO_WHITE);

    float score_gap = 24.0f;
    float score_w = (w - 20.0f - score_gap) * 0.5f;
    Rectangle chips_box = {x + 10, y + 36, score_w, 28};
    Rectangle mult_box = {chips_box.x + chips_box.width + score_gap, y + 36, score_w, 28};
    DrawRectangleRounded(chips_box, 0.2f, 2, BALATRO_BLUE_CHIPS);
    const char *chips_text = TextFormat("%d Chips", preview_chips);
    DrawText(chips_text, (int)(chips_box.x + (chips_box.width - MeasureText(chips_text, 14)) * 0.5f),
             (int)(chips_box.y + 7), 14, BALATRO_WHITE);

    DrawText("X", (int)(chips_box.x + chips_box.width +
             (score_gap - MeasureText("X", 16)) * 0.5f), (int)(y + 42), 16, BALATRO_GOLD);

    DrawRectangleRounded(mult_box, 0.2f, 2, BALATRO_RED_MULT);
    const char *mult_text = TextFormat("%d Mult", preview_mult);
    DrawText(mult_text, (int)(mult_box.x + (mult_box.width - MeasureText(mult_text, 14)) * 0.5f),
             (int)(mult_box.y + 7), 14, BALATRO_WHITE);
    if (status[0])
        DrawText(status, (int)(x + 12), (int)(y + 76), 12, (Color){180, 200, 210, 255});
    y += 120.0f;

    Rectangle boss_panel = {x, y, w, 102};
    DrawRectangleRounded(boss_panel, 0.15f, 2, (Color){24, 39, 37, 255});
    DrawRectangleRoundedLinesEx(boss_panel, 0.15f, 2, 1.5f, Fade(BALATRO_RED_MULT, 0.45f));
    DrawText("BOSS THIS ANTE", (int)(x + 10), (int)(y + 8), 10, (Color){185, 145, 145, 255});
    const char *boss_name = get_blind_name(state->next_boss_id);
    DrawText(boss_name, (int)(x + 10), (int)(y + 25), 16, BALATRO_WHITE);
    draw_centered_multiline_text(get_blind_desc(state->next_boss_id), (int)(x + w * 0.5f),
                                 (int)(y + 52), (int)(w - 20), 2, 10,
                                 (Color){145, 175, 170, 255});
    y += 112.0f;

    uint8_t tag_ids[12];
    uint8_t tag_counts[12];
    int tag_count = 0;
    if (state->tag_force_rarity_count) {
        tag_ids[tag_count] = state->tag_force_rarity == 3 ? TAG_TAG_RARE : TAG_TAG_UNCOMMON;
        tag_counts[tag_count] = state->tag_force_rarity_count;
        tag_count++;
    }
    if (state->tag_force_edition_count) {
        assert(state->tag_force_edition >= EDITION_FOIL && state->tag_force_edition <= EDITION_NEGATIVE);
        static const uint8_t edition_tags[] = {
            TAG_NONE, TAG_TAG_FOIL, TAG_TAG_HOLO, TAG_TAG_POLYCHROME, TAG_TAG_NEGATIVE
        };
        tag_ids[tag_count] = edition_tags[state->tag_force_edition];
        tag_counts[tag_count] = state->tag_force_edition_count;
        tag_count++;
    }
    if (state->tag_investment_pending) {
        tag_ids[tag_count] = TAG_TAG_INVESTMENT;
        tag_counts[tag_count] = state->tag_investment_pending;
        tag_count++;
    }
    if (state->tag_voucher_pending) {
        tag_ids[tag_count] = TAG_TAG_VOUCHER;
        tag_counts[tag_count] = state->tag_voucher_pending;
        tag_count++;
    }
    if (state->tag_coupon_pending || state->tag_coupon_active) {
        tag_ids[tag_count] = TAG_TAG_COUPON;
        tag_counts[tag_count] = 1;
        tag_count++;
    }
    if (state->double_tag) {
        tag_ids[tag_count] = TAG_TAG_DOUBLE;
        tag_counts[tag_count] = 1;
        tag_count++;
    }
    if (state->tag_hand_bonus) {
        tag_ids[tag_count] = TAG_TAG_JUGGLE;
        tag_counts[tag_count] = state->tag_hand_bonus / 3;
        tag_count++;
    }
    if (state->tag_d_six_pending || state->tag_d_six_active) {
        tag_ids[tag_count] = TAG_TAG_D_SIX;
        tag_counts[tag_count] = 1;
        tag_count++;
    }

    Rectangle tag_panel = {x, y, w, 90};
    DrawRectangleRounded(tag_panel, 0.15f, 2, (Color){20, 37, 35, 255});
    DrawRectangleRoundedLinesEx(tag_panel, 0.15f, 2, 1.2f, BALATRO_PANEL_BORDER);
    DrawText("OWNED TAGS", (int)(x + 10), (int)(y + 8), 10, BALATRO_COLOR_TAROT);
    int visible_tags = 2;
    if (visible_tags > tag_count) visible_tags = tag_count;
    for (int i = 0; i < visible_tags; ++i) {
        Rectangle tag_rect = {x + 8, y + 25.0f + i * 28.0f, w - 16, 23};
        bool hovered = CheckCollisionPointRec(mouse, tag_rect);
        DrawRectangleRounded(tag_rect, 0.25f, 2,
            hovered ? (Color){58, 67, 83, 255} : (Color){32, 51, 50, 255});
        const char *tag_name = get_tag_name(tag_ids[i]);
        DrawText(tag_name, (int)(tag_rect.x + 8), (int)(tag_rect.y + 5), 11, BALATRO_WHITE);
        if (tag_counts[i] > 1) {
            const char *count_text = TextFormat("x%d", tag_counts[i]);
            DrawText(count_text, (int)(tag_rect.x + tag_rect.width - MeasureText(count_text, 10) - 7),
                     (int)(tag_rect.y + 6), 10, BALATRO_GOLD);
        }
        if (hovered) {
            client->show_tooltip = true;
            snprintf(client->tooltip_title, sizeof(client->tooltip_title), "%s", tag_name);
            snprintf(client->tooltip_type, sizeof(client->tooltip_type), "Owned tag effect");
            snprintf(client->tooltip_desc, sizeof(client->tooltip_desc), "%s", get_tag_description(tag_ids[i]));
            if (tag_counts[i] > 1)
                snprintf(client->tooltip_stats, sizeof(client->tooltip_stats),
                         "%d copies pending", tag_counts[i]);
            else
                snprintf(client->tooltip_stats, sizeof(client->tooltip_stats), "1 effect");
            client->tooltip_pos = (Vector2){mouse.x + 16, mouse.y + 16};
        }
    }
    if (!tag_count) {
        DrawText("No owned tags", (int)(x + 10), (int)(y + 34), 10,
                 (Color){110, 135, 132, 255});
    } else if (visible_tags < tag_count) {
        const char *more_text = TextFormat("+%d more", tag_count - visible_tags);
        DrawText(more_text, (int)(x + w - MeasureText(more_text, 10) - 10),
                 (int)(tag_panel.y + tag_panel.height - 15), 10, (Color){140, 160, 158, 255});
    }

    float menu_gap = 6.0f;
    float menu_button_w = (w - menu_gap) * 0.5f;
    Rectangle info_button = {x, rect.y + rect.height - 38, menu_button_w, 28};
    Rectangle controls_button = {x + menu_button_w + menu_gap, info_button.y, menu_button_w, 28};
    bool info_hovered = CheckCollisionPointRec(mouse, info_button);
    bool controls_hovered = CheckCollisionPointRec(mouse, controls_button);
    DrawRectangleRounded(info_button, 0.25f, 2,
        info_hovered ? (Color){47, 78, 74, 255} : BALATRO_HEADER_BG);
    DrawRectangleRounded(controls_button, 0.25f, 2,
        controls_hovered ? (Color){47, 78, 74, 255} : BALATRO_HEADER_BG);
    DrawRectangleRoundedLinesEx(info_button, 0.25f, 2, 1.2f, BALATRO_PANEL_BORDER);
    DrawRectangleRoundedLinesEx(controls_button, 0.25f, 2, 1.2f, BALATRO_PANEL_BORDER);
    const char *info_text = "RUN INFO";
    const char *controls_text = "CONTROLS";
    DrawText(info_text, (int)(info_button.x + (info_button.width - MeasureText(info_text, 10)) * 0.5f),
             (int)(info_button.y + 9), 10, BALATRO_WHITE);
    DrawText(controls_text, (int)(controls_button.x + (controls_button.width - MeasureText(controls_text, 10)) * 0.5f),
             (int)(controls_button.y + 9), 10, BALATRO_WHITE);
    if (info_hovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        client->show_run_menu = true;
        client->show_controls_menu = false;
    }
    if (controls_hovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        client->show_run_menu = true;
        client->show_controls_menu = true;
    }
}

/* Draws Jokers Bar at top with adaptive flex layout for any count of jokers (>5) */
static void draw_jokers_bar(const State *state, Rectangle rect, Action *out_action, Vector2 mouse, Client *client) {
    DrawRectangleRec(rect, BALATRO_HEADER_BG);
    DrawRectangleLinesEx(rect, 1.5f, BALATRO_PANEL_BORDER);

    /* Slot Header */
    char header_buf[64];
    snprintf(header_buf, sizeof(header_buf), "JOKERS (%d / %d)", state->joker_count, state->joker_slots);
    DrawText(header_buf, (int)(rect.x + 10), (int)(rect.y + 4), 12, BALATRO_GOLD);

    if (state->joker_count == 0) {
        DrawText("No Jokers owned - Purchase from Shop", (int)(rect.x + 20), (int)(rect.y + rect.height * 0.5f - 6), 12, (Color){140, 150, 160, 255});
        return;
    }

    /* Adaptive spacing */
    float card_w = 72.0f;
    float card_h = 96.0f;
    float avail_w = rect.width - 24.0f;
    float pitch = card_w + 8.0f;

    if (state->joker_count * pitch > avail_w && state->joker_count > 1) {
        pitch = (avail_w - card_w) / (float)(state->joker_count - 1);
        if (pitch < 32.0f) pitch = 32.0f;
    }

    for (int i = 0; i < state->joker_count; ++i) {
        float cx = rect.x + 12.0f + i * pitch;
        float cy = rect.y + 22.0f;
        Rectangle card_r = {cx, cy, card_w, card_h};
        bool hovered = CheckCollisionPointRec(mouse, card_r);

        int selected_count = state->phase == PHASE_SELECTING_HAND ? client->selected_count : 0;
        const uint8_t *selected_indices = selected_count ? client->selected_indices : NULL;
        draw_joker_card(&state->jokers[i], card_r, hovered, false, 0, client->puffer);

        if (hovered) {
            client->show_tooltip = true;
            snprintf(client->tooltip_title, sizeof(client->tooltip_title), "%s", get_center_name(state->jokers[i].center_id));
            snprintf(client->tooltip_type, sizeof(client->tooltip_type), "%sJoker | %s",
                     (state->jokers[i].flags & CARD_DEBUFFED) ? "DEBUFFED | " : "",
                     get_edition_name(state->jokers[i].edition));
            get_center_description(&state->jokers[i], state, client->tooltip_desc, sizeof(client->tooltip_desc));
            char value[40];
            if (state->jokers[i].center_id == CENTER_J_RAISED_FIST)
                get_raised_fist_value(state, selected_count, selected_indices, value, sizeof(value));
            else
                get_joker_value(&state->jokers[i], state, value, sizeof(value));
            snprintf(client->tooltip_stats, sizeof(client->tooltip_stats), "%s | Sell $%d | Slot %d of %d",
                     value, state->jokers[i].sell_cost, i + 1, state->joker_count);
            client->tooltip_pos = (Vector2){card_r.x + card_r.width * 0.5f - 150.0f,
                                            card_r.y + card_r.height + 10.0f};

            /* Right-click or Shift-click to sell joker */
            if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) || (IsKeyDown(KEY_LEFT_SHIFT) && IsMouseButtonPressed(MOUSE_BUTTON_LEFT))) {
                out_action->type = ACTION_SELL_JOKER;
                out_action->primary = (uint8_t)i;
            }
            if (i > 0 && IsKeyPressed(KEY_LEFT)) {
                out_action->type = ACTION_SWAP_JOKERS_LEFT;
                out_action->primary = (uint8_t)i;
            }
            if (i + 1 < state->joker_count && IsKeyPressed(KEY_RIGHT)) {
                out_action->type = ACTION_SWAP_JOKERS_RIGHT;
                out_action->primary = (uint8_t)i;
            }
        }
    }
}

/* Draws Consumables Bar at top with adaptive flex layout for any count (>2) */
static void draw_consumables_bar(const State *state, Rectangle rect, Action *out_action, Vector2 mouse, Client *client) {
    DrawRectangleRec(rect, BALATRO_HEADER_BG);
    DrawRectangleLinesEx(rect, 1.5f, BALATRO_PANEL_BORDER);

    /* Slot Header */
    char header_buf[64];
    snprintf(header_buf, sizeof(header_buf), "CONSUMABLES (%d / %d)", state->consumable_count, state->consumable_slots);
    DrawText(header_buf, (int)(rect.x + 10), (int)(rect.y + 4), 12, BALATRO_COLOR_TAROT);

    if (state->consumable_count == 0) {
        DrawText("No Consumables", (int)(rect.x + 14), (int)(rect.y + rect.height * 0.5f - 6), 11, (Color){140, 150, 160, 255});
        return;
    }

    float card_w = 64.0f;
    float card_h = 92.0f;
    float avail_w = rect.width - 20.0f;
    float pitch = card_w + 8.0f;

    if (state->consumable_count * pitch > avail_w && state->consumable_count > 1) {
        pitch = (avail_w - card_w) / (float)(state->consumable_count - 1);
    }

    for (int i = 0; i < state->consumable_count; ++i) {
        float cx = rect.x + 10.0f + i * pitch;
        float cy = rect.y + 24.0f;
        Rectangle card_r = {cx, cy, card_w, card_h};
        bool hovered = CheckCollisionPointRec(mouse, card_r);

        draw_consumable_card(&state->consumables[i], card_r, hovered, i, false, 0);

        if (hovered) {
            client->show_tooltip = true;
            snprintf(client->tooltip_title, sizeof(client->tooltip_title), "%s", get_center_name(state->consumables[i].center_id));
            snprintf(client->tooltip_type, sizeof(client->tooltip_type), "Consumable");
            get_center_description(&state->consumables[i], state, client->tooltip_desc, sizeof(client->tooltip_desc));
            snprintf(client->tooltip_stats, sizeof(client->tooltip_stats), "Sell Value: $%d", state->consumables[i].sell_cost);
            client->tooltip_pos = (Vector2){mouse.x + 16, mouse.y + 16};

            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                out_action->type = ACTION_USE_CONSUMABLE;
                out_action->primary = (uint8_t)i;
                out_action->selection_count = (uint8_t)client->selected_count;
                for (int s = 0; s < client->selected_count; ++s) {
                    out_action->selection[s] = client->selected_indices[s];
                }
            } else if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
                out_action->type = ACTION_SELL_CONSUMABLE;
                out_action->primary = (uint8_t)i;
            }
        }
    }
}

/* Phase 0: Blind Select View */
static void draw_phase_blind_select(const State *state, const LegalMasks *legal, Rectangle rect, Action *out_action, Vector2 mouse, Client *client) {
    float card_w = 230.0f;
    float card_h = 320.0f;
    float start_x = rect.x + (rect.width - (3 * card_w + 2 * 30.0f)) * 0.5f;
    float start_y = rect.y + (rect.height - card_h) * 0.45f;

    const char *names[3] = {"Small Blind", "Big Blind", get_blind_name(state->next_boss_id)};
    const char *descriptions[3] = {
        get_blind_desc(BLIND_BL_SMALL), get_blind_desc(BLIND_BL_BIG),
        get_blind_desc(state->next_boss_id)
    };
    int rewards[3] = {state->config.stake >= 2 ? 0 : 3, 4,
        state->next_boss_id >= BLIND_BL_FINAL_ACORN &&
        state->next_boss_id <= BLIND_BL_FINAL_VESSEL ? 8 : 5};
    static const int32_t blind_amounts[3][8] = {
        {300, 800, 2000, 5000, 11000, 20000, 35000, 50000},
        {300, 900, 2600, 8000, 20000, 36000, 60000, 100000},
        {300, 1000, 3200, 9000, 25000, 60000, 110000, 200000},
    };
    double base_chips;
    assert(state->stake_scaling >= 1 && state->stake_scaling <= 3);
    if (state->ante <= 8) {
        base_chips = state->ante ? blind_amounts[state->stake_scaling - 1][state->ante - 1] : 100;
    } else {
        double c = (double)state->ante - 8.0;
        double d = 1.0 + 0.2 * c;
        double raw = floor(blind_amounts[state->stake_scaling - 1][7] *
            pow(1.6 + pow(0.75 * c, d), c));
        double rounding = pow(10.0, floor(log10(raw) - 1.0));
        base_chips = raw - fmod(raw, rounding);
    }
    if (state->config.deck == CENTER_B_PLASMA) base_chips *= 2.0;

    for (int i = 0; i < 3; ++i) {
        float bx = start_x + i * (card_w + 30.0f);
        Rectangle b_rect = {bx, start_y, card_w, card_h};
        bool is_current = (state->blind_on_deck == i);
        bool is_past = (state->blind_on_deck > i);

        DrawRectangleRounded(b_rect, 0.08f, 4, is_current ? BALATRO_HEADER_BG : (Color){18, 35, 30, 255});
        DrawRectangleRoundedLinesEx(b_rect, 0.08f, 4, is_current ? 3.0f : 1.5f, is_current ? BALATRO_GOLD : BALATRO_PANEL_BORDER);

        /* Blind title */
        DrawText(names[i], (int)(bx + card_w * 0.5f - (MeasureText(names[i], 18) / 2)), (int)(start_y + 16), 18, is_current ? BALATRO_GOLD : BALATRO_WHITE);

        /* Score requirement */
        double req_chips = base_chips * (i == 0 ? 1.0 : i == 1 ? 1.5 : 2.0);
        if (i == 2 && state->next_boss_id == BLIND_BL_NEEDLE) req_chips *= 0.5;
        if (i == 2 && state->next_boss_id == BLIND_BL_WALL) req_chips *= 2.0;
        if (i == 2 && state->next_boss_id == BLIND_BL_FINAL_VESSEL) req_chips *= 3.0;
        char req_buf[32];
        snprintf(req_buf, sizeof(req_buf), "Score: %.0f", req_chips);
        DrawText(req_buf, (int)(bx + card_w * 0.5f - (MeasureText(req_buf, 15) / 2)), (int)(start_y + 60), 15, BALATRO_BLUE_CHIPS);

        /* Reward */
        char rew_buf[32];
        snprintf(rew_buf, sizeof(rew_buf), "Reward: +$%d", rewards[i]);
        DrawText(rew_buf, (int)(bx + card_w * 0.5f - (MeasureText(rew_buf, 14) / 2)), (int)(start_y + 90), 14, BALATRO_GOLD);

        draw_centered_multiline_text(descriptions[i], (int)(bx + card_w * 0.5f),
                                     (int)(start_y + 116), (int)(card_w - 28), 2, 10,
                                     is_current ? (Color){175, 210, 200, 255}
                                                : (Color){125, 155, 150, 255});

        /* Tag info */
        if (i < 2 && state->blind_tags[i] != TAG_NONE) {
            const char *tag_name = get_tag_name(state->blind_tags[i]);
            float tag_y = start_y + 164.0f;
            DrawRectangleRounded((Rectangle){bx + 14, tag_y, card_w - 28, 48}, 0.2f, 2, (Color){30, 60, 52, 255});
            DrawText("SKIP REWARD", (int)(bx + card_w * 0.5f - MeasureText("SKIP REWARD", 10) * 0.5f),
                     (int)(tag_y + 6), 10, (Color){180, 200, 220, 255});
            DrawText(tag_name, (int)(bx + card_w * 0.5f - MeasureText(tag_name, 12) * 0.5f),
                     (int)(tag_y + 24), 12, BALATRO_GOLD);
            Rectangle tag_rect = {bx + 14, tag_y, card_w - 28, 48};
            if (CheckCollisionPointRec(mouse, tag_rect)) {
                DrawRectangleRoundedLinesEx(tag_rect, 0.2f, 2, 1.5f, BALATRO_GOLD);
                client->show_tooltip = true;
                snprintf(client->tooltip_title, sizeof(client->tooltip_title), "%s", tag_name);
                snprintf(client->tooltip_type, sizeof(client->tooltip_type), "Skip Blind reward");
                snprintf(client->tooltip_desc, sizeof(client->tooltip_desc), "%s",
                         get_tag_description(state->blind_tags[i]));
                snprintf(client->tooltip_stats, sizeof(client->tooltip_stats), "Skip this Blind to claim it");
                client->tooltip_pos = (Vector2){mouse.x + 16, mouse.y + 16};
            }
        }

        /* Buttons */
        if (is_current) {
            /* SELECT Button */
            Rectangle select_btn = {bx + 20, start_y + card_h - 75, card_w - 40, 32};
            bool sel_hov = CheckCollisionPointRec(mouse, select_btn);
            DrawRectangleRounded(select_btn, 0.25f, 2, sel_hov ? (Color){40, 200, 100, 255} : (Color){30, 160, 80, 255});
            DrawText("SELECT BLIND", (int)(select_btn.x + (select_btn.width - MeasureText("SELECT BLIND", 14)) * 0.5f),
                     (int)(select_btn.y + (select_btn.height - 14) * 0.5f), 14, BALATRO_WHITE);
            if (sel_hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                out_action->type = ACTION_SELECT_BLIND;
            }

            /* SKIP Button (if available) */
            if (i < 2) {
                Rectangle skip_btn = {bx + 20, start_y + card_h - 36, card_w - 40, 26};
                bool skip_hov = CheckCollisionPointRec(mouse, skip_btn);
                DrawRectangleRounded(skip_btn, 0.25f, 2, skip_hov ? (Color){180, 60, 60, 255} : (Color){140, 45, 45, 255});
                DrawText("SKIP BLIND", (int)(skip_btn.x + (skip_btn.width - MeasureText("SKIP BLIND", 12)) * 0.5f),
                         (int)(skip_btn.y + (skip_btn.height - 12) * 0.5f), 12, BALATRO_WHITE);
                if (skip_hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                    out_action->type = ACTION_SKIP_BLIND;
                }
            }
        } else if (is_past) {
            DrawText("DEFEATED", (int)(bx + card_w * 0.5f - MeasureText("DEFEATED", 16) * 0.5f),
                     (int)(start_y + card_h - 45), 16, (Color){100, 120, 110, 255});
        } else {
            DrawText("UPCOMING", (int)(bx + card_w * 0.5f - MeasureText("UPCOMING", 14) * 0.5f),
                     (int)(start_y + card_h - 45), 14, (Color){80, 100, 90, 255});
        }
    }
}

/* Phase 1 & 2: Hand Selection & Playing Area */
static void draw_phase_hand_play(State *state, const LegalMasks *legal, Rectangle rect, int *selected_count, uint8_t *selected_indices, Action *out_action, Vector2 mouse, Client *client) {
    float cx0 = rect.x + rect.width * 0.5f;
    float hint_y = rect.y + 22.0f;

    /* Selection hint */
    if (*selected_count == 0) {
        const char *hint = "Select 1-5 cards to score  |  1-9 to toggle, SPACE to play";
        DrawText(hint, (int)(cx0 - MeasureText(hint, 13) * 0.5f), (int)hint_y, 13, Fade((Color){170, 190, 210, 255}, 0.55f));
    } else {
        char hint_buf[48];
        snprintf(hint_buf, sizeof(hint_buf), "%d card(s) selected", *selected_count);
        DrawText(hint_buf, (int)(cx0 - MeasureText(hint_buf, 13) * 0.5f), (int)hint_y, 13, Fade(BALATRO_GOLD, 0.75f));
    }

    /* Dynamic hand cards spacing */
    int count = state->hand_count;
    if (count > 0) {
        float card_w = 74.0f;
        float card_h = 106.0f;
        float avail_w = rect.width - 40.0f;
        float pitch = card_w + 10.0f;

        if (count * pitch > avail_w && count > 1) {
            pitch = (avail_w - card_w) / (float)(count - 1);
        }

        float total_w = (count - 1) * pitch + card_w;
        float start_x = rect.x + (rect.width - total_w) * 0.5f;
        float start_y = rect.y + rect.height - 190.0f;

        for (int i = 0; i < count; ++i) {
            float cx = start_x + i * pitch;
            float cy = start_y;
            Rectangle card_r = {cx, cy, card_w, card_h};

            bool is_selected = false;
            for (int s = 0; s < *selected_count; ++s) {
                if (selected_indices[s] == i) {
                    is_selected = true;
                    break;
                }
            }

            bool hovered = CheckCollisionPointRec(mouse, (Rectangle){cx, is_selected ? cy - 22.0f : cy, card_w, card_h});

            draw_playing_card(&state->hand[i], card_r, is_selected, hovered, false, 0, client->puffer);

            if (hovered) {
                client->show_tooltip = true;
                if (state->hand[i].flags & CARD_FACEDOWN) {
                    snprintf(client->tooltip_title, sizeof(client->tooltip_title), "Face-down Card");
                    snprintf(client->tooltip_type, sizeof(client->tooltip_type), "Hidden until played");
                    snprintf(client->tooltip_desc, sizeof(client->tooltip_desc), "Rank, suit and enhancements unknown (Slot %d)", i + 1);
                    snprintf(client->tooltip_stats, sizeof(client->tooltip_stats), "Plays and scores normally");
                } else if (state->hand[i].enhancement == ENHANCEMENT_STONE) {
                    snprintf(client->tooltip_title, sizeof(client->tooltip_title), "Stone Card");
                    snprintf(client->tooltip_type, sizeof(client->tooltip_type), "Stone (+50 Chips, No Rank)");
                    snprintf(client->tooltip_desc, sizeof(client->tooltip_desc), "Adds +50 Chips when scored - has no rank or suit (Slot %d)", i + 1);
                    snprintf(client->tooltip_stats, sizeof(client->tooltip_stats), "Edition: %s | Seal: %s",
                             get_edition_name(state->hand[i].edition), get_seal_name(state->hand[i].seal));
                } else {
                    snprintf(client->tooltip_title, sizeof(client->tooltip_title), "%s of %s", get_rank_str(state->hand[i].rank), get_suit_name(state->hand[i].suit));
                    snprintf(client->tooltip_type, sizeof(client->tooltip_type), "%s | %s", get_enhancement_name(state->hand[i].enhancement), get_edition_name(state->hand[i].edition));
                    snprintf(client->tooltip_desc, sizeof(client->tooltip_desc), "Playing card in active hand (Slot %d)", i + 1);
                    snprintf(client->tooltip_stats, sizeof(client->tooltip_stats), "Seal: %s", get_seal_name(state->hand[i].seal));
                }
                client->tooltip_pos = (Vector2){mouse.x + 16, mouse.y + 16};

                if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                    if (is_selected) {
                        if (!(state->hand[i].flags & CARD_FORCED)) {
                            /* Deselect (the Cerulean Bell forced card is locked) */
                            for (int s = 0; s < *selected_count; ++s) {
                                if (selected_indices[s] == i) {
                                    for (int k = s; k < *selected_count - 1; ++k) {
                                        selected_indices[k] = selected_indices[k + 1];
                                    }
                                    (*selected_count)--;
                                    break;
                                }
                            }
                        }
                    } else if (*selected_count < MAX_SELECTION) {
                        /* Select */
                        selected_indices[(*selected_count)++] = (uint8_t)i;
                    }
                }
                if (i > 0 && IsKeyPressed(KEY_LEFT)) {
                    out_action->type = ACTION_SWAP_HAND_LEFT;
                    out_action->primary = (uint8_t)i;
                }
                if (i + 1 < state->hand_count && IsKeyPressed(KEY_RIGHT)) {
                    out_action->type = ACTION_SWAP_HAND_RIGHT;
                    out_action->primary = (uint8_t)i;
                }
            }
        }
    }

    /* Action Buttons Bar at bottom */
    float btn_y = rect.y + rect.height - 56.0f;
    float btn_x = rect.x + (rect.width - 522.0f) * 0.5f;

    /* PLAY HAND Button */
    Rectangle play_btn = {btn_x, btn_y, 160, 42};
    bool can_play = (*selected_count >= 1 && *selected_count <= 5 && state->hands_left > 0);
    bool play_hov = can_play && CheckCollisionPointRec(mouse, play_btn);
    DrawRectangleRounded((Rectangle){btn_x, btn_y + 3, 160, 42}, 0.25f, 2, Fade(BLACK, 0.4f));
    DrawRectangleRounded(play_btn, 0.25f, 2, can_play ? (play_hov ? (Color){40, 210, 110, 255} : (Color){25, 170, 85, 255}) : (Color){50, 60, 55, 255});
    DrawText("PLAY HAND", (int)(play_btn.x + (play_btn.width - MeasureText("PLAY HAND", 16)) * 0.5f),
             (int)(play_btn.y + (play_btn.height - 16) * 0.5f), 16,
             can_play ? BALATRO_WHITE : (Color){120, 130, 125, 255});

    if (play_hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        out_action->type = ACTION_PLAY_HAND;
        out_action->selection_count = (uint8_t)*selected_count;
        for (int i = 0; i < *selected_count; ++i) out_action->selection[i] = selected_indices[i];
        *selected_count = 0;
    }

    /* DISCARD Button */
    btn_x += 174.0f;
    Rectangle disc_btn = {btn_x, btn_y, 140, 42};
    bool can_disc = (*selected_count >= 1 && *selected_count <= 5 && state->discards_left > 0);
    bool disc_hov = can_disc && CheckCollisionPointRec(mouse, disc_btn);
    DrawRectangleRounded((Rectangle){btn_x, btn_y + 3, 140, 42}, 0.25f, 2, Fade(BLACK, 0.4f));
    DrawRectangleRounded(disc_btn, 0.25f, 2, can_disc ? (disc_hov ? (Color){220, 60, 60, 255} : (Color){180, 45, 45, 255}) : (Color){60, 50, 50, 255});
    DrawText("DISCARD", (int)(disc_btn.x + (disc_btn.width - MeasureText("DISCARD", 16)) * 0.5f),
             (int)(disc_btn.y + (disc_btn.height - 16) * 0.5f), 16,
             can_disc ? BALATRO_WHITE : (Color){130, 120, 120, 255});

    if (disc_hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        out_action->type = ACTION_DISCARD;
        out_action->selection_count = (uint8_t)*selected_count;
        for (int i = 0; i < *selected_count; ++i) out_action->selection[i] = selected_indices[i];
        *selected_count = 0;
    }

    /* SORT RANK Button */
    btn_x += 154.0f;
    Rectangle sort_r_btn = {btn_x, btn_y + 5, 90, 32};
    bool sort_r_hov = CheckCollisionPointRec(mouse, sort_r_btn);
    DrawRectangleRounded((Rectangle){sort_r_btn.x, sort_r_btn.y + 3, sort_r_btn.width, sort_r_btn.height}, 0.25f, 8, Fade(BLACK, 0.4f));
    DrawRectangleRounded(sort_r_btn, 0.25f, 8, sort_r_hov ? BALATRO_HEADER_BG : BALATRO_PANEL_BG);
    DrawRectangleRoundedLinesEx(sort_r_btn, 0.25f, 8, 1.2f, BALATRO_PANEL_BORDER);
    DrawText("SORT RANK", (int)(sort_r_btn.x + (sort_r_btn.width - MeasureText("SORT RANK", 11)) * 0.5f),
             (int)(sort_r_btn.y + (sort_r_btn.height - 11) * 0.5f), 11, BALATRO_WHITE);
    if (sort_r_hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        out_action->type = ACTION_SORT_HAND_RANK;
    }

    /* SORT SUIT Button */
    btn_x += 104.0f;
    Rectangle sort_s_btn = {btn_x, btn_y + 5, 90, 32};
    bool sort_s_hov = CheckCollisionPointRec(mouse, sort_s_btn);
    DrawRectangleRounded((Rectangle){sort_s_btn.x, sort_s_btn.y + 3, sort_s_btn.width, sort_s_btn.height}, 0.25f, 8, Fade(BLACK, 0.4f));
    DrawRectangleRounded(sort_s_btn, 0.25f, 8, sort_s_hov ? BALATRO_HEADER_BG : BALATRO_PANEL_BG);
    DrawRectangleRoundedLinesEx(sort_s_btn, 0.25f, 8, 1.2f, BALATRO_PANEL_BORDER);
    DrawText("SORT SUIT", (int)(sort_s_btn.x + (sort_s_btn.width - MeasureText("SORT SUIT", 11)) * 0.5f),
             (int)(sort_s_btn.y + (sort_s_btn.height - 11) * 0.5f), 11, BALATRO_WHITE);
    if (sort_s_hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        out_action->type = ACTION_SORT_HAND_SUIT;
    }
}

/* ========================================================================= */
/* Scoring Phase Animation                                                   */
/* ========================================================================= */

static void draw_playing_transform(const Card *card, float x, float y, float width, float height,
                                   float scale_x, float scale_y, float rotation, Texture2D puffer) {
    rlPushMatrix();
    rlTranslatef(x, y, 0.0f);
    rlRotatef(rotation, 0.0f, 0.0f, 1.0f);
    rlScalef(scale_x, scale_y, 1.0f);
    draw_playing_card(card, (Rectangle){-width * 0.5f, -height * 0.5f, width, height},
                      false, false, false, 0, puffer);
    rlPopMatrix();
}

static void draw_joker_transform(const Card *card, float x, float y, float width, float height,
                                 float scale_x, float scale_y, float rotation, Texture2D puffer) {
    rlPushMatrix();
    rlTranslatef(x, y, 0.0f);
    rlRotatef(rotation, 0.0f, 0.0f, 1.0f);
    rlScalef(scale_x, scale_y, 1.0f);
    draw_joker_card(card, (Rectangle){-width * 0.5f, -height * 0.5f, width, height},
                    false, false, 0, puffer);
    rlPopMatrix();
}

static float ease_out_cubic(float t) {
    float u = 1.0f - t;
    return 1.0f - u * u * u;
}

static float ease_out_back(float t) {
    const float c1 = 1.70158f, c3 = c1 + 1.0f;
    float u = t - 1.0f;
    return 1.0f + c3 * u * u * u + c1 * u * u;
}

static int find_card(const Card *cards, int count, uint16_t sort_id) {
    assert(cards);
    assert(count >= 0);
    for (int i = 0; i < count; ++i)
        if (cards[i].sort_id == sort_id) return i;
    return -1;
}

static int same_card(const Card *a, const Card *b) {
    assert(a);
    assert(b);
    return memcmp(a, b, sizeof(*a)) == 0;
}

static void capture_consumable(Client *client, const State *state, const Action *action) {
    assert(client);
    assert(state);
    assert(action);
    Card used = {0};
    if (action->type == ACTION_USE_CONSUMABLE) {
        assert(action->primary < state->consumable_count);
        used = state->consumables[action->primary];
    } else if (action->type == ACTION_BUY_AND_USE) {
        assert(action->primary < state->shop_main_count);
        used = state->shop_main[action->primary];
    } else {
        assert(action->type == ACTION_PICK_PACK_CARD);
        assert(action->primary < state->pack_count);
        used = state->pack_cards[action->primary];
    }
    uint8_t set = centers[used.center_id].set;
    if (set != SET_TAROT && set != SET_PLANET && set != SET_SPECTRAL) return;

    ConsumableAnim *anim = &client->consumable_anim;
    memset(anim, 0, sizeof(*anim));
    anim->pending = true;
    anim->used = used;
    anim->before_hand_count = state->hand_count;
    anim->before_joker_count = state->joker_count;
    anim->before_deck_count = state->deck_count;
    anim->pack_pick = action->type == ACTION_PICK_PACK_CARD;
    memcpy(anim->before_hand, state->hand,
           (size_t)anim->before_hand_count * sizeof(*anim->before_hand));
    memcpy(anim->before_jokers, state->jokers,
           (size_t)anim->before_joker_count * sizeof(*anim->before_jokers));
    for (uint16_t i = 0; i < state->deck_count; ++i)
        anim->before_deck_ids[i] = state->deck[i].sort_id;
    memcpy(anim->before_levels, state->hand_levels, sizeof(anim->before_levels));
    anim->before_dollars = state->dollars;
    anim->selected_count = action->selection_count;
    for (uint8_t i = 0; i < action->selection_count; ++i) {
        assert(action->selection[i] < state->hand_count);
        anim->selected_ids[i] = state->hand[action->selection[i]].sort_id;
    }
}

static void start_consumable(Client *client, const State *state) {
    assert(client);
    assert(state);
    ConsumableAnim *anim = &client->consumable_anim;
    assert(anim->pending);
    anim->pending = false;
    anim->after_joker_count = state->joker_count;
    if (anim->pack_pick && anim->before_hand_count && state->hand_count == 0) {
        for (uint16_t i = 0; i < state->deck_count; ++i) {
            int existed = 0;
            for (uint16_t j = 0; j < anim->before_deck_count; ++j)
                existed |= state->deck[i].sort_id == anim->before_deck_ids[j];
            if (!existed) {
                assert(anim->after_hand_count < MAX_HAND);
                anim->after_hand[anim->after_hand_count++] = state->deck[i];
            }
        }
        for (uint8_t i = 0; i < anim->after_hand_count / 2; ++i) {
            Card card = anim->after_hand[i];
            anim->after_hand[i] = anim->after_hand[anim->after_hand_count - 1 - i];
            anim->after_hand[anim->after_hand_count - 1 - i] = card;
        }
    } else {
        anim->after_hand_count = state->hand_count;
        memcpy(anim->after_hand, state->hand,
               (size_t)anim->after_hand_count * sizeof(*anim->after_hand));
    }
    memcpy(anim->after_jokers, state->jokers,
           (size_t)anim->after_joker_count * sizeof(*anim->after_jokers));
    memcpy(anim->after_levels, state->hand_levels, sizeof(anim->after_levels));
    anim->after_dollars = state->dollars;
    anim->hand_changed = anim->before_hand_count != anim->after_hand_count ||
        memcmp(anim->before_hand, anim->after_hand,
               (size_t)anim->before_hand_count * sizeof(*anim->before_hand)) != 0;
    anim->jokers_changed = anim->before_joker_count != anim->after_joker_count ||
        memcmp(anim->before_jokers, anim->after_jokers,
               (size_t)anim->before_joker_count * sizeof(*anim->before_jokers)) != 0;
    anim->duration = 1.35f;
    anim->t = 0.0f;
    anim->active = true;
}

/* Snapshot the played cards + evaluated hand before the step mutates state. */
static void score_anim_capture(Client *client, const State *state, const Action *action) {
    ScoreAnim *a = &client->score_anim;
    memset(a, 0, sizeof(*a));
    a->pending = true;
    a->played_count = action->selection_count;
    /* Normalize to hand order (left to right) regardless of click order,
       so the scored display always reads left to right. */
    uint8_t order[MAX_SELECTION];
    for (int i = 0; i < a->played_count; ++i) order[i] = action->selection[i];
    for (int i = 0; i < a->played_count; ++i)
        for (int j = i + 1; j < a->played_count; ++j)
            if (order[j] < order[i]) {
                uint8_t t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
    for (int i = 0; i < a->played_count; ++i) {
        a->played[i] = state->hand[order[i]];
        a->played[i].flags &= (uint8_t)~CARD_FACEDOWN; /* revealed when played */
    }
    /* Mirror score_hand's mask: joker-aware classification, Splash, stones */
    int four_fingers = joker_active(state, CENTER_J_FOUR_FINGERS);
    int shortcut = joker_active(state, CENTER_J_SHORTCUT);
    int smeared = joker_active(state, CENTER_J_SMEARED);
    uint8_t mask = 0;
    a->hand_type = classify_hand(a->played, (size_t)a->played_count, &mask, four_fingers, shortcut, smeared);
    a->hand_not_allowed = !state->blind_disabled && state->blind_id == BLIND_BL_MOUTH &&
        state->blind_only_hand != UINT8_MAX && state->blind_only_hand != (uint8_t)a->hand_type;
    if (a->hand_not_allowed) {
        a->allowed_hand_type = (HandType)state->blind_only_hand;
        mask = 0;
    }
    if (!a->hand_not_allowed && joker_active(state, CENTER_J_SPLASH))
        mask = (uint8_t)((1u << a->played_count) - 1u);
    if (!a->hand_not_allowed)
        for (int i = 0; i < a->played_count; ++i)
            if (a->played[i].enhancement == ENHANCEMENT_STONE) mask |= (uint8_t)(1u << i);
    a->scoring_mask = mask;
    a->level = state->hand_levels[a->hand_type] ? state->hand_levels[a->hand_type] : 1;
    a->chips_before = state->chips;
    a->blind_target = state->blind_chips;
    a->duration = 1.65f;
}

/* Kick off the animation once the step result is known. */
static void score_anim_start(Client *client, const State *state) {
    ScoreAnim *a = &client->score_anim;
    if (!a->pending) return;
    a->pending = false;
    /* Aggregated engine values: hand base + jokers + enhancements + debuffs */
    a->chips = (int)state->last_hand_chips;
    a->mult = (int)state->last_hand_mult;
    a->total = state->last_hand_score;
    a->blind_beaten = state->phase == PHASE_ROUND_EVAL;
    a->bust = state->phase == PHASE_GAME_OVER;
    a->active = true;
    a->t = 0.0f;
}

/* Full-screen scoring sequence: cards fly up, hand-type banner, chips x mult
   tally, then the running total against the blind target. */
static void draw_phase_scoring(const Client *client, Rectangle rect) {
    const ScoreAnim *a = &client->score_anim;

    DrawRectangleRounded(rect, 0.025f, 4, (Color){6, 12, 18, 248});
    for (int ring = 0; ring < 5; ++ring) {
        float radius = 110.0f + ring * 74.0f + sinf(a->t * 2.0f + ring) * 8.0f;
        DrawCircleLines((int)(rect.x + rect.width * 0.5f), (int)(rect.y + rect.height * 0.48f), radius,
                        Fade(ring & 1 ? BALATRO_RED_MULT : BALATRO_BLUE_CHIPS, 0.075f));
    }

    float cx = rect.x + rect.width * 0.5f;
    float cy = rect.y + rect.height * 0.52f;

    const char *phase_title = "HAND SCORING";
    DrawText(phase_title, (int)(cx - MeasureText(phase_title, 12) * 0.5f), (int)(rect.y + 18), 12, (Color){115, 145, 165, 255});
    DrawRectangle((int)(cx - 42), (int)(rect.y + 37), 84, 2, Fade(BALATRO_GOLD, 0.55f));

    /* Played cards: rise + settle into a fan. Cards use the full hand size
       and a uniform rlgl scale so every element (corner pips, face emblem,
       enhancement badge) keeps hand-view spacing instead of colliding. */
    float card_w = 74.0f, card_h = 106.0f;
    float pop = ease_out_back((a->t < 0.24f ? a->t : 0.24f) / 0.24f);
    float card_y = cy - card_h * 0.5f - 26.0f + (1.0f - pop) * 100.0f;
    float pitch = (a->played_count > 1) ? card_w + 8.0f : 0.0f;
    float total_w = (a->played_count - 1) * pitch;
    for (int i = 0; i < a->played_count; ++i) {
        float x = cx - total_w * 0.5f + i * pitch;
        /* Every card in the classified scoring hand rises. Debuffed cards
           keep their overlay but do not receive the contributor ring. */
        bool in_scoring_hand = (a->scoring_mask & (1u << i)) != 0;
        bool scores = in_scoring_hand && !(a->played[i].flags & CARD_DEBUFFED);
        float lift = in_scoring_hand ? 0.0f : 30.0f;
        rlPushMatrix();
        rlTranslatef(x, card_y + card_h * 0.5f + lift, 0);
        rlScalef(pop, pop, 1.0f);
        draw_playing_card(&a->played[i], (Rectangle){-card_w * 0.5f, -card_h * 0.5f, card_w, card_h}, false, false, false, 0, client->puffer);
        if (scores) {
            DrawRectangleRoundedLinesEx((Rectangle){-card_w * 0.5f, -card_h * 0.5f, card_w, card_h},
                                        0.08f, 4, 3.0f, Fade(BALATRO_GOLD, 0.85f));
        }
        rlPopMatrix();
    }

    /* Hand type banner with pop-in */
    if (a->t > 0.12f) {
        float banner_progress = (a->t - 0.12f) / 0.22f;
        if (banner_progress > 1.0f) banner_progress = 1.0f;
        float bt = ease_out_back(banner_progress);
        const char *name = get_hand_type_name(a->hand_type);
        int fs = 44;
        int alpha = (int)(255.0f * (bt > 1.0f ? 1.0f : bt));
        if (alpha > 255) alpha = 255;
        Color banner_col = a->hand_not_allowed ? BALATRO_RED_MULT : BALATRO_GOLD;
        banner_col.a = (unsigned char)alpha;
        DrawText(name, (int)(cx - MeasureText(name, fs) * 0.5f), (int)(cy - 168.0f - (1.0f - bt) * 24.0f), fs, banner_col);
        char lvl_buf[64];
        if (a->hand_not_allowed)
            snprintf(lvl_buf, sizeof(lvl_buf), "MOUTH LOCKED TO %s",
                     get_hand_type_name(a->allowed_hand_type));
        else
            snprintf(lvl_buf, sizeof(lvl_buf), "LEVEL %d", a->level);
        DrawText(lvl_buf, (int)(cx - MeasureText(lvl_buf, 14) * 0.5f), (int)(cy - 118.0f), 14, Fade(BALATRO_WHITE, bt > 1.0f ? 0.8f : 0.8f * bt));
    }

    /* Chips x Mult pills with counting chips */
    if (a->t > 0.38f) {
        float count_progress = (a->t - 0.38f) / 0.30f;
        if (count_progress > 1.0f) count_progress = 1.0f;
        float ct = ease_out_cubic(count_progress);
        if (a->hand_not_allowed) {
            Rectangle warning = {cx - 125.0f, cy + 46.0f, 250.0f, 42.0f};
            DrawRectangleRounded((Rectangle){warning.x + 3, warning.y + 4, warning.width, warning.height},
                                 0.2f, 3, Fade(BLACK, 0.5f));
            DrawRectangleRounded(warning, 0.2f, 3, Fade(BALATRO_RED_MULT, ct));
            const char *message = "NOT ALLOWED - 0 CHIPS";
            DrawText(message, (int)(cx - MeasureText(message, 18) * 0.5f),
                     (int)(warning.y + 12), 18, Fade(BALATRO_WHITE, ct));
        } else {
        float pw = 110.0f, ph = 34.0f;
        float gap = 26.0f;
        float x0 = cx - pw - gap * 0.5f, x1 = cx + gap * 0.5f;
        float y0 = cy + 46.0f;

        DrawRectangleRounded((Rectangle){x0 + 3, y0 + 4, pw, ph}, 0.25f, 3, Fade(BLACK, 0.5f));
        DrawRectangleRounded((Rectangle){x0, y0, pw, ph}, 0.25f, 3, BALATRO_BLUE_CHIPS);
        DrawText(TextFormat("%d", (int)(a->chips * ct)), (int)(x0 + 12), (int)(y0 + 8), 18, BALATRO_WHITE);
        DrawText("Chips", (int)(x0 + pw - 52), (int)(y0 + 12), 12, Fade(BALATRO_WHITE, 0.85f));

        DrawText("x", (int)(cx - 5), (int)(y0 + 9), 20, BALATRO_GOLD);

        DrawRectangleRounded((Rectangle){x1 + 3, y0 + 4, pw, ph}, 0.25f, 3, Fade(BLACK, 0.5f));
        DrawRectangleRounded((Rectangle){x1, y0, pw, ph}, 0.25f, 3, BALATRO_RED_MULT);
        DrawText(TextFormat("%d", (int)(a->mult * ct)), (int)(x1 + 12), (int)(y0 + 8), 18, BALATRO_WHITE);
        DrawText("Mult", (int)(x1 + pw - 44), (int)(y0 + 12), 12, Fade(BALATRO_WHITE, 0.85f));
        }
    }

    /* Running total vs blind target */
    if (a->t > 0.70f) {
        float total_progress = (a->t - 0.70f) / 0.55f;
        if (total_progress > 1.0f) total_progress = 1.0f;
        float tt = ease_out_cubic(total_progress);
        double shown = a->total * tt;
        char total_buf[48];
        snprintf(total_buf, sizeof(total_buf), "+ %.0f CHIPS", shown);
        DrawText(total_buf, (int)(cx - MeasureText(total_buf, 34) * 0.5f), (int)(cy + 100.0f), 34, BALATRO_WHITE);

        /* Blind progress */
        if (a->blind_target > 0.0) {
            float bw = rect.width * 0.55f;
            float bx = cx - bw * 0.5f, by = cy + 160.0f;
            char goal_buf[48];
            snprintf(goal_buf, sizeof(goal_buf), "BLIND PROGRESS   %.0f / %.0f", a->chips_before + shown, a->blind_target);
            DrawText(goal_buf, (int)(bx + 4), (int)(by - 20.0f), 13, (Color){170, 195, 220, 255});

            DrawRectangle((int)bx, (int)by, (int)bw, 16, (Color){10, 20, 18, 255});
            double before = a->chips_before, after = a->chips_before + shown;
            double f0 = before / a->blind_target, f1 = after / a->blind_target;
            if (f0 > 1.0) f0 = 1.0;
            if (f1 > 1.0) f1 = 1.0;
            DrawRectangle((int)bx, (int)by, (int)(bw * f0), 16, (Color){30, 90, 70, 255});
            DrawRectangle((int)(bx + bw * f0), (int)by, (int)(bw * (f1 - f0)), 16, BALATRO_BLUE_CHIPS);
            DrawRectangleLines((int)bx, (int)by, (int)bw, 16, BALATRO_PANEL_BORDER);
        }
    }

    /* Round outcome */
    if (a->t > 1.25f) {
        float pulse = 0.75f + 0.25f * sinf(a->t * 10.0f);
        if (a->blind_beaten) {
            const char *msg = "BLIND DEFEATED!";
            DrawText(msg, (int)(cx - MeasureText(msg, 26) * 0.5f), (int)(cy - 205.0f), 26,
                     (Color){(unsigned char)(255 * pulse), 210, 60, 255});
        } else if (a->bust) {
            const char *msg = "OUT OF HANDS";
            DrawText(msg, (int)(cx - MeasureText(msg, 26) * 0.5f), (int)(cy - 205.0f), 26, BALATRO_RED_MULT);
        }
    }
}

/* Phase 2b: Blind defeated — collect earnings and move to the shop */
static void draw_phase_round_eval(const State *state, Rectangle rect, Action *out_action, Vector2 mouse, Client *client) {
    float cx = rect.x + rect.width * 0.5f;
    float cy = rect.y + rect.height * 0.45f;

    Rectangle panel = {cx - 220.0f, cy - 110.0f, 440.0f, 220.0f};
    DrawRectangleRounded(panel, 0.1f, 4, BALATRO_HEADER_BG);
    DrawRectangleRoundedLinesEx(panel, 0.1f, 4, 2.5f, BALATRO_GOLD);

    const char *title = "BLIND DEFEATED!";
    DrawText(title, (int)(cx - MeasureText(title, 30) * 0.5f), (int)(cy - 80.0f), 30, BALATRO_GOLD);

    const char *sub = get_blind_name(state->blind_id);
    DrawText(sub, (int)(cx - MeasureText(sub, 15) * 0.5f), (int)(cy - 42.0f), 15, (Color){180, 205, 230, 255});

    char reward_buf[64];
    snprintf(reward_buf, sizeof(reward_buf), "Round Earnings: +$%d", state->round_earnings);
    DrawText(reward_buf, (int)(cx - MeasureText(reward_buf, 16) * 0.5f), (int)(cy - 12.0f), 16, BALATRO_GOLD);

    char score_buf[64];
    snprintf(score_buf, sizeof(score_buf), "Final Score: %.0f / %.0f", state->chips, state->blind_chips);
    DrawText(score_buf, (int)(cx - MeasureText(score_buf, 13) * 0.5f), (int)(cy + 16.0f), 13, BALATRO_BLUE_CHIPS);

    /* CASH OUT button */
    Rectangle btn = {cx - 100.0f, cy + 48.0f, 200.0f, 42.0f};
    bool hov = CheckCollisionPointRec(mouse, btn);
    DrawRectangleRounded((Rectangle){btn.x, btn.y + 3, btn.width, btn.height}, 0.25f, 3, Fade(BLACK, 0.4f));
    DrawRectangleRounded(btn, 0.25f, 3, hov ? (Color){40, 200, 100, 255} : (Color){25, 160, 80, 255});
    DrawText("CASH OUT  (ENTER)", (int)(cx - MeasureText("CASH OUT  (ENTER)", 15) * 0.5f), (int)(btn.y + 12), 15, BALATRO_WHITE);
    if (hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        out_action->type = ACTION_CASH_OUT;
    }
}

/* Phase 3: Shop View */
static void draw_phase_shop(const State *state, const LegalMasks *legal, Rectangle rect, Action *out_action, Vector2 mouse, Client *client) {
    float margin = 20.0f;
    Rectangle marquee = {rect.x + margin, rect.y + 14, rect.width - margin * 2.0f, 54};
    DrawRectangleRounded((Rectangle){marquee.x + 3, marquee.y + 4, marquee.width, marquee.height}, 0.18f, 3, Fade(BLACK, 0.42f));
    DrawRectangleRounded(marquee, 0.18f, 3, (Color){25, 45, 43, 255});
    DrawRectangleRoundedLinesEx(marquee, 0.18f, 3, 1.5f, Fade(BALATRO_GOLD, 0.45f));
    DrawText("THE SHOP", (int)(marquee.x + 18),
             (int)(marquee.y + (marquee.height - 23) * 0.5f), 23, BALATRO_GOLD);

    Rectangle next_btn = {marquee.x + marquee.width - 150, marquee.y + 9, 132, 36};
    bool next_hov = CheckCollisionPointRec(mouse, next_btn);
    DrawRectangleRounded((Rectangle){next_btn.x + 2, next_btn.y + 3, next_btn.width, next_btn.height}, 0.25f, 2, Fade(BLACK, 0.4f));
    DrawRectangleRounded(next_btn, 0.25f, 2, next_hov ? (Color){40, 200, 100, 255} : (Color){25, 160, 80, 255});
    DrawText("LEAVE SHOP", (int)(next_btn.x + 19), (int)(next_btn.y + 10), 14, BALATRO_WHITE);
    if (next_hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        out_action->type = ACTION_NEXT_ROUND;
    }

    Rectangle reroll_btn = {next_btn.x - 142, marquee.y + 9, 128, 36};
    assert(!state->free_rerolls || state->reroll_cost == 0);
    bool can_reroll = state->free_rerolls || can_afford(state, state->reroll_cost);
    bool reroll_hov = can_reroll && CheckCollisionPointRec(mouse, reroll_btn);
    DrawRectangleRounded((Rectangle){reroll_btn.x + 2, reroll_btn.y + 3, reroll_btn.width, reroll_btn.height}, 0.25f, 2, Fade(BLACK, 0.4f));
    DrawRectangleRounded(reroll_btn, 0.25f, 2, can_reroll ? (reroll_hov ? (Color){220, 160, 40, 255} : (Color){180, 125, 30, 255}) : (Color){60, 55, 50, 255});
    const char *reroll_text = state->free_rerolls
        ? TextFormat("REROLL  FREE (%u)", state->free_rerolls)
        : TextFormat("REROLL  $%d", state->reroll_cost);
    DrawText(reroll_text, (int)(reroll_btn.x + (reroll_btn.width - MeasureText(reroll_text, 13)) * 0.5f),
             (int)(reroll_btn.y + (reroll_btn.height - 13) * 0.5f), 13,
             can_reroll ? BALATRO_WHITE : (Color){130, 125, 120, 255});
    if (reroll_hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        out_action->type = ACTION_REROLL;
    }

    float upper_y = rect.y + 80.0f;
    float voucher_w = 170.0f;
    Rectangle offers = {rect.x + margin, upper_y, rect.width - margin * 2.0f - voucher_w - 10.0f, 224};
    Rectangle vouchers = {offers.x + offers.width + 10, upper_y, voucher_w, 224};
    DrawRectangleRounded(offers, 0.05f, 3, (Color){14, 34, 34, 235});
    DrawRectangleRoundedLinesEx(offers, 0.05f, 3, 1.2f, (Color){43, 88, 78, 255});
    DrawRectangleRounded(vouchers, 0.05f, 3, (Color){30, 27, 23, 235});
    DrawRectangleRoundedLinesEx(vouchers, 0.05f, 3, 1.2f, Fade(BALATRO_COLOR_VOUCHER, 0.65f));
    DrawText("FEATURED CARDS", (int)(offers.x + 14), (int)(offers.y + 11), 13, (Color){178, 215, 205, 255});
    DrawText(TextFormat("%d AVAILABLE", state->shop_main_count), (int)(offers.x + offers.width - 88), (int)(offers.y + 13), 9, (Color){90, 125, 120, 255});
    DrawText("VOUCHERS", (int)(vouchers.x + 14), (int)(vouchers.y + 11), 13, (Color){225, 190, 125, 255});
    DrawLine((int)(offers.x + 14), (int)(offers.y + 34), (int)(offers.x + offers.width - 14), (int)(offers.y + 34), Fade(BALATRO_WHITE, 0.08f));
    DrawLine((int)(vouchers.x + 14), (int)(vouchers.y + 34), (int)(vouchers.x + vouchers.width - 14), (int)(vouchers.y + 34), Fade(BALATRO_WHITE, 0.08f));

    for (int i = 0; i < state->shop_main_count; ++i) {
        float card_w = 86.0f;
        float pitch = state->shop_main_count > 1
            ? (offers.width - 48.0f - card_w) / (state->shop_main_count - 1) : 0.0f;
        if (pitch > 140.0f) pitch = 140.0f;
        float cards_w = card_w + (state->shop_main_count - 1) * pitch;
        Rectangle card_r = {offers.x + (offers.width - cards_w) * 0.5f + i * pitch, offers.y + 50.0f, card_w, 126};
        bool hovered = CheckCollisionPointRec(mouse, card_r);

        const Card *sc = &state->shop_main[i];
        assert(sc->center_id < CENTER_COUNT);
        uint8_t set = centers[sc->center_id].set;
        if (set == SET_JOKER) {
            draw_joker_card(sc, card_r, hovered, true, sc->cost, client->puffer);
        } else if (set == SET_TAROT || set == SET_PLANET || set == SET_SPECTRAL) {
            draw_consumable_card(sc, card_r, hovered, i, true, sc->cost);
        } else {
            draw_playing_card(sc, card_r, false, hovered, true, sc->cost, client->puffer);
        }

        bool affordable = can_afford(state, sc->cost);
        if (!affordable) {
            DrawRectangleRounded(card_r, 0.08f, 4, Fade((Color){15, 10, 12, 255}, 0.50f));
            DrawText("TOO EXPENSIVE", (int)(card_r.x + 5), (int)(card_r.y + card_r.height + 7), 9, (Color){205, 95, 95, 255});
        } else {
            DrawText("CLICK TO BUY", (int)(card_r.x + 10), (int)(card_r.y + card_r.height + 7), 9, (Color){90, 180, 130, 255});
        }

        if (hovered) {
            client->show_tooltip = true;
            if (set == SET_DEFAULT || set == SET_ENHANCED) {
                if (sc->enhancement == ENHANCEMENT_STONE) {
                    snprintf(client->tooltip_title, sizeof(client->tooltip_title), "Stone Card");
                    snprintf(client->tooltip_type, sizeof(client->tooltip_type), "Stone (+50 Chips, No Rank) | $%d", sc->cost);
                } else {
                    snprintf(client->tooltip_title, sizeof(client->tooltip_title), "%s of %s", get_rank_str(sc->rank), get_suit_name(sc->suit));
                    snprintf(client->tooltip_type, sizeof(client->tooltip_type), "Playing Card | $%d", sc->cost);
                }
                snprintf(client->tooltip_desc, sizeof(client->tooltip_desc), "%s | %s | Seal: %s",
                         get_enhancement_name(sc->enhancement), get_edition_name(sc->edition), get_seal_name(sc->seal));
            } else {
                snprintf(client->tooltip_title, sizeof(client->tooltip_title), "%s", get_center_name(sc->center_id));
                snprintf(client->tooltip_type, sizeof(client->tooltip_type), "%s | $%d",
                         set == SET_JOKER ? get_edition_name(sc->edition) : "Shop Card", sc->cost);
                get_center_description(sc, state, client->tooltip_desc, sizeof(client->tooltip_desc));
            }
            snprintf(client->tooltip_stats, sizeof(client->tooltip_stats), affordable ? "Affordable" : "Need $%d more", sc->cost - state->dollars);
            if (set == SET_JOKER)
                client->tooltip_pos = (Vector2){card_r.x + card_r.width * 0.5f - 150.0f,
                                                card_r.y + card_r.height + 10.0f};
            else
                client->tooltip_pos = (Vector2){mouse.x + 16, mouse.y + 16};

            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && affordable) {
                out_action->type = ACTION_BUY_CARD;
                out_action->primary = (uint8_t)i;
            }
        }
    }

    assert(state->shop_voucher_count <= 1);
    for (int i = 0; i < state->shop_voucher_count; ++i) {
        float ticket_w = 112.0f;
        Rectangle v_rect = {vouchers.x + (vouchers.width - ticket_w) * 0.5f, vouchers.y + 52.0f, ticket_w, 120};
        bool hovered = CheckCollisionPointRec(mouse, v_rect);

        DrawRectangleRounded((Rectangle){v_rect.x + 3, v_rect.y + 4, v_rect.width, v_rect.height}, 0.1f, 3, Fade(BLACK, 0.45f));
        DrawRectangleRounded(v_rect, 0.1f, 3, (Color){102, 74, 34, 255});
        DrawRectangleRoundedLinesEx(v_rect, 0.1f, 3, hovered ? 2.5f : 1.2f, hovered ? BALATRO_GOLD : (Color){218, 176, 98, 255});
        DrawCircle((int)v_rect.x, (int)(v_rect.y + v_rect.height * 0.5f), 7, (Color){30, 27, 23, 255});
        DrawCircle((int)(v_rect.x + v_rect.width), (int)(v_rect.y + v_rect.height * 0.5f), 7, (Color){30, 27, 23, 255});
        DrawLine((int)(v_rect.x + 9), (int)(v_rect.y + 78), (int)(v_rect.x + v_rect.width - 9), (int)(v_rect.y + 78), Fade(BALATRO_WHITE, 0.28f));

        const Card *voucher_card = &state->shop_vouchers[i];
        const char *vname = get_center_name(voucher_card->center_id);
        DrawText("VOUCHER", (int)(v_rect.x + 31), (int)(v_rect.y + 10), 9, (Color){236, 203, 137, 255});
        draw_centered_multiline_text(vname, (int)(v_rect.x + v_rect.width * 0.5f), (int)(v_rect.y + 35), (int)(v_rect.width - 14), 2, 11, BALATRO_WHITE);
        const char *voucher_price = TextFormat("$%d", voucher_card->cost);
        DrawText(voucher_price, (int)(v_rect.x + (v_rect.width - MeasureText(voucher_price, 16)) * 0.5f),
                 (int)(v_rect.y + 91), 16, BALATRO_GOLD);

        if (hovered) {
            client->show_tooltip = true;
            snprintf(client->tooltip_title, sizeof(client->tooltip_title), "%s", vname);
            snprintf(client->tooltip_type, sizeof(client->tooltip_type), "Voucher ($%d)", voucher_card->cost);
            Card vc = {.center_id = state->shop_vouchers[i].center_id};
            get_center_description(&vc, state, client->tooltip_desc, sizeof(client->tooltip_desc));
            snprintf(client->tooltip_stats, sizeof(client->tooltip_stats), "Cost: $%d", voucher_card->cost);
            client->tooltip_pos = (Vector2){mouse.x + 16, mouse.y + 16};

            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && can_afford(state, voucher_card->cost)) {
                out_action->type = ACTION_REDEEM_VOUCHER;
                out_action->primary = (uint8_t)i;
            }
        }
    }

    Rectangle boosters = {rect.x + margin, upper_y + 234.0f, rect.width - margin * 2.0f, rect.height - (upper_y - rect.y) - 248.0f};
    DrawRectangleRounded(boosters, 0.05f, 3, (Color){25, 20, 38, 235});
    DrawRectangleRoundedLinesEx(boosters, 0.05f, 3, 1.2f, Fade(BALATRO_COLOR_BOOSTER, 0.7f));
    DrawText("BOOSTER PACKS", (int)(boosters.x + 14), (int)(boosters.y + 11), 13, (Color){210, 170, 235, 255});
    DrawText(TextFormat("%d AVAILABLE", state->shop_booster_count), (int)(boosters.x + boosters.width - 88), (int)(boosters.y + 13), 9, (Color){125, 105, 145, 255});

    for (int i = 0; i < state->shop_booster_count; ++i) {
        float pack_w = 280.0f;
        float packs_w = state->shop_booster_count * pack_w + (state->shop_booster_count - 1) * 18.0f;
        Rectangle p_rect = {boosters.x + (boosters.width - packs_w) * 0.5f + i * (pack_w + 18.0f), boosters.y + 40.0f, pack_w, boosters.height - 54.0f};
        bool hovered = CheckCollisionPointRec(mouse, p_rect);

        Color pack_color = i & 1 ? (Color){73, 53, 139, 255} : BALATRO_COLOR_BOOSTER;
        DrawRectangleRounded((Rectangle){p_rect.x + 3, p_rect.y + 4, p_rect.width, p_rect.height}, 0.12f, 3, Fade(BLACK, 0.45f));
        DrawRectangleRounded(p_rect, 0.12f, 3, pack_color);
        DrawRectangleRoundedLinesEx(p_rect, 0.12f, 3, hovered ? 2.5f : 1.5f, hovered ? BALATRO_GOLD : Fade(BALATRO_WHITE, 0.65f));

        const Card *pack = &state->shop_boosters[i];
        assert(pack->center_id < CENTER_COUNT);
        uint8_t kind = centers[pack->center_id].kind;
        Rectangle art = {p_rect.x + 14, p_rect.y + 12, 78, p_rect.height - 24};
        Vector2 motif = {art.x + art.width * 0.5f, art.y + art.height * 0.5f};
        DrawRectangleRounded(art, 0.12f, 3, Fade(BLACK, 0.16f));
        DrawRectangleRoundedLinesEx(art, 0.12f, 3, 1.0f, Fade(BALATRO_WHITE, 0.22f));
        if (kind == 1) {
            Rectangle tarot = {motif.x - 19, motif.y - 29, 38, 58};
            DrawRectangleRounded(tarot, 0.08f, 2, (Color){45, 20, 77, 255});
            DrawRectangleRoundedLinesEx(tarot, 0.08f, 2, 1.8f, BALATRO_GOLD);
            DrawPoly((Vector2){motif.x, motif.y - 14}, 8, 10, 22.5f, Fade(BALATRO_GOLD, 0.55f));
            DrawEllipse((int)motif.x, (int)(motif.y - 14), 10, 5, BALATRO_CARD_WHITE);
            DrawCircleV((Vector2){motif.x, motif.y - 14}, 3, BALATRO_COLOR_TAROT);
            DrawCircleV((Vector2){motif.x, motif.y + 14}, 6, BALATRO_GOLD);
        } else if (kind == 2) {
            DrawCircleV((Vector2){motif.x - 21, motif.y - 22}, 2, BALATRO_WHITE);
            DrawCircleV((Vector2){motif.x + 24, motif.y - 12}, 2, BALATRO_GOLD);
            DrawCircleV((Vector2){motif.x + 16, motif.y + 25}, 1.5f, BALATRO_WHITE);
            DrawCircleGradient((int)motif.x, (int)motif.y, 20, (Color){150, 210, 255, 255}, (Color){40, 80, 180, 255});
            DrawEllipseLines((int)motif.x, (int)motif.y, 31, 10, BALATRO_GOLD);
        } else if (kind == 3) {
            DrawCircleV((Vector2){motif.x - 6, motif.y}, 22, (Color){110, 235, 225, 255});
            DrawCircleV((Vector2){motif.x + 6, motif.y - 8}, 21, pack_color);
            DrawCircleV((Vector2){motif.x + 20, motif.y - 24}, 3, BALATRO_WHITE);
            DrawCircleV((Vector2){motif.x + 27, motif.y + 15}, 2, BALATRO_GOLD);
        } else if (kind == 4) {
            Rectangle left_card = {motif.x - 25, motif.y - 21, 29, 43};
            Rectangle right_card = {motif.x - 2, motif.y - 25, 29, 43};
            DrawRectangleRounded(left_card, 0.10f, 2, BALATRO_CARD_WHITE);
            DrawRectangleRoundedLinesEx(left_card, 0.10f, 2, 1.4f, BALATRO_GOLD);
            draw_suit_icon_rot(HEARTS, left_card.x + left_card.width * 0.5f, left_card.y + left_card.height * 0.5f, 14, 0, BALATRO_HEART_RED);
            DrawRectangleRounded(right_card, 0.10f, 2, BALATRO_CARD_WHITE);
            DrawRectangleRoundedLinesEx(right_card, 0.10f, 2, 1.4f, BALATRO_BLUE_CHIPS);
            draw_suit_icon_rot(SPADES, right_card.x + right_card.width * 0.5f, right_card.y + right_card.height * 0.5f, 14, 0, BALATRO_SPADE_DARK);
        } else {
            /* Buffoon (joker) pack: PufferLib mascot portrait */
            float pr = 16.0f;
            DrawCircle((int)motif.x, (int)motif.y, pr, Fade(BALATRO_COLOR_BOOSTER, 0.35f));
            DrawCircleLines((int)motif.x, (int)motif.y, pr, Fade(BALATRO_WHITE, 0.7f));
            assert(client->puffer.id > 0);
            DrawTexturePro(client->puffer, (Rectangle){0, 0, 128, 128},
                           (Rectangle){motif.x - pr, motif.y - pr, pr * 2.0f, pr * 2.0f},
                           (Vector2){0, 0}, 0, WHITE);
        }

        const char *pname = get_center_name(pack->center_id);
        draw_centered_multiline_text(pname, (int)(p_rect.x + 188), (int)(p_rect.y + 11), 168, 2, 12, BALATRO_WHITE);
        char pack_desc[160];
        get_center_description(pack, state, pack_desc, sizeof(pack_desc));
        draw_centered_multiline_text(pack_desc, (int)(p_rect.x + 188), (int)(p_rect.y + 40), 168, 3, 9, Fade(BALATRO_WHITE, 0.72f));
        DrawText(TextFormat("$%d", pack->cost), (int)(p_rect.x + 180), (int)(p_rect.y + p_rect.height - 27), 17, BALATRO_GOLD);

        if (hovered) {
            client->show_tooltip = true;
            snprintf(client->tooltip_title, sizeof(client->tooltip_title), "%s", pname);
            snprintf(client->tooltip_type, sizeof(client->tooltip_type), "Booster Pack");
            Card pc = {.center_id = state->shop_boosters[i].center_id};
            get_center_description(&pc, state, client->tooltip_desc, sizeof(client->tooltip_desc));
            snprintf(client->tooltip_stats, sizeof(client->tooltip_stats), "Cost: $%d", state->shop_boosters[i].cost);
            client->tooltip_pos = (Vector2){mouse.x + 16, mouse.y + 16};

            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && can_afford(state, state->shop_boosters[i].cost)) {
                out_action->type = ACTION_OPEN_BOOSTER;
                out_action->primary = (uint8_t)i;
            }
        }
    }
}

/* Phase 4: Booster Pack Opening View */
static void draw_phase_pack_opening(const State *state, const LegalMasks *legal, Rectangle rect, int *selected_count, uint8_t *selected_indices, Action *out_action, Vector2 mouse, Client *client) {
    float y = rect.y + 20.0f;
    DrawText(TextFormat("CHOOSE %d OF %d CARDS", state->pack_choices, state->pack_count), (int)(rect.x + rect.width * 0.5f - 120), (int)y, 20, BALATRO_GOLD);

    /* Skip Pack Button */
    Rectangle skip_btn = {rect.x + rect.width - 160, y - 5, 120, 34};
    bool skip_hov = CheckCollisionPointRec(mouse, skip_btn);
    DrawRectangleRounded(skip_btn, 0.25f, 2, skip_hov ? (Color){180, 60, 60, 255} : (Color){140, 45, 45, 255});
    DrawText("SKIP PACK", (int)(skip_btn.x + 22), (int)(skip_btn.y + 10), 13, BALATRO_WHITE);
    if (skip_hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        out_action->type = ACTION_SKIP_PACK;
    }

    y += 60.0f;
    float card_w = 84.0f;
    float card_h = 120.0f;
    float start_x = rect.x + (rect.width - (state->pack_count * (card_w + 16.0f))) * 0.5f;

    for (int i = 0; i < state->pack_count; ++i) {
        float cx = start_x + i * (card_w + 16.0f);
        Rectangle c_rect = {cx, y, card_w, card_h};
        bool hovered = CheckCollisionPointRec(mouse, c_rect);

        const Card *pc = &state->pack_cards[i];
        assert(pc->center_id < CENTER_COUNT);
        uint8_t set = centers[pc->center_id].set;
        if (set == SET_JOKER) {
            draw_joker_card(pc, c_rect, hovered, false, 0, client->puffer);
        } else if (set == SET_TAROT || set == SET_PLANET || set == SET_SPECTRAL) {
            draw_consumable_card(pc, c_rect, hovered, i, false, 0);
        } else {
            draw_playing_card(pc, c_rect, false, hovered, false, 0, client->puffer);
        }

        if (hovered) {
            client->show_tooltip = true;
            if (set == SET_DEFAULT || set == SET_ENHANCED) {
                if (pc->enhancement == ENHANCEMENT_STONE) {
                    snprintf(client->tooltip_title, sizeof(client->tooltip_title), "Stone Card");
                    snprintf(client->tooltip_type, sizeof(client->tooltip_type), "Stone (+50 Chips, No Rank)");
                } else {
                    snprintf(client->tooltip_title, sizeof(client->tooltip_title), "%s of %s", get_rank_str(pc->rank), get_suit_name(pc->suit));
                    snprintf(client->tooltip_type, sizeof(client->tooltip_type), "Playing Card");
                }
                snprintf(client->tooltip_desc, sizeof(client->tooltip_desc), "%s | %s | Seal: %s",
                         get_enhancement_name(pc->enhancement), get_edition_name(pc->edition), get_seal_name(pc->seal));
                snprintf(client->tooltip_stats, sizeof(client->tooltip_stats), "Adds this card to your deck");
            } else {
                snprintf(client->tooltip_title, sizeof(client->tooltip_title), "%s", get_center_name(pc->center_id));
                snprintf(client->tooltip_type, sizeof(client->tooltip_type), "%s",
                         set == SET_JOKER ? get_edition_name(pc->edition) : "Pack Choice");
                get_center_description(pc, state, client->tooltip_desc, sizeof(client->tooltip_desc));
                snprintf(client->tooltip_stats, sizeof(client->tooltip_stats), "Available pack choice");
            }
            if (set == SET_JOKER)
                client->tooltip_pos = (Vector2){c_rect.x + c_rect.width * 0.5f - 150.0f,
                                                c_rect.y + c_rect.height + 10.0f};
            else
                client->tooltip_pos = (Vector2){mouse.x + 16, mouse.y + 16};

            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                out_action->type = ACTION_PICK_PACK_CARD;
                out_action->primary = (uint8_t)i;
                out_action->selection_count = (uint8_t)*selected_count;
                for (int s = 0; s < *selected_count; ++s) out_action->selection[s] = selected_indices[s];
            }
        }
    }

    /* Tarot/Spectral packs are used on the hand: show it selectable */
    if ((state->pack_kind == PACK_TAROT || state->pack_kind == PACK_SPECTRAL) && state->hand_count > 0) {
        float hy = rect.y + rect.height - 150.0f;
        const char *label = "YOUR HAND  -  select cards to target";
        DrawText(label, (int)(rect.x + rect.width * 0.5f - MeasureText(label, 13) * 0.5f), (int)hy, 13,
                 Fade((Color){170, 190, 210, 255}, 0.8f));
        hy += 30.0f;
        float hcard_w = 64.0f, hcard_h = 92.0f;
        float pitch = hcard_w + 8.0f;
        float avail = rect.width - 24.0f;
        if (state->hand_count * pitch > avail && state->hand_count > 1)
            pitch = (avail - hcard_w) / (float)(state->hand_count - 1);
        float total_w = (state->hand_count - 1) * pitch + hcard_w;
        float start_x = rect.x + (rect.width - total_w) * 0.5f;

        for (int i = 0; i < state->hand_count; ++i) {
            float cx = start_x + i * pitch;
            bool is_selected = false;
            for (int s = 0; s < *selected_count; ++s)
                if (selected_indices[s] == i) {
                    is_selected = true;
                    break;
                }
            bool hovered = CheckCollisionPointRec(mouse,
                (Rectangle){cx, is_selected ? hy - 22.0f : hy, hcard_w, hcard_h});
            draw_playing_card(&state->hand[i], (Rectangle){cx, hy, hcard_w, hcard_h},
                              is_selected, hovered, false, 0, client->puffer);

            if (hovered) {
                client->show_tooltip = true;
                if (state->hand[i].enhancement == ENHANCEMENT_STONE) {
                    snprintf(client->tooltip_title, sizeof(client->tooltip_title), "Stone Card");
                    snprintf(client->tooltip_type, sizeof(client->tooltip_type), "Stone (+50 Chips, No Rank)");
                    snprintf(client->tooltip_desc, sizeof(client->tooltip_desc), "Adds +50 Chips when scored - no rank or suit (Slot %d)", i + 1);
                    snprintf(client->tooltip_stats, sizeof(client->tooltip_stats), "Edition: %s | Seal: %s",
                             get_edition_name(state->hand[i].edition), get_seal_name(state->hand[i].seal));
                } else {
                    snprintf(client->tooltip_title, sizeof(client->tooltip_title), "%s of %s",
                             get_rank_str(state->hand[i].rank), get_suit_name(state->hand[i].suit));
                    snprintf(client->tooltip_type, sizeof(client->tooltip_type), "%s | %s",
                             get_enhancement_name(state->hand[i].enhancement), get_edition_name(state->hand[i].edition));
                    snprintf(client->tooltip_desc, sizeof(client->tooltip_desc), "Pack target (Slot %d)", i + 1);
                    snprintf(client->tooltip_stats, sizeof(client->tooltip_stats), "Seal: %s", get_seal_name(state->hand[i].seal));
                }
                client->tooltip_pos = (Vector2){mouse.x + 16, mouse.y + 16};

                if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                    if (is_selected) {
                        if (!(state->hand[i].flags & CARD_FORCED)) {
                            for (int s = 0; s < *selected_count; ++s)
                                if (selected_indices[s] == i) {
                                    for (int k = s; k < *selected_count - 1; ++k)
                                        selected_indices[k] = selected_indices[k + 1];
                                    (*selected_count)--;
                                    break;
                                }
                        }
                    } else if (*selected_count < MAX_SELECTION) {
                        selected_indices[(*selected_count)++] = (uint8_t)i;
                    }
                }
            }
        }
        char sel_buf[64];
        snprintf(sel_buf, sizeof(sel_buf), "%d card(s) selected - click a pack card to use it on them", *selected_count);
        DrawText(sel_buf, (int)(rect.x + rect.width * 0.5f - MeasureText(sel_buf, 12) * 0.5f),
                 (int)(hy + hcard_h + 8.0f), 12, Fade(BALATRO_GOLD, 0.85f));
    }
}

/* Phase 5: Game Over View */
static void draw_phase_game_over(const State *state, Rectangle rect, Action *out_action, Vector2 mouse, Client *client) {
    DrawRectangleRounded(rect, 0.08f, 4, Fade(BLACK, 0.8f));

    float cx = rect.x + rect.width * 0.5f;
    float cy = rect.y + rect.height * 0.35f;

    if (state->won) {
        DrawText("VICTORY! RUN WON!", (int)(cx - 150), (int)cy, 32, BALATRO_GOLD);
        DrawText(TextFormat("Ante %d Conquered!", state->ante), (int)(cx - 80), (int)(cy + 45), 18, BALATRO_WHITE);
    } else {
        DrawText("GAME OVER - DEFEATED", (int)(cx - 190), (int)cy, 32, BALATRO_RED_MULT);
        DrawText(TextFormat("Defeated on Ante %d", state->ante), (int)(cx - 90), (int)(cy + 45), 18, (Color){200, 200, 200, 255});
    }

    char stats_buf[128];
    snprintf(stats_buf, sizeof(stats_buf), "Total Dollars: $%d  |  Rounds Played: %d  |  Jokers Owned: %d", state->dollars, state->round, state->joker_count);
    DrawText(stats_buf, (int)(cx - (MeasureText(stats_buf, 14) / 2)), (int)(cy + 90), 14, (Color){180, 220, 240, 255});

    Rectangle rst_btn = {cx - 90, cy + 140, 180, 44};
    bool rst_hov = CheckCollisionPointRec(mouse, rst_btn);
    DrawRectangleRounded(rst_btn, 0.25f, 2, rst_hov ? (Color){40, 200, 100, 255} : (Color){25, 160, 80, 255});
    DrawText("START NEW RUN (R)", (int)(rst_btn.x + 20), (int)(rst_btn.y + 14), 14, BALATRO_WHITE);
    if (rst_hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        /* Reset trigger handled in main loop */
        out_action->type = ACTION_CASH_OUT;
    }
}

static void draw_multiline_text(const char *text, int x, int y, int max_w, int font_size, Color color) {
    if (!text || !text[0]) return;
    char buffer[256];
    strncpy(buffer, text, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';

    char *start = buffer;
    char *word = strtok(start, " ");
    char line[256] = {0};
    int cur_y = y;

    while (word != NULL) {
        char test_line[256];
        if (line[0] == '\0') {
            snprintf(test_line, sizeof(test_line), "%s", word);
        } else {
            snprintf(test_line, sizeof(test_line), "%s %s", line, word);
        }

        int width = MeasureText(test_line, font_size);
        if (width > max_w && line[0] != '\0') {
            DrawText(line, x, cur_y, font_size, color);
            cur_y += font_size + 3;
            snprintf(line, sizeof(line), "%s", word);
        } else {
            snprintf(line, sizeof(line), "%s", test_line);
        }
        word = strtok(NULL, " ");
    }
    if (line[0] != '\0') {
        DrawText(line, x, cur_y, font_size, color);
    }
}

/* Tooltip detail panel */
static void draw_tooltip(const Client *client) {
    if (!client->show_tooltip) return;

    float tw = 300.0f;
    char desc[sizeof(client->tooltip_desc)];
    snprintf(desc, sizeof(desc), "%s", client->tooltip_desc);
    char line[sizeof(client->tooltip_desc)] = {0};
    int lines = 1;
    char *word = strtok(desc, " ");
    while (word) {
        char next[sizeof(client->tooltip_desc)];
        if (line[0])
            snprintf(next, sizeof(next), "%s %s", line, word);
        else
            snprintf(next, sizeof(next), "%s", word);
        if (MeasureText(next, 10) > (int)tw - 24) {
            lines++;
            snprintf(line, sizeof(line), "%s", word);
        } else {
            snprintf(line, sizeof(line), "%s", next);
        }
        word = strtok(NULL, " ");
    }
    float th = 86.0f + (lines - 1) * 13.0f;
    if (th < 112.0f) th = 112.0f;
    float tx = client->tooltip_pos.x;
    float ty = client->tooltip_pos.y;

    if (tx + tw > client->width - 10) tx = client->width - tw - 10;
    if (ty + th > client->height - 10) ty = client->height - th - 10;
    if (tx < 10) tx = 10;
    if (ty < 10) ty = 10;

    Rectangle tr = {tx, ty, tw, th};
    DrawRectangleRounded((Rectangle){tx + 5, ty + 6, tw, th}, 0.09f, 4, Fade(BLACK, 0.48f));
    DrawRectangleRounded(tr, 0.09f, 4, (Color){12, 27, 29, 252});
    DrawRectangleRounded((Rectangle){tx + 2, ty + 2, tw - 4, 38}, 0.09f, 4, (Color){24, 49, 47, 255});
    DrawRectangle((int)(tx + 2), (int)(ty + 25), (int)(tw - 4), 15, (Color){24, 49, 47, 255});
    DrawRectangleRoundedLinesEx(tr, 0.09f, 4, 1.5f, Fade(BALATRO_GOLD, 0.88f));
    DrawRectangle((int)(tx + 10), (int)(ty + 39), (int)(tw - 20), 1, Fade(BALATRO_GOLD, 0.42f));

    DrawCircle((int)(tx + 15), (int)(ty + 14), 3.0f, BALATRO_GOLD);
    DrawText(client->tooltip_title, (int)(tx + 24), (int)(ty + 7), 14, BALATRO_GOLD);
    DrawText(client->tooltip_type, (int)(tx + 12), (int)(ty + 25), 9, (Color){157, 190, 194, 255});
    draw_multiline_text(client->tooltip_desc, (int)(tx + 12), (int)(ty + 48), (int)(tw - 24), 10, BALATRO_WHITE);

    Rectangle stats = {tx + 9, ty + th - 25, tw - 18, 18};
    DrawRectangleRounded(stats, 0.35f, 3, (Color){20, 62, 50, 255});
    int stats_font = 10;
    while (stats_font > 7 && MeasureText(client->tooltip_stats, stats_font) > stats.width - 12) stats_font--;
    DrawText(client->tooltip_stats, (int)(stats.x + 7),
             (int)(stats.y + (stats.height - stats_font) * 0.5f), stats_font, (Color){145, 235, 180, 255});
}

static void draw_controls_section(Rectangle area, const char *title,
                                  const char *const *keys, const char *const *descriptions,
                                  int count) {
    DrawRectangleRounded(area, 0.05f, 3, BALATRO_HEADER_BG);
    DrawRectangleRoundedLinesEx(area, 0.05f, 3, 1.0f, BALATRO_PANEL_BORDER);
    DrawText(title, (int)(area.x + 12), (int)(area.y + 10), 14, BALATRO_COLOR_PLANET_TEXT);
    DrawLine((int)(area.x + 12), (int)(area.y + 31),
             (int)(area.x + area.width - 12), (int)(area.y + 31), Fade(BALATRO_WHITE, 0.12f));
    for (int i = 0; i < count; ++i) {
        float row_y = area.y + 42.0f + i * 25.0f;
        Rectangle key_box = {area.x + 12, row_y, 70, 19};
        DrawRectangleRounded(key_box, 0.25f, 2, (Color){35, 61, 59, 255});
        DrawText(keys[i], (int)(key_box.x + (key_box.width - MeasureText(keys[i], 9)) * 0.5f),
                 (int)(key_box.y + 5), 9, BALATRO_GOLD);
        DrawText(descriptions[i], (int)(area.x + 92), (int)(row_y + 4), 10, BALATRO_WHITE);
    }
}

/* ========================================================================= */
/* Client Creation & Main Renderer Entry Point                               */
/* ========================================================================= */

static inline Client *make_client(int width, int height) {
    Client *client = (Client *)calloc(1, sizeof(Client));
    client->width = width > 0 ? width : 1280;
    client->height = height > 0 ? height : 720;
    client->auto_play = false;
    client->auto_play_timer = 0;
    client->step_delay_frames = 20;
    client->paused = false;
    client->forced_index = -1;

    InitWindow(client->width, client->height, "PufferLib Balatro");
    SetTargetFPS(60);
    client->puffer = LoadTexture("resources/shared/puffers_128.png");
    assert(client->puffer.id > 0);
    return client;
}

static inline void close_client(Client *client) {
    if (!client) return;
    UnloadTexture(client->puffer);
    CloseWindow();
    free(client);
}

static inline void balatro_render(Env *env) {
    if (IsKeyDown(KEY_ESCAPE) && env->state.phase == PHASE_GAME_OVER) {
        exit(0);
    }

    if (!env->client) {
        env->client = make_client(1280, 720);
    }

    Client *client = env->client;
    client->show_tooltip = false;
    if (client->show_run_menu && IsKeyPressed(KEY_ESCAPE)) client->show_run_menu = false;
    float playback_rate = client->step_delay_frames == 0
        ? 8.0f : 20.0f / client->step_delay_frames;
    Vector2 mouse = GetMousePosition();
    Action action = {0};
    action.type = UINT8_MAX;

    /* Keyboard Shortcuts */
    if (IsKeyPressed(KEY_TAB)) {
        client->auto_play = !client->auto_play;
    }
    if (IsKeyPressed(KEY_P)) {
        client->paused = !client->paused;
    }
    /* Frozen frames still run the full body (hover, tooltips, run menu)
       but must not advance the sim: gate dispatch and anim clocks on it.
       N steps exactly one frame while frozen. */
    bool frozen = client->paused;
    bool step_once = frozen && IsKeyPressed(KEY_N);
    bool advance = !frozen || step_once;
    if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_RIGHT_BRACKET)) {
        if (client->step_delay_frames > 0) client->step_delay_frames -= 5;
    }
    if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_LEFT_BRACKET)) {
        client->step_delay_frames += 5;
    }
    if (IsKeyPressed(KEY_R) && env->state.phase == PHASE_GAME_OVER) {
        puf_reset(env);
        client->selected_count = 0;
    }
    /* Entering the hand-play phase with a fresh hand drops leftover
       selection (e.g. pack-targeting picks carried into the next hand). */
    if (env->state.phase == PHASE_SELECTING_HAND && client->last_phase != PHASE_SELECTING_HAND)
        client->selected_count = 0;
    client->last_phase = env->state.phase;

    if (!client->score_anim.active && !client->consumable_anim.active && env->state.phase == PHASE_SELECTING_HAND) {
        /* Cerulean Bell: exactly the forced card is auto-selected. When the
           hand is reordered (sort/swap) the old forced slot goes stale and
           must be dropped, or two cards stay highlighted. */
        int forced = -1;
        for (int k = 0; k < env->state.hand_count; ++k)
            if (env->state.hand[k].flags & CARD_FORCED) {
                forced = k;
                break;
            }
        if (client->forced_index >= 0 && client->forced_index != forced) {
            for (int s = 0; s < client->selected_count; ++s)
                if (client->selected_indices[s] == client->forced_index) {
                    for (int m = s; m < client->selected_count - 1; ++m)
                        client->selected_indices[m] = client->selected_indices[m + 1];
                    client->selected_count--;
                    break;
                }
        }
        client->forced_index = forced;
        if (forced >= 0) {
            bool found = false;
            for (int s = 0; s < client->selected_count; ++s)
                if (client->selected_indices[s] == forced) {
                    found = true;
                    break;
                }
            if (!found && client->selected_count < MAX_SELECTION)
                client->selected_indices[client->selected_count++] = (uint8_t)forced;
        }
        /* 1..9 to toggle hand cards */
        for (int k = 0; k < 9; ++k) {
            if (IsKeyPressed(KEY_ONE + k) && k < env->state.hand_count) {
                bool found = false;
                for (int s = 0; s < client->selected_count; ++s) {
                    if (client->selected_indices[s] == k) {
                        found = true;
                        if (env->state.hand[k].flags & CARD_FORCED) break; /* cannot deselect */
                        for (int m = s; m < client->selected_count - 1; ++m) client->selected_indices[m] = client->selected_indices[m + 1];
                        client->selected_count--;
                        break;
                    }
                }
                if (!found && client->selected_count < MAX_SELECTION) {
                    client->selected_indices[client->selected_count++] = (uint8_t)k;
                }
            }
        }
        if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_ENTER)) {
            if (client->selected_count > 0) {
                action.type = ACTION_PLAY_HAND;
                action.selection_count = (uint8_t)client->selected_count;
                for (int i = 0; i < client->selected_count; ++i) action.selection[i] = client->selected_indices[i];
                client->selected_count = 0;
            }
        }
        if (IsKeyPressed(KEY_D) && client->selected_count > 0 && env->state.discards_left > 0) {
            action.type = ACTION_DISCARD;
            action.selection_count = (uint8_t)client->selected_count;
            for (int i = 0; i < client->selected_count; ++i) action.selection[i] = client->selected_indices[i];
            client->selected_count = 0;
        }
        if (IsKeyPressed(KEY_R)) action.type = ACTION_SORT_HAND_RANK;
        if (IsKeyPressed(KEY_S)) action.type = ACTION_SORT_HAND_SUIT;
    } else if (!client->score_anim.active && !client->consumable_anim.active && env->state.phase == PHASE_BLIND_SELECT) {
        if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_ENTER)) action.type = ACTION_SELECT_BLIND;
    } else if (!client->score_anim.active && !client->consumable_anim.active && env->state.phase == PHASE_ROUND_EVAL) {
        if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_ENTER)) action.type = ACTION_CASH_OUT;
    } else if (!client->score_anim.active && !client->consumable_anim.active && env->state.phase == PHASE_SHOP) {
        if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_ENTER)) action.type = ACTION_NEXT_ROUND;
    } else if (!client->score_anim.active && !client->consumable_anim.active && env->state.phase == PHASE_PACK_OPENING &&
               (env->state.pack_kind == PACK_TAROT || env->state.pack_kind == PACK_SPECTRAL)) {
        /* 1..9 to toggle hand target cards */
        for (int k = 0; k < 9; ++k) {
            if (IsKeyPressed(KEY_ONE + k) && k < env->state.hand_count) {
                bool found = false;
                for (int s = 0; s < client->selected_count; ++s)
                    if (client->selected_indices[s] == k) {
                        found = true;
                        if (env->state.hand[k].flags & CARD_FORCED) break;
                        for (int m = s; m < client->selected_count - 1; ++m)
                            client->selected_indices[m] = client->selected_indices[m + 1];
                        client->selected_count--;
                        break;
                    }
                if (!found && client->selected_count < MAX_SELECTION)
                    client->selected_indices[client->selected_count++] = (uint8_t)k;
            }
        }
    }

    BeginDrawing();
    /* Layered casino-felt backdrop with a quiet diamond weave. */
    DrawRectangleGradientV(0, 0, client->width, client->height,
                           (Color){15, 43, 38, 255}, (Color){5, 15, 17, 255});
    for (int y = 26; y < client->height; y += 48) {
        for (int x = (y / 48 & 1) ? 24 : 0; x < client->width; x += 48) {
            DrawLine(x, y - 5, x + 5, y, Fade((Color){115, 190, 160, 255}, 0.055f));
            DrawLine(x + 5, y, x, y + 5, Fade((Color){115, 190, 160, 255}, 0.055f));
            DrawLine(x, y + 5, x - 5, y, Fade((Color){115, 190, 160, 255}, 0.055f));
            DrawLine(x - 5, y, x, y - 5, Fade((Color){115, 190, 160, 255}, 0.055f));
        }
    }
    DrawCircleGradient((int)(client->width * 0.5f), (int)(client->height * 0.42f),
                       (float)(client->height * 0.75f), (Color){45, 105, 82, 62}, (Color){4, 8, 12, 0});

    /* Header Bar */
    DrawRectangle(0, 0, client->width, 34, (Color){6, 13, 17, 245});
    DrawRectangleGradientH(0, 32, client->width, 2, BALATRO_GOLD, Fade(BALATRO_BLUE_CHIPS, 0.15f));
    DrawText("PUFFER", 16, 7, 18, BALATRO_WHITE);
    int balatro_x = 16 + MeasureText("PUFFER", 18) + 6;
    DrawText("BALATRO", balatro_x, 7, 18, BALATRO_GOLD);
    const char *phase_name = "BLIND SELECT";
    if (client->score_anim.active) phase_name = "SCORING";
    else if (client->consumable_anim.active) phase_name = "CARD EFFECT";
    else if (env->state.phase == PHASE_SELECTING_HAND) phase_name = "PLAYING BLIND";
    else if (env->state.phase == PHASE_ROUND_EVAL) phase_name = "ROUND COMPLETE";
    else if (env->state.phase == PHASE_SHOP) phase_name = "THE SHOP";
    else if (env->state.phase == PHASE_PACK_OPENING) phase_name = "BOOSTER PACK";
    else if (env->state.phase == PHASE_GAME_OVER) phase_name = "RUN OVER";
    int phase_width = MeasureText(phase_name, 11) + 22;
    int phase_x = balatro_x + MeasureText("BALATRO", 18) + 24;
    DrawRectangleRounded((Rectangle){phase_x, 7, phase_width, 20}, 0.5f, 3, (Color){25, 53, 55, 255});
    DrawText(phase_name, phase_x + 11, 12, 11, (Color){164, 220, 205, 255});
    /* Playback speed (right-aligned pill): multiplier from step delay. */
    const char *speed_text = client->paused ? "PAUSED" : TextFormat("%.1fx", playback_rate);
    int speed_width = MeasureText(speed_text, 11) + 22;
    int speed_x = client->width - 16 - speed_width;
    DrawRectangleRounded((Rectangle){speed_x, 7, speed_width, 20}, 0.5f, 3, (Color){25, 53, 55, 255});
    DrawText(speed_text, speed_x + 11, 12, 11,
             client->paused ? BALATRO_RED_MULT : (Color){164, 220, 205, 255});

    /* Layout Geometry */
    float sidebar_w = 260.0f;
    Rectangle sidebar_r = {10, 42, sidebar_w, client->height - 52};
    draw_hud_sidebar(&env->state, &env->legal_masks, sidebar_r, client->selected_count,
                     client->selected_indices, &action, mouse, client);

    float top_x = sidebar_w + 20.0f;
    float top_w = client->width - top_x - 10.0f;
    float jokers_w = top_w * 0.65f;
    float cons_w = top_w - jokers_w - 10.0f;

    /* Jokers & Consumables Bars */
    Rectangle jokers_r = {top_x, 42, jokers_w, 130};
    draw_jokers_bar(&env->state, jokers_r, &action, mouse, client);

    Rectangle cons_r = {top_x + jokers_w + 10.0f, 42, cons_w, 130};
    draw_consumables_bar(&env->state, cons_r, &action, mouse, client);

    /* Main Center Viewport by Phase (scoring animation wins while active) */
    Rectangle main_r = {top_x, 182, top_w, client->height - 192};
    DrawRectangleRounded((Rectangle){main_r.x + 4, main_r.y + 6, main_r.width, main_r.height}, 0.025f, 4, Fade(BLACK, 0.30f));
    DrawRectangleRounded(main_r, 0.025f, 4, Fade((Color){8, 24, 24, 255}, 0.48f));
    DrawRectangleRoundedLinesEx(main_r, 0.025f, 4, 1.0f, Fade(BALATRO_PANEL_BORDER, 0.75f));
    if (client->score_anim.active) {
        draw_phase_scoring(client, main_r);
    } else if (client->consumable_anim.active) {
        const ConsumableAnim *anim = &client->consumable_anim;
        float progress = anim->t / anim->duration;
        if (progress > 1.0f) progress = 1.0f;
        float transition = (progress - 0.14f) / 0.58f;
        if (transition < 0.0f) transition = 0.0f;
        if (transition > 1.0f) transition = 1.0f;
        float eased = ease_out_cubic(transition);
        uint8_t set = centers[anim->used.center_id].set;
        Color magic = set == SET_SPECTRAL ? (Color){124, 235, 226, 255} : (Color){205, 112, 255, 255};

        DrawRectangleRounded(main_r, 0.025f, 4, (Color){7, 13, 24, 248});
        float center_x = main_r.x + main_r.width * 0.5f;
        float center_y = main_r.y + main_r.height * 0.56f;
        for (int ring = 0; ring < 5; ++ring) {
            float radius = 90.0f + ring * 76.0f + sinf(anim->t * 3.0f + ring) * 9.0f;
            DrawCircleLines((int)center_x, (int)center_y, radius,
                            Fade(ring & 1 ? magic : BALATRO_GOLD, 0.09f));
        }
        for (int spark = 0; spark < 24; ++spark) {
            float angle = spark * PI / 12.0f + anim->t * (spark & 1 ? 0.7f : -0.5f);
            float radius = 115.0f + (spark % 6) * 47.0f;
            DrawCircleV((Vector2){center_x + cosf(angle) * radius, center_y + sinf(angle) * radius * 0.55f},
                        spark % 3 == 0 ? 1.8f : 1.0f, Fade(magic, 0.32f));
        }

        const char *name = get_center_name(anim->used.center_id);
        DrawText(name, (int)(center_x - MeasureText(name, 30) * 0.5f), (int)(main_r.y + 22), 30, BALATRO_GOLD);
        const char *kind = set == SET_SPECTRAL ? "SPECTRAL EFFECT" : set == SET_PLANET ? "PLANET EFFECT" : "TAROT EFFECT";
        DrawText(kind, (int)(center_x - MeasureText(kind, 11) * 0.5f), (int)(main_r.y + 57), 11, magic);
        DrawRectangle((int)(center_x - 58), (int)(main_r.y + 75), 116, 2, Fade(magic, 0.62f));

        if (anim->hand_changed) {
            float card_width = 70.0f;
            float card_height = 100.0f;
            float before_pitch = anim->before_hand_count > 1
                ? fminf(card_width + 8.0f, (main_r.width - 70.0f - card_width) / (anim->before_hand_count - 1)) : 0.0f;
            float after_pitch = anim->after_hand_count > 1
                ? fminf(card_width + 8.0f, (main_r.width - 70.0f - card_width) / (anim->after_hand_count - 1)) : 0.0f;
            float before_start = center_x - ((anim->before_hand_count - 1) * before_pitch + card_width) * 0.5f;
            float after_start = center_x - ((anim->after_hand_count - 1) * after_pitch + card_width) * 0.5f;
            int removed = 0;
            int added = 0;
            int changed = 0;

            if (anim->used.center_id == CENTER_C_DEATH && anim->selected_count == 2) {
                int left = find_card(anim->before_hand, anim->before_hand_count, anim->selected_ids[0]);
                int right = find_card(anim->before_hand, anim->before_hand_count, anim->selected_ids[1]);
                if (left >= 0 && right >= 0 && transition > 0.05f && transition < 0.92f) {
                    float left_x = before_start + left * before_pitch + card_width * 0.5f;
                    float right_x = before_start + right * before_pitch + card_width * 0.5f;
                    float pulse = sinf(transition * PI);
                    for (int arc = 0; arc < 3; ++arc)
                        DrawLineEx((Vector2){right_x, center_y - 62.0f + arc * 4.0f},
                                   (Vector2){left_x, center_y - 62.0f + arc * 4.0f},
                                   1.0f + pulse * 2.0f, Fade(magic, pulse * (0.8f - arc * 0.15f)));
                }
            }

            for (int i = 0; i < anim->before_hand_count; ++i) {
                const Card *before = &anim->before_hand[i];
                int after_index = find_card(anim->after_hand, anim->after_hand_count, before->sort_id);
                float before_x = before_start + i * before_pitch + card_width * 0.5f;
                int targeted = 0;
                for (int target = 0; target < anim->selected_count; ++target)
                    targeted |= anim->selected_ids[target] == before->sort_id;
                if (after_index >= 0) {
                    const Card *after = &anim->after_hand[after_index];
                    float after_x = after_start + after_index * after_pitch + card_width * 0.5f;
                    float x = before_x + (after_x - before_x) * eased;
                    if (!same_card(before, after)) {
                        changed++;
                        float flip = transition < 0.5f ? 1.0f - transition * 2.0f : transition * 2.0f - 1.0f;
                        if (flip < 0.04f) flip = 0.04f;
                        draw_playing_transform(transition < 0.5f ? before : after, x, center_y, card_width, card_height,
                                               flip, 1.0f, 0.0f, client->puffer);
                        float glow = sinf(transition * PI);
                        DrawRectangleRoundedLinesEx((Rectangle){x - card_width * 0.5f - 4, center_y - card_height * 0.5f - 4,
                                                                card_width + 8, card_height + 8},
                                                    0.08f, 4, 2.5f, Fade(magic, glow * 0.9f));
                    } else {
                        draw_playing_transform(before, x, center_y, card_width, card_height, 1.0f, 1.0f, 0.0f, client->puffer);
                    }
                    if (targeted && transition < 0.42f)
                        DrawRectangleRoundedLinesEx((Rectangle){x - card_width * 0.5f - 3, center_y - card_height * 0.5f - 3,
                                                                card_width + 6, card_height + 6},
                                                    0.08f, 4, 2.0f, Fade(BALATRO_GOLD, 1.0f - transition));
                } else {
                    removed++;
                    float scale = 1.0f - eased;
                    if (scale > 0.03f)
                        draw_playing_transform(before, before_x, center_y - eased * 26.0f, card_width, card_height,
                                               scale, scale, (i & 1 ? 18.0f : -18.0f) * eased, client->puffer);
                    float burst = sinf(transition * PI);
                    for (int shard = 0; shard < 8; ++shard) {
                        float angle = shard * PI * 0.25f + i * 0.31f;
                        float inner = 20.0f + eased * 22.0f;
                        float outer = inner + 12.0f + burst * 18.0f;
                        DrawLineEx((Vector2){before_x + cosf(angle) * inner, center_y + sinf(angle) * inner},
                                   (Vector2){before_x + cosf(angle) * outer, center_y + sinf(angle) * outer},
                                   2.0f, Fade(set == SET_SPECTRAL ? magic : BALATRO_RED_MULT, burst * 0.85f));
                    }
                }
            }

            for (int i = 0; i < anim->after_hand_count; ++i) {
                const Card *after = &anim->after_hand[i];
                if (find_card(anim->before_hand, anim->before_hand_count, after->sort_id) >= 0) continue;
                added++;
                float x = after_start + i * after_pitch + card_width * 0.5f;
                float scale = ease_out_back(transition);
                if (scale < 0.03f) continue;
                draw_playing_transform(after, x, center_y + (1.0f - eased) * 55.0f, card_width, card_height,
                                       scale, scale, (1.0f - eased) * 12.0f, client->puffer);
                for (int sparkle = 0; sparkle < 6; ++sparkle) {
                    float angle = sparkle * PI / 3.0f + anim->t;
                    float radius = 45.0f + sinf(transition * PI) * 16.0f;
                    DrawCircleV((Vector2){x + cosf(angle) * radius, center_y + sinf(angle) * radius},
                                2.0f, Fade(magic, sinf(transition * PI) * 0.9f));
                }
            }

            char outcome[96];
            if (anim->used.center_id == CENTER_C_DEATH)
                snprintf(outcome, sizeof(outcome), "LEFT CARD BECAME THE RIGHT CARD");
            else if (removed && added)
                snprintf(outcome, sizeof(outcome), "%d DESTROYED  /  %d CREATED", removed, added);
            else if (removed)
                snprintf(outcome, sizeof(outcome), "%d CARD%s DESTROYED", removed, removed == 1 ? "" : "S");
            else if (added)
                snprintf(outcome, sizeof(outcome), "%d CARD%s CREATED", added, added == 1 ? "" : "S");
            else
                snprintf(outcome, sizeof(outcome), "%d CARD%s TRANSFORMED", changed, changed == 1 ? "" : "S");
            float reveal = (progress - 0.64f) / 0.16f;
            if (reveal > 0.0f) {
                if (reveal > 1.0f) reveal = 1.0f;
                DrawText(outcome, (int)(center_x - MeasureText(outcome, 15) * 0.5f),
                         (int)(center_y + card_height * 0.5f + 34), 15, Fade(magic, reveal));
            }
        } else if (anim->jokers_changed) {
            float card_width = 76.0f;
            float card_height = 104.0f;
            float before_pitch = anim->before_joker_count > 1
                ? fminf(card_width + 10.0f, (main_r.width - 70.0f - card_width) / (anim->before_joker_count - 1)) : 0.0f;
            float after_pitch = anim->after_joker_count > 1
                ? fminf(card_width + 10.0f, (main_r.width - 70.0f - card_width) / (anim->after_joker_count - 1)) : 0.0f;
            float before_start = center_x - ((anim->before_joker_count - 1) * before_pitch + card_width) * 0.5f;
            float after_start = center_x - ((anim->after_joker_count - 1) * after_pitch + card_width) * 0.5f;

            for (int i = 0; i < anim->before_joker_count; ++i) {
                const Card *before = &anim->before_jokers[i];
                int after_index = find_card(anim->after_jokers, anim->after_joker_count, before->sort_id);
                float before_x = before_start + i * before_pitch + card_width * 0.5f;
                if (after_index >= 0) {
                    const Card *after = &anim->after_jokers[after_index];
                    float after_x = after_start + after_index * after_pitch + card_width * 0.5f;
                    float x = before_x + (after_x - before_x) * eased;
                    float flip = same_card(before, after) ? 1.0f
                        : transition < 0.5f ? 1.0f - transition * 2.0f : transition * 2.0f - 1.0f;
                    if (flip < 0.04f) flip = 0.04f;
                    draw_joker_transform(transition < 0.5f ? before : after, x, center_y, card_width, card_height,
                                         flip, 1.0f, 0.0f, client->puffer);
                    if (!same_card(before, after))
                        DrawRectangleRoundedLinesEx((Rectangle){x - card_width * 0.5f - 4, center_y - card_height * 0.5f - 4,
                                                                card_width + 8, card_height + 8},
                                                    0.08f, 4, 2.5f, Fade(magic, sinf(transition * PI)));
                } else {
                    float scale = 1.0f - eased;
                    if (scale > 0.03f)
                        draw_joker_transform(before, before_x, center_y, card_width, card_height, scale, scale,
                                             (i & 1 ? 16.0f : -16.0f) * eased, client->puffer);
                }
            }
            for (int i = 0; i < anim->after_joker_count; ++i) {
                const Card *after = &anim->after_jokers[i];
                if (find_card(anim->before_jokers, anim->before_joker_count, after->sort_id) >= 0) continue;
                float x = after_start + i * after_pitch + card_width * 0.5f;
                float scale = ease_out_back(transition);
                if (scale > 0.03f)
                    draw_joker_transform(after, x, center_y + (1.0f - eased) * 55.0f, card_width, card_height,
                                         scale, scale, 0.0f, client->puffer);
            }
            const char *outcome = "JOKERS TRANSFORMED";
            DrawText(outcome, (int)(center_x - MeasureText(outcome, 15) * 0.5f),
                     (int)(center_y + card_height * 0.5f + 34), 15, Fade(magic, progress > 0.65f ? 1.0f : 0.0f));
        } else {
            float pulse = 0.92f + 0.08f * sinf(anim->t * 6.0f);
            draw_consumable_card(&anim->used,
                                 (Rectangle){center_x - 42.0f * pulse, center_y - 60.0f * pulse,
                                             84.0f * pulse, 120.0f * pulse},
                                 false, 0, false, 0);
            char outcome[128] = "EFFECT RESOLVED";
            int upgraded = 0, first_hand = -1;
            for (int hand = 0; hand < HAND_COUNT; ++hand)
                if (anim->after_levels[hand] > anim->before_levels[hand]) {
                    if (first_hand < 0) first_hand = hand;
                    upgraded++;
                }
            if (upgraded == 1)
                snprintf(outcome, sizeof(outcome), "%s  LEVEL %d -> %d", get_hand_type_name((HandType)first_hand),
                         anim->before_levels[first_hand], anim->after_levels[first_hand]);
            else if (upgraded == HAND_COUNT)
                snprintf(outcome, sizeof(outcome), "ALL HANDS  +%d LEVEL",
                         anim->after_levels[first_hand] - anim->before_levels[first_hand]);
            else if (upgraded > 1)
                snprintf(outcome, sizeof(outcome), "%d HAND TYPES UPGRADED", upgraded);
            /* Buy-and-use / pack purchases move money in the same step as the
               card effect; the money delta must not clobber the effect line.
               Money-only effects (Hermit, Temperance) keep the single line. */
            int money_changed = anim->after_dollars != anim->before_dollars;
            if (upgraded == 0 && money_changed)
                snprintf(outcome, sizeof(outcome), "MONEY  $%d -> $%d", anim->before_dollars, anim->after_dollars);
            int outcome_y = (int)center_y + (upgraded > 0 ? 76 : 88);
            DrawText(outcome, (int)(center_x - MeasureText(outcome, 16) * 0.5f),
                     outcome_y, 16, magic);
            if (upgraded > 0 && money_changed) {
                char money[48];
                snprintf(money, sizeof(money), "MONEY  $%d -> $%d", anim->before_dollars, anim->after_dollars);
                DrawText(money, (int)(center_x - MeasureText(money, 13) * 0.5f),
                         (int)center_y + 102, 13, BALATRO_GOLD);
            }
        }
    } else {
        switch (env->state.phase) {
            case PHASE_BLIND_SELECT:
                draw_phase_blind_select(&env->state, &env->legal_masks, main_r, &action, mouse, client);
                break;
            case PHASE_SELECTING_HAND:
                draw_phase_hand_play(&env->state, &env->legal_masks, main_r, &client->selected_count, client->selected_indices, &action, mouse, client);
                break;
            case PHASE_ROUND_EVAL:
                draw_phase_round_eval(&env->state, main_r, &action, mouse, client);
                break;
            case PHASE_SHOP:
                draw_phase_shop(&env->state, &env->legal_masks, main_r, &action, mouse, client);
                break;
            case PHASE_PACK_OPENING:
                draw_phase_pack_opening(&env->state, &env->legal_masks, main_r, &client->selected_count, client->selected_indices, &action, mouse, client);
                break;
            case PHASE_GAME_OVER:
                draw_phase_game_over(&env->state, main_r, &action, mouse, client);
                break;
        }
    }

    if (client->show_run_menu) {
        action.type = UINT8_MAX;
        client->show_tooltip = false;
        DrawRectangle(0, 34, client->width, client->height - 34, Fade(BLACK, 0.72f));
        Rectangle menu = {client->width * 0.5f - 330.0f, 70, 660, 580};
        DrawRectangleRounded(menu, 0.04f, 4, (Color){15, 31, 31, 255});
        DrawRectangleRoundedLinesEx(menu, 0.04f, 4, 2.0f, BALATRO_GOLD);
        DrawText("GAME MENU", (int)(menu.x + 22), (int)(menu.y + 21), 20, BALATRO_GOLD);

        Rectangle run_tab = {menu.x + 174, menu.y + 14, 100, 30};
        Rectangle controls_tab = {menu.x + 282, menu.y + 14, 100, 30};
        bool run_hovered = CheckCollisionPointRec(mouse, run_tab);
        bool controls_hovered = CheckCollisionPointRec(mouse, controls_tab);
        DrawRectangleRounded(run_tab, 0.25f, 2,
            !client->show_controls_menu ? (Color){50, 91, 84, 255}
            : run_hovered ? (Color){39, 69, 65, 255} : BALATRO_HEADER_BG);
        DrawRectangleRounded(controls_tab, 0.25f, 2,
            client->show_controls_menu ? (Color){50, 91, 84, 255}
            : controls_hovered ? (Color){39, 69, 65, 255} : BALATRO_HEADER_BG);
        DrawText("RUN INFO", (int)(run_tab.x + (run_tab.width - MeasureText("RUN INFO", 10)) * 0.5f),
                 (int)(run_tab.y + 9), 10, BALATRO_WHITE);
        DrawText("CONTROLS", (int)(controls_tab.x + (controls_tab.width - MeasureText("CONTROLS", 10)) * 0.5f),
                 (int)(controls_tab.y + 9), 10, BALATRO_WHITE);
        if (run_hovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) client->show_controls_menu = false;
        if (controls_hovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) client->show_controls_menu = true;

        Rectangle close_button = {menu.x + menu.width - 92, menu.y + 14, 70, 30};
        bool close_hovered = CheckCollisionPointRec(mouse, close_button);
        DrawRectangleRounded(close_button, 0.25f, 2,
            close_hovered ? (Color){170, 55, 55, 255} : (Color){105, 44, 44, 255});
        DrawText("CLOSE", (int)(close_button.x + (close_button.width - MeasureText("CLOSE", 11)) * 0.5f),
                 (int)(close_button.y + 9), 11, BALATRO_WHITE);
        if (close_hovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) client->show_run_menu = false;

        if (client->show_controls_menu) {
            static const char *general_keys[] = {"P", "N", "[ / DOWN", "] / UP", "TAB", "ESC"};
            static const char *general_desc[] = {
                "Pause or resume", "Advance one paused frame", "Reduce playback speed",
                "Increase playback speed", "Toggle autoplay", "Close this menu"
            };
            static const char *hand_keys[] = {"1 - 9", "CLICK", "ENTER / SPACE", "D", "R", "S", "LEFT / RIGHT"};
            static const char *hand_desc[] = {
                "Toggle hand cards", "Toggle a hand card", "Play selected cards",
                "Discard selected cards", "Sort hand by rank", "Sort hand by suit", "Move hovered card"
            };
            static const char *pointer_keys[] = {
                "CLICK", "RIGHT-CLICK", "SHIFT + CLICK", "CLICK", "RIGHT-CLICK", "HOVER"
            };
            static const char *pointer_desc[] = {
                "Use consumable", "Sell consumable or Joker", "Sell hovered Joker",
                "Buy or choose item", "Sell where available", "Show card details"
            };
            static const char *phase_keys[] = {"ENTER / SPACE", "CLICK", "1 - 9", "R"};
            static const char *phase_desc[] = {
                "Confirm or continue phase", "Select Blind / cash out", "Choose pack targets",
                "Start new run after game over"
            };
            draw_controls_section((Rectangle){menu.x + 20, menu.y + 58, 304, 204},
                                  "GENERAL", general_keys, general_desc, 6);
            draw_controls_section((Rectangle){menu.x + 20, menu.y + 272, 304, 288},
                                  "HAND", hand_keys, hand_desc, 7);
            draw_controls_section((Rectangle){menu.x + 336, menu.y + 58, 304, 204},
                                  "MOUSE & ITEMS", pointer_keys, pointer_desc, 6);
            draw_controls_section((Rectangle){menu.x + 336, menu.y + 272, 304, 288},
                                  "PHASES & PACKS", phase_keys, phase_desc, 4);
        } else {
            DrawRectangleRounded((Rectangle){menu.x + 20, menu.y + 58, menu.width - 40, 88},
                                 0.08f, 3, BALATRO_HEADER_BG);
            DrawText(TextFormat("ANTE  %d / %d", env->state.ante, env->state.config.win_ante),
                     (int)(menu.x + 36), (int)(menu.y + 74), 14, BALATRO_GOLD);
            DrawText(TextFormat("ROUND  %d", env->state.round),
                     (int)(menu.x + 36), (int)(menu.y + 102), 13, BALATRO_WHITE);
            DrawText(TextFormat("HANDS PLAYED  %u", env->state.run_hands_played),
                     (int)(menu.x + 220), (int)(menu.y + 74), 13, BALATRO_WHITE);
            DrawText(TextFormat("BLINDS SKIPPED  %d", env->state.skips),
                     (int)(menu.x + 220), (int)(menu.y + 102), 13, BALATRO_WHITE);
            DrawText(TextFormat("DECK  %d CARDS", env->state.deck_count + env->state.hand_count + env->state.discard_count),
                     (int)(menu.x + 445), (int)(menu.y + 74), 13, BALATRO_WHITE);
            DrawText(TextFormat("MONEY  $%d", env->state.dollars),
                     (int)(menu.x + 445), (int)(menu.y + 102), 13, BALATRO_GOLD);

            DrawText("HAND LEVELS", (int)(menu.x + 22), (int)(menu.y + 166), 16, BALATRO_COLOR_PLANET_TEXT);
            DrawLine((int)(menu.x + 22), (int)(menu.y + 190),
                     (int)(menu.x + menu.width - 22), (int)(menu.y + 190), Fade(BALATRO_WHITE, 0.12f));
            int hand_rows = (HAND_COUNT + 1) / 2;
            for (int hand = 0; hand < HAND_COUNT; ++hand) {
                int column = hand / hand_rows;
                int row = hand % hand_rows;
                float row_x = menu.x + 22.0f + column * 310.0f;
                float row_y = menu.y + 202.0f + row * 54.0f;
                Rectangle hand_row = {row_x, row_y, 294, 44};
                DrawRectangleRounded(hand_row, 0.12f, 2,
                    row & 1 ? (Color){20, 42, 41, 255} : (Color){24, 48, 46, 255});
                int level = env->state.hand_levels[hand];
                assert(level > 0);
                int chips = 0;
                int mult = 0;
                hand_base_stats((HandType)hand, level, &chips, &mult);
                const char *hand_name = get_hand_type_name((HandType)hand);
                DrawText(hand_name, (int)(hand_row.x + 10), (int)(hand_row.y + 7), 12, BALATRO_WHITE);
                DrawText(TextFormat("LV %d", level), (int)(hand_row.x + hand_row.width - 48),
                         (int)(hand_row.y + 7), 11, BALATRO_COLOR_PLANET_TEXT);
                DrawText(TextFormat("%d chips  x  %d mult", chips, mult),
                         (int)(hand_row.x + 10), (int)(hand_row.y + 25), 9,
                         (Color){145, 175, 188, 255});
                DrawText(TextFormat("%d played", env->state.hand_plays[hand]),
                         (int)(hand_row.x + hand_row.width - 72), (int)(hand_row.y + 25), 9,
                         (Color){125, 150, 148, 255});
            }
        }
    }

    if (advance && (client->score_anim.active || client->consumable_anim.active)) {
        action.type = UINT8_MAX;
        client->show_tooltip = false;
    }

    /* Tooltip overlay */
    draw_tooltip(client);

    /* Paused: big red bars top-right, drawn over everything. */
    if (client->paused) {
        DrawRectangle(client->width - 72, 4, 24, 64, BALATRO_RED_MULT);
        DrawRectangle(client->width - 40, 4, 24, 64, BALATRO_RED_MULT);
    }

    EndDrawing();

    /* Dispatch action if triggered interactively */
    if (advance && action.type != UINT8_MAX) {
        if (action.type == ACTION_PLAY_HAND) {
            score_anim_capture(client, &env->state, &action);
        }
        if (action.type == ACTION_USE_CONSUMABLE || action.type == ACTION_BUY_AND_USE ||
            action.type == ACTION_PICK_PACK_CARD)
            capture_consumable(client, &env->state, &action);
        StepResult res = {0};
        int error = apply_step(&env->state, &action, &env->legal_masks, &res);
        assert(error == OK);
        if (action.type == ACTION_PLAY_HAND) {
            score_anim_start(client, &env->state);
        }
        if (client->consumable_anim.pending)
            start_consumable(client, &env->state);
        /* Consumables and pack picks can remove/reorder hand cards: drop the
           targeting selection so stale indices never outlive the hand. */
        if (action.type == ACTION_USE_CONSUMABLE || action.type == ACTION_BUY_AND_USE ||
            action.type == ACTION_PICK_PACK_CARD)
            client->selected_count = 0;
        puffer_observe(env);
    }

    /* Cosmetic and animation clocks; frozen while paused. */
    if (advance) {
        g_balatro_ui_time += GetFrameTime();
        if (client->score_anim.active) {
            float frame_time = GetFrameTime();
            if (frame_time > 1.0f / 30.0f) frame_time = 1.0f / 30.0f;
            client->score_anim.t += frame_time * playback_rate;
            if (client->score_anim.t >= client->score_anim.duration) {
                client->score_anim.active = false;
            }
        }
    }
    if (advance && client->consumable_anim.active) {
        float frame_time = GetFrameTime();
        if (frame_time > 1.0f / 30.0f) frame_time = 1.0f / 30.0f;
        client->consumable_anim.t += frame_time * playback_rate;
        if (client->consumable_anim.t >= client->consumable_anim.duration)
            client->consumable_anim.active = false;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* BALATRO_RENDER_H */
