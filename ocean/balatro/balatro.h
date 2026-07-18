#ifndef PUFFER_BALATRO_H
#define PUFFER_BALATRO_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "pufferenv.h"
#include <balatro_core.h>

/* PufferLib's tensor boundary is flat, while Simulatro's public observation
   is a fixed-layout structure. Encode its fields semantically rather than
   exposing C padding/endianness as neural-network features. The capacity is
   intentionally fixed so the Puffer tensor ABI remains stable. */
#define OBS_SIZE 8448
#define NUM_ATNS 9
#define ACT_SIZES {23, 64, 6, 64, 64, 64, 64, 64, 64}
#define ACTION_MASK_SIZE (23 + 64 + 6 + 64 + 64 + 64 + 64 + 64 + 64)

typedef float obs_t;

typedef struct Log {
    float score;
	float reward;
    float wins;
    float ante;
    float invalid_actions;
    float n;
} Log;

typedef struct Env {
    Log log;
    int num_agents;
    int tag;
    int boundary_reached;
    float episode_reward;
    unsigned int rng;
    Agent agents[1];
    BalatroConfig config;
    BalatroState state;
    BalatroAction legal[BALATRO_MAX_LEGAL_ACTIONS];
    int legal_count;
} Env;

typedef struct BalatroObservationEncoder {
    float *out;
    int count;
} BalatroObservationEncoder;

static void balatro_encode_float(BalatroObservationEncoder *encoder, float value) {
    encoder->out[encoder->count++] = isfinite(value) ? value : 0.0f;
}

static void balatro_encode_u8(BalatroObservationEncoder *encoder, uint8_t value) {
    balatro_encode_float(encoder, (float)value * (1.0f / 255.0f));
}

static void balatro_encode_u16(BalatroObservationEncoder *encoder, uint16_t value) {
    balatro_encode_float(encoder, (float)value * (1.0f / 65535.0f));
}

static void balatro_encode_u32(BalatroObservationEncoder *encoder, uint32_t value) {
    balatro_encode_u16(encoder, (uint16_t)value);
    balatro_encode_u16(encoder, (uint16_t)(value >> 16));
}

static void balatro_encode_u64(BalatroObservationEncoder *encoder, uint64_t value) {
    balatro_encode_u16(encoder, (uint16_t)value);
    balatro_encode_u16(encoder, (uint16_t)(value >> 16));
    balatro_encode_u16(encoder, (uint16_t)(value >> 32));
    balatro_encode_u16(encoder, (uint16_t)(value >> 48));
}

static void balatro_encode_selection(BalatroObservationEncoder *encoder,
                                      const BalatroObservedSelection *selection) {
    balatro_encode_u64(encoder, selection->allowed_hand);
    balatro_encode_u64(encoder, selection->required_hand);
    balatro_encode_u8(encoder, selection->minimum);
    balatro_encode_u8(encoder, selection->maximum);
    balatro_encode_u8(encoder, selection->valid);
}

static void balatro_encode_deck_summary(BalatroObservationEncoder *encoder,
                                        const BalatroDeckSummary *deck) {
    for (int i = 0; i < 13; ++i) balatro_encode_u16(encoder, deck->rank[i]);
    for (int i = 0; i < 4; ++i) balatro_encode_u16(encoder, deck->suit[i]);
    for (int suit = 0; suit < 4; ++suit)
        for (int rank = 0; rank < 13; ++rank)
            balatro_encode_u16(encoder, deck->rank_suit[suit][rank]);
    for (int i = 0; i < 9; ++i) balatro_encode_u16(encoder, deck->enhancement[i]);
    for (int i = 0; i < 5; ++i) balatro_encode_u16(encoder, deck->edition[i]);
    for (int i = 0; i < 5; ++i) balatro_encode_u16(encoder, deck->seal[i]);
    balatro_encode_u16(encoder, deck->face);
    balatro_encode_u16(encoder, deck->numbered);
    balatro_encode_u16(encoder, deck->ace);
    balatro_encode_u16(encoder, deck->stone);
    balatro_encode_u16(encoder, deck->wild);
    balatro_encode_u16(encoder, deck->steel);
    balatro_encode_u16(encoder, deck->gold);
    balatro_encode_u16(encoder, deck->glass);
    balatro_encode_u16(encoder, deck->enhanced);
    balatro_encode_u16(encoder, deck->unmodified);
    balatro_encode_u16(encoder, deck->total);
}

#define BALATRO_ENCODE_CARD_TABLE(encoder, table, capacity) \
    do { \
        balatro_encode_u16(&(encoder), (table).count); \
        for (int i = 0; i < (capacity); ++i) { \
            balatro_encode_u16(&(encoder), (table).center_id[i]); \
            balatro_encode_u8(&(encoder), (table).rank[i]); \
            balatro_encode_u8(&(encoder), (table).suit[i]); \
            balatro_encode_u8(&(encoder), (table).enhancement[i]); \
            balatro_encode_u8(&(encoder), (table).edition[i]); \
            balatro_encode_u8(&(encoder), (table).seal[i]); \
            balatro_encode_u8(&(encoder), (table).flags[i]); \
            balatro_encode_float(&(encoder), (table).perma_bonus[i]); \
            balatro_encode_float(&(encoder), (table).cost[i]); \
            balatro_encode_float(&(encoder), (table).sell_cost[i]); \
            for (int k = 0; k < 4; ++k) balatro_encode_u32(&(encoder), (uint32_t)(table).mutable_raw[k][i]); \
            for (int k = 0; k < 4; ++k) balatro_encode_float(&(encoder), (table).mutable_value[k][i]); \
            balatro_encode_u8(&(encoder), (table).valid[i]); \
        } \
    } while (0)

static int balatro_encode_observation(const BalatroObservation *observation, float *out) {
    BalatroObservationEncoder encoder = {.out = out, .count = 0};
    const BalatroObservationScalars *scalars = &observation->scalars;
    balatro_encode_u16(&encoder, observation->profile.playing_cards);
    balatro_encode_u16(&encoder, observation->profile.playing_variants);
    balatro_encode_u16(&encoder, observation->profile.hand);
    balatro_encode_u16(&encoder, observation->profile.jokers);
    balatro_encode_u16(&encoder, observation->profile.consumables);
    balatro_encode_u16(&encoder, observation->profile.tags);
    balatro_encode_u16(&encoder, observation->profile.shop_vouchers);
    balatro_encode_u32(&encoder, observation->encoded_bytes);
    balatro_encode_u16(&encoder, observation->required_capacity);
    balatro_encode_u8(&encoder, observation->truncation_reason);
    balatro_encode_u8(&encoder, observation->overflow_section);

    balatro_encode_u16(&encoder, scalars->deck_id);
    balatro_encode_u16(&encoder, scalars->blind_id);
    balatro_encode_u16(&encoder, scalars->next_boss_id);
    balatro_encode_u16(&encoder, scalars->next_voucher_id);
    balatro_encode_u16(&encoder, scalars->last_tarot_planet);
    const uint8_t *scalar_bytes = &scalars->stake;
    for (int i = 0; i < 36; ++i) balatro_encode_u8(&encoder, scalar_bytes[i]);
    balatro_encode_u32(&encoder, scalars->ante);
    balatro_encode_u32(&encoder, scalars->round);
    balatro_encode_u32(&encoder, scalars->actions_taken);
    balatro_encode_u32(&encoder, scalars->run_hands_played);
    balatro_encode_u16(&encoder, scalars->hands_left);
    balatro_encode_u16(&encoder, scalars->discards_left);
    balatro_encode_u16(&encoder, scalars->hands_played);
    balatro_encode_u16(&encoder, scalars->discards_used);
    balatro_encode_u16(&encoder, scalars->hand_size);
    balatro_encode_u16(&encoder, scalars->joker_slots);
    balatro_encode_u16(&encoder, scalars->consumable_slots);
    balatro_encode_u16(&encoder, scalars->skips);
    balatro_encode_u16(&encoder, scalars->pack_choices);
    balatro_encode_u16(&encoder, scalars->unused_discards);
    balatro_encode_u16(&encoder, scalars->blind_hands_mask);
    balatro_encode_u16(&encoder, scalars->tarots_used);
    balatro_encode_u16(&encoder, scalars->planet_usage_mask);
    balatro_encode_u8(&encoder, scalars->chips_number);
    balatro_encode_u8(&encoder, scalars->blind_chips_number);
    balatro_encode_u8(&encoder, scalars->last_hand_score_number);
    balatro_encode_u8(&encoder, scalars->chips_over_blind_number);
    balatro_encode_float(&encoder, scalars->dollars);
    balatro_encode_float(&encoder, scalars->chips);
    balatro_encode_float(&encoder, scalars->blind_chips);
    balatro_encode_float(&encoder, scalars->last_hand_score);
    balatro_encode_float(&encoder, scalars->chips_over_blind);
    balatro_encode_float(&encoder, scalars->reroll_cost);
    balatro_encode_float(&encoder, scalars->round_earnings);
    balatro_encode_float(&encoder, scalars->interest_cap);
    balatro_encode_float(&encoder, scalars->interest_amount);
    balatro_encode_float(&encoder, scalars->blind_reward);
    balatro_encode_float(&encoder, scalars->joker_rate);
    balatro_encode_float(&encoder, scalars->tarot_rate);
    balatro_encode_float(&encoder, scalars->planet_rate);
    balatro_encode_float(&encoder, scalars->spectral_rate);
    balatro_encode_float(&encoder, scalars->playing_card_rate);
    balatro_encode_float(&encoder, scalars->edition_rate);
    for (int i = 0; i < BALATRO_CENTER_COUNT; ++i)
        balatro_encode_u8(&encoder, scalars->redeemed_vouchers[i]);

    balatro_encode_u16(&encoder, observation->variants.count);
    for (int i = 0; i < BALATRO_OBS_MAX_PLAYING_VARIANTS; ++i) {
        balatro_encode_u8(&encoder, observation->variants.rank[i]);
        balatro_encode_u8(&encoder, observation->variants.suit[i]);
        balatro_encode_u8(&encoder, observation->variants.enhancement[i]);
        balatro_encode_u8(&encoder, observation->variants.edition[i]);
        balatro_encode_u8(&encoder, observation->variants.seal[i]);
        balatro_encode_u8(&encoder, observation->variants.flags[i]);
        balatro_encode_float(&encoder, observation->variants.perma_bonus[i]);
        balatro_encode_u16(&encoder, observation->variants.owned_count[i]);
        balatro_encode_u16(&encoder, observation->variants.draw_count[i]);
        balatro_encode_u16(&encoder, observation->variants.hand_count[i]);
        balatro_encode_u16(&encoder, observation->variants.discard_count[i]);
        balatro_encode_u8(&encoder, observation->variants.valid[i]);
    }
    balatro_encode_u16(&encoder, observation->hand.count);
    for (int i = 0; i < BALATRO_OBS_MAX_HAND; ++i) {
        balatro_encode_u16(&encoder, observation->hand.variant[i]);
        balatro_encode_u8(&encoder, observation->hand.flags[i]);
        balatro_encode_u8(&encoder, observation->hand.valid[i]);
    }
    balatro_encode_deck_summary(&encoder, &observation->owned_deck);
    balatro_encode_deck_summary(&encoder, &observation->draw_pile);
    BALATRO_ENCODE_CARD_TABLE(encoder, observation->jokers, BALATRO_OBS_MAX_JOKERS);
    BALATRO_ENCODE_CARD_TABLE(encoder, observation->consumables, BALATRO_OBS_MAX_CONSUMABLES);
    BALATRO_ENCODE_CARD_TABLE(encoder, observation->shop, BALATRO_OBS_MAX_SHOP_MAIN);
    BALATRO_ENCODE_CARD_TABLE(encoder, observation->shop_vouchers, BALATRO_OBS_MAX_SHOP_VOUCHERS);
    BALATRO_ENCODE_CARD_TABLE(encoder, observation->shop_boosters, BALATRO_OBS_MAX_SHOP_BOOSTERS);
    BALATRO_ENCODE_CARD_TABLE(encoder, observation->pack, BALATRO_OBS_MAX_PACK_CARDS);
    balatro_encode_u16(&encoder, observation->tags.count);
    for (int i = 0; i < BALATRO_OBS_MAX_TAGS; ++i) {
        balatro_encode_u8(&encoder, observation->tags.tag_id[i]);
        balatro_encode_u8(&encoder, observation->tags.orbital_hand[i]);
        balatro_encode_u8(&encoder, observation->tags.flags[i]);
        balatro_encode_u8(&encoder, observation->tags.valid[i]);
    }
    for (int i = 0; i < BALATRO_HAND_COUNT; ++i) {
        balatro_encode_u8(&encoder, observation->poker_hands.visible[i]);
        balatro_encode_u32(&encoder, observation->poker_hands.level[i]);
        balatro_encode_float(&encoder, observation->poker_hands.chips[i]);
        balatro_encode_float(&encoder, observation->poker_hands.mult[i]);
        balatro_encode_u32(&encoder, observation->poker_hands.total_plays[i]);
        balatro_encode_u32(&encoder, observation->poker_hands.round_plays[i]);
    }
    for (int i = 0; i < BALATRO_ACTION_TYPE_COUNT; ++i) {
        balatro_encode_u8(&encoder, observation->legal.action_type[i]);
        balatro_encode_u64(&encoder, observation->legal.primary[i]);
    }
    balatro_encode_selection(&encoder, &observation->legal.play);
    balatro_encode_selection(&encoder, &observation->legal.discard);
    for (int i = 0; i < BALATRO_OBS_MAX_CONSUMABLES; ++i)
        balatro_encode_selection(&encoder, &observation->legal.consumable[i]);
    for (int i = 0; i < BALATRO_OBS_MAX_SHOP_MAIN; ++i)
        balatro_encode_selection(&encoder, &observation->legal.shop[i]);
    for (int i = 0; i < BALATRO_OBS_MAX_PACK_CARDS; ++i)
        balatro_encode_selection(&encoder, &observation->legal.pack[i]);
    for (int i = 0; i < BALATRO_OBS_MAX_HAND; ++i)
        balatro_encode_u64(&encoder, observation->legal.hand_reorder_destination[i]);
    for (int i = 0; i < BALATRO_OBS_MAX_JOKERS; ++i)
        balatro_encode_u64(&encoder, observation->legal.joker_reorder_destination[i]);
    return encoder.count;
}
#undef BALATRO_ENCODE_CARD_TABLE

static void balatro_mask_range(unsigned char *mask, int offset, int count) {
    for (int i = 0; i < count; ++i) mask[offset + i] = 1;
}

static void balatro_puffer_mask(const BalatroObservation *observation, unsigned char *mask) {
    memset(mask, 0, ACTION_MASK_SIZE);
    int type_offset = 0;
    int primary_offset = type_offset + 23;
    int count_offset = primary_offset + 64;
    int selection_offset = count_offset + 6;
    int reorder_offset = selection_offset + 5 * 64;
    int any_type = 0;
    int any_selection = 0;
    int any_reorder = 0;

    for (int type = 0; type < BALATRO_ACTION_TYPE_COUNT; ++type) {
        if (!observation->legal.action_type[type]) continue;
        mask[type_offset + type] = 1;
        any_type = 1;
        uint64_t primary = observation->legal.primary[type];
        for (int slot = 0; slot < 64; ++slot)
            if (primary & (UINT64_C(1) << slot)) mask[primary_offset + slot] = 1;
    }

    const BalatroObservedSelection *families[BALATRO_OBS_MAX_CONSUMABLES +
        BALATRO_OBS_MAX_SHOP_MAIN + BALATRO_OBS_MAX_PACK_CARDS + 2];
    int family_count = 0;
    families[family_count++] = &observation->legal.play;
    families[family_count++] = &observation->legal.discard;
    for (int i = 0; i < BALATRO_OBS_MAX_CONSUMABLES; ++i)
        families[family_count++] = &observation->legal.consumable[i];
    for (int i = 0; i < BALATRO_OBS_MAX_SHOP_MAIN; ++i)
        families[family_count++] = &observation->legal.shop[i];
    for (int i = 0; i < BALATRO_OBS_MAX_PACK_CARDS; ++i)
        families[family_count++] = &observation->legal.pack[i];
    for (int i = 0; i < family_count; ++i) {
        const BalatroObservedSelection *selection = families[i];
        if (!selection->valid) continue;
        any_selection = 1;
        for (int card = 0; card < 64; ++card)
            if (selection->allowed_hand & (UINT64_C(1) << card))
                for (int position = 0; position < 5; ++position)
                    mask[selection_offset + position * 64 + card] = 1;
    }
    for (int i = 0; i < BALATRO_OBS_MAX_HAND; ++i)
        if (observation->legal.hand_reorder_destination[i]) {
            any_reorder = 1;
            for (int destination = 0; destination < 64; ++destination)
                if (observation->legal.hand_reorder_destination[i] &
                    (UINT64_C(1) << destination))
                    mask[reorder_offset + destination] = 1;
        }
    for (int i = 0; i < BALATRO_OBS_MAX_JOKERS; ++i)
        if (observation->legal.joker_reorder_destination[i]) {
            any_reorder = 1;
            for (int destination = 0; destination < 64; ++destination)
                if (observation->legal.joker_reorder_destination[i] &
                    (UINT64_C(1) << destination))
                    mask[reorder_offset + destination] = 1;
        }

    /* Every categorical head must have at least one finite option. The
       structured validator remains authoritative for cross-head combinations. */
    if (!any_type) mask[type_offset] = 1;
    int any_primary = 0;
    for (int i = 0; i < 64; ++i) any_primary |= mask[primary_offset + i] != 0;
    if (!any_primary) mask[primary_offset] = 1;
    if (any_selection) balatro_mask_range(mask, count_offset, 6);
    else mask[count_offset] = 1;
    if (!any_selection)
        for (int position = 0; position < 5; ++position) mask[selection_offset + position * 64] = 1;
    if (!any_reorder) mask[reorder_offset] = 1;
}

static int balatro_puffer_observe(Env *env) {
    BalatroObservation observation;
    int error = balatro_observe(&env->state, &observation);
    float *out = (float *)env->agents[0].observations;
    memset(out, 0, OBS_SIZE * sizeof(float));
    if (error == BALATRO_OK) {
        int encoded = balatro_encode_observation(&observation, out);
        if (encoded > OBS_SIZE) error = BALATRO_ERR_OBSERVATION_CAPACITY;
        if (env->agents[0].action_mask)
            balatro_puffer_mask(&observation, env->agents[0].action_mask);
    }
    env->legal_count = 0;
    return error;
}

void puf_init(Env *env, Dict *kwargs) {
    env->num_agents = 1;
    env->tag = 0;
    env->boundary_reached = 0;
    env->episode_reward = 0.0f;
    env->agents[0].policy = 0;
    balatro_default_config(&env->config);
    /* Training defaults to the progress potential, while the same wrapper
       can run the ABI's sparse win-only objective for an apples-to-apples
       baseline.  Optional env.* overrides are useful for curriculum runs. */
    DictItem *shaped = dict_find(kwargs, "shaped_reward");
    DictItem *win_ante = dict_find(kwargs, "win_ante");
    env->config.shaped_reward = shaped ? (shaped->value != 0.0) : 1;
    if (win_ante) {
        int value = (int)win_ante->value;
        if (value < 1) value = 1;
        if (value > UINT8_MAX) value = UINT8_MAX;
        env->config.win_ante = (uint8_t)value;
    }
}

void puf_reset(Env *env) {
    uint64_t seed = ((uint64_t)rand_r(&env->rng) << 32) | rand_r(&env->rng);
    balatro_init(&env->state, &env->config, seed);
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    env->boundary_reached = 0;
    env->episode_reward = 0.0f;
    balatro_puffer_observe(env);
}

void puf_step(Env *env) {
    BalatroPolicyAction policy = {0};
    policy.type = (uint8_t)env->agents[0].actions[0];
    policy.primary = (uint16_t)env->agents[0].actions[1];
    policy.selection_count = (uint8_t)env->agents[0].actions[2];
    if (policy.selection_count > BALATRO_MAX_SELECTION)
        policy.selection_count = BALATRO_MAX_SELECTION;
    for (int i = 0; i < BALATRO_MAX_SELECTION; ++i)
        policy.selection[i] = (uint16_t)env->agents[0].actions[3 + i];
    policy.reorder_destination = (uint16_t)env->agents[0].actions[8];
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    BalatroStepResult result;
    BalatroObservation observation;
    int error = balatro_step_observe(&env->state, &policy, &result, &observation);
    if (error != BALATRO_OK) {
        env->log.invalid_actions += 1.0f;
        env->agents[0].rewards[0] = -1.0f;
        balatro_puffer_observe(env);
        return;
    }
    env->agents[0].rewards[0] = result.reward;
    env->episode_reward += result.reward;
    if (result.terminal) {
        env->boundary_reached = 1;
        env->agents[0].terminals[0] = 1.0f;
        env->log.score += result.won;
        env->log.wins += result.won;
        env->log.ante += result.ante;
        env->log.n += 1.0f;
		env->log.reward += env->episode_reward;
        puf_reset(env);
        env->agents[0].terminals[0] = 1.0f;
    } else {
        balatro_puffer_observe(env);
    }
}

void puf_render(Env *env) { (void)env; }
void puf_close(Env *env) { (void)env; }

void puf_log(Log *log, Dict *out) {
    dict_set(out, "score", log->score);
    dict_set(out, "wins", log->wins);
    dict_set(out, "ante", log->ante);
    dict_set(out, "invalid_actions", log->invalid_actions);
	dict_set(out, "reward", log->reward);
}

#endif
