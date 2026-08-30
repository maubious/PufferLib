#ifndef PUFFER_POLICY_H
#define PUFFER_POLICY_H

#include "balatro_core.h"

#define POLICY_PRIMARY_COUNT 64
#define POLICY_PRIMARY_BYTES 8
#define POLICY_SELECTION_ENTRIES \
    (2 + OBS_MAX_CONSUMABLES + OBS_MAX_PACK_CARDS + OBS_MAX_SHOP_MAIN)
#define POLICY_SELECTION_BYTES 18
#define POLICY_PRIMARY_OFFSET ACTION_TYPE_COUNT
#define POLICY_SELECTION_OFFSET \
    (POLICY_PRIMARY_OFFSET + POLICY_PRIMARY_BYTES * ACTION_TYPE_COUNT)
#define POLICY_SELECTION_SIZE \
    (POLICY_SELECTION_OFFSET + POLICY_SELECTION_ENTRIES * POLICY_SELECTION_BYTES)
/* Per-option card attributes for the AR selection heads: one byte per hand
   slot, (suit << 4) | rank, written by the env from the live hand. Lets the
   decoder condition selection scores on the set's suit/rank structure
   (flush / straight / pair completion) instead of only the last card. */
#define POLICY_CARD_ATTR_OFFSET POLICY_SELECTION_SIZE
#define POLICY_CARD_ATTR_BYTES 64
#define POLICY_CARD_ATTR_SUIT_SHIFT 4
#define POLICY_CARD_ATTR_RANK_MASK 0x0F
#define POLICY_MASK_SIZE \
    (POLICY_CARD_ATTR_OFFSET + POLICY_CARD_ATTR_BYTES)

/* Compact learned dependencies layered over the shared state-conditioned
   logits. This keeps decoding cheap while making later distributions depend
   on the sampled prefix rather than only on its legality mask. */
#define AR_TYPE_PRIMARY_OFFSET 0
#define AR_TYPE_PRIMARY_SIZE (ACTION_TYPE_COUNT * 64)
#define AR_TYPE_COUNT_OFFSET \
    (AR_TYPE_PRIMARY_OFFSET + AR_TYPE_PRIMARY_SIZE)
#define AR_TYPE_COUNT_SIZE (ACTION_TYPE_COUNT * 6)
#define AR_PRIMARY_COUNT_OFFSET \
    (AR_TYPE_COUNT_OFFSET + AR_TYPE_COUNT_SIZE)
#define AR_PRIMARY_COUNT_SIZE (64 * 6)
#define AR_TYPE_CARD_OFFSET \
    (AR_PRIMARY_COUNT_OFFSET + AR_PRIMARY_COUNT_SIZE)
#define AR_TYPE_CARD_SIZE (ACTION_TYPE_COUNT * 5 * 64)
#define AR_PRIMARY_CARD_OFFSET \
    (AR_TYPE_CARD_OFFSET + AR_TYPE_CARD_SIZE)
#define AR_PRIMARY_CARD_SIZE (64 * 5 * 64)
#define AR_COUNT_CARD_OFFSET \
    (AR_PRIMARY_CARD_OFFSET + AR_PRIMARY_CARD_SIZE)
#define AR_COUNT_CARD_SIZE (6 * 5 * 64)
#define AR_PREVIOUS_CARD_OFFSET \
    (AR_COUNT_CARD_OFFSET + AR_COUNT_CARD_SIZE)
#define AR_PREVIOUS_CARD_SIZE (64 * 4 * 64)
/* Set-synergy tables: context = summary of the already-selected cards relative
   to the option, per position (0..4) and option (64). 5 contexts each:
   suit-match count 0..4 (flush), rank-match count 0..4 (pair/full-house),
   run-length 1..5 through the option's rank (straight, ace dual). */
#define AR_SUIT_CARD_OFFSET \
    (AR_PREVIOUS_CARD_OFFSET + AR_PREVIOUS_CARD_SIZE)
#define AR_SUIT_CARD_SIZE (5 * 5 * 64)
#define AR_RANK_CARD_OFFSET \
    (AR_SUIT_CARD_OFFSET + AR_SUIT_CARD_SIZE)
#define AR_RANK_CARD_SIZE (5 * 5 * 64)
#define AR_RUN_CARD_OFFSET \
    (AR_RANK_CARD_OFFSET + AR_RANK_CARD_SIZE)
#define AR_RUN_CARD_SIZE (5 * 5 * 64)
#define AR_CONDITION_SIZE \
    (AR_RUN_CARD_OFFSET + AR_RUN_CARD_SIZE)
/* Standalone CPU eval (puffercpu.h) aliases the trainer's condition size. */
#define BALATRO_AR_CONDITION_SIZE AR_CONDITION_SIZE

#if defined(__CUDACC__) || defined(__HIPCC__)
#define POLICY_INLINE __host__ __device__ __forceinline__
#else
#define POLICY_INLINE static inline
#endif

POLICY_INLINE int policy_selection_entry(int type, int primary) {
    if (type == ACTION_PLAY_HAND) return primary == 0 ? 0 : -1;
    if (type == ACTION_DISCARD) return primary == 0 ? 1 : -1;
    if (type == ACTION_USE_CONSUMABLE)
        return primary >= 0 && primary < OBS_MAX_CONSUMABLES ? 2 + primary : -1;
    if (type == ACTION_PICK_PACK_CARD)
        return primary >= 0 && primary < OBS_MAX_PACK_CARDS
            ? 2 + OBS_MAX_CONSUMABLES + primary : -1;
    if (type == ACTION_BUY_AND_USE)
        return primary >= 0 && primary < OBS_MAX_SHOP_MAIN
            ? 2 + OBS_MAX_CONSUMABLES + OBS_MAX_PACK_CARDS + primary : -1;
    return -1;
}

#undef POLICY_INLINE
#endif
