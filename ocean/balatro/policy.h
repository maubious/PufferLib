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
#define POLICY_ORDER_META_OFFSET POLICY_SELECTION_SIZE
#define POLICY_ORDER_META_BYTES 4
#define POLICY_ORDER_HAND_COUNT_OFFSET POLICY_ORDER_META_OFFSET
#define POLICY_ORDER_JOKER_COUNT_OFFSET (POLICY_ORDER_META_OFFSET + 1)
#define POLICY_ORDER_ENABLED_OFFSET (POLICY_ORDER_META_OFFSET + 2)
#define POLICY_MASK_SIZE \
    (POLICY_ORDER_META_OFFSET + POLICY_ORDER_META_BYTES)

/* Cached 32-channel entities and a 64-channel causal decision controller. */
#define AR_EMBED_DIM 32
#define DECODER_STATE 64
#define DECODER_STEPS (8 + OBS_MAX_HAND + OBS_MAX_JOKERS)
#define DECODER_CATEGORIES (ACTION_TYPE_COUNT + 6)
#define CATEGORY_OFFSET 0
#define ROLE_OFFSET (CATEGORY_OFFSET + DECODER_CATEGORIES * AR_EMBED_DIM)
#define AR_CONDITION_SIZE (ROLE_OFFSET + DECODER_STEPS * AR_EMBED_DIM)
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
