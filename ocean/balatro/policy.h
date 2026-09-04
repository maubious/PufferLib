#ifndef PUFFER_POLICY_H
#define PUFFER_POLICY_H

#include "balatro_core.h"

#define POLICY_PRIMARY_COUNT 64
#define POLICY_PRIMARY_HEADS 12
#define POLICY_PRIMARY_HEAD_SIZE (POLICY_PRIMARY_HEADS * POLICY_PRIMARY_COUNT)
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

/* Prefix context parameters. Candidate and selected-item content comes from
   the encoder keys; these tables encode only action grammar and set shape. */
#define AR_EMBED_DIM 16

#define AR_E_TYPE_OFFSET    0
#define AR_E_TYPE_SIZE      (ACTION_TYPE_COUNT * AR_EMBED_DIM)
#define AR_E_COUNT_OFFSET   (AR_E_TYPE_OFFSET + AR_E_TYPE_SIZE)
#define AR_E_COUNT_SIZE     (6 * AR_EMBED_DIM)
#define AR_E_POS_OFFSET     (AR_E_COUNT_OFFSET + AR_E_COUNT_SIZE)
#define AR_E_POS_SIZE       (5 * AR_EMBED_DIM)
#define AR_W_COUNT_OFFSET   (AR_E_POS_OFFSET + AR_E_POS_SIZE)
#define AR_W_COUNT_SIZE     (6 * AR_EMBED_DIM)
#define AR_GATE_OFFSET      (AR_W_COUNT_OFFSET + AR_W_COUNT_SIZE)
#define AR_GATE_SIZE        (ACTION_TYPE_COUNT * AR_EMBED_DIM)
#define AR_CONDITION_SIZE   (AR_GATE_OFFSET + AR_GATE_SIZE)
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

/* Primary logits are laid out as one 64-way slice per action family.  The
   action stored in the environment remains the local option index; this
   helper is the single mapping between action type and decoder slice. */
POLICY_INLINE int policy_primary_head_offset(int type) {
    switch (type) {
    case ACTION_BUY_CARD: return 0;
    case ACTION_SELL_JOKER: return 1;
    case ACTION_SELL_CONSUMABLE: return 2;
    case ACTION_USE_CONSUMABLE: return 3;
    case ACTION_REDEEM_VOUCHER: return 4;
    case ACTION_OPEN_BOOSTER: return 5;
    case ACTION_PICK_PACK_CARD: return 6;
    case ACTION_SWAP_JOKERS_LEFT: return 7;
    case ACTION_SWAP_JOKERS_RIGHT: return 8;
    case ACTION_SWAP_HAND_LEFT: return 9;
    case ACTION_SWAP_HAND_RIGHT: return 10;
    case ACTION_BUY_AND_USE: return 11;
    default: return -1;
    }
}

#undef POLICY_INLINE
#endif
