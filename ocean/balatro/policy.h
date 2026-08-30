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

/* Factored low-rank prefix embedding parameter layout (Redesign A).
   Replaces combinatorial sparse tables with continuous prefix embeddings
   and head projections (AR_EMBED_DIM=16). */
#define AR_EMBED_DIM 16

#define AR_E_TYPE_OFFSET    0
#define AR_E_TYPE_SIZE      (ACTION_TYPE_COUNT * AR_EMBED_DIM)   /* 23 * 16 = 368 */
#define AR_E_PRIMARY_OFFSET (AR_E_TYPE_OFFSET + AR_E_TYPE_SIZE)
#define AR_E_PRIMARY_SIZE   (64 * AR_EMBED_DIM)                  /* 64 * 16 = 1024 */
#define AR_E_COUNT_OFFSET   (AR_E_PRIMARY_OFFSET + AR_E_PRIMARY_SIZE)
#define AR_E_COUNT_SIZE     (6 * AR_EMBED_DIM)                   /* 6 * 16 = 96 */
#define AR_E_CARD_OFFSET    (AR_E_COUNT_OFFSET + AR_E_COUNT_SIZE)
#define AR_E_CARD_SIZE      (64 * AR_EMBED_DIM)                  /* 64 * 16 = 1024 */
#define AR_E_POS_OFFSET     (AR_E_CARD_OFFSET + AR_E_CARD_SIZE)
#define AR_E_POS_SIZE       (5 * AR_EMBED_DIM)                   /* 5 * 16 = 80 */
#define AR_E_SUIT_OFFSET    (AR_E_POS_OFFSET + AR_E_POS_SIZE)
#define AR_E_SUIT_SIZE      (5 * AR_EMBED_DIM)                   /* 5 * 16 = 80 */
#define AR_E_RANK_OFFSET    (AR_E_SUIT_OFFSET + AR_E_SUIT_SIZE)
#define AR_E_RANK_SIZE      (5 * AR_EMBED_DIM)                   /* 5 * 16 = 80 */
#define AR_E_RUN_OFFSET     (AR_E_RANK_OFFSET + AR_E_RANK_SIZE)
#define AR_E_RUN_SIZE       (5 * AR_EMBED_DIM)                   /* 5 * 16 = 80 */

#define AR_W_PRIMARY_OFFSET (AR_E_RUN_OFFSET + AR_E_RUN_SIZE)
#define AR_W_PRIMARY_SIZE   (64 * AR_EMBED_DIM)                  /* 64 * 16 = 1024 */
#define AR_W_COUNT_OFFSET   (AR_W_PRIMARY_OFFSET + AR_W_PRIMARY_SIZE)
#define AR_W_COUNT_SIZE     (6 * AR_EMBED_DIM)                   /* 6 * 16 = 96 */
#define AR_W_CARD_OFFSET    (AR_W_COUNT_OFFSET + AR_W_COUNT_SIZE)
#define AR_W_CARD_SIZE      (64 * AR_EMBED_DIM)                  /* 64 * 16 = 1024 */

#define AR_CONDITION_SIZE   (AR_W_CARD_OFFSET + AR_W_CARD_SIZE)
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
