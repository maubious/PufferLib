// Balatro GPU encoder: decode packed Observation records, gather shared
// center embeddings, concatenate semantic features, and project once.

static constexpr int BA_CENTER_EMBED = 8;
static constexpr int BA_BLIND_EMBED = 4;
static constexpr int BA_TAG_EMBED = 4;
static constexpr int BA_VARIANT_FEATURES = 11;
static constexpr int BA_HAND_FEATURES = 2;
static constexpr int BA_CARD_FEATURES = 21;
static constexpr int BA_POKER_FEATURES = 6;
static constexpr int BA_NUM_CARDS = OBS_MAX_JOKERS + OBS_MAX_CONSUMABLES
    + OBS_MAX_SHOP_MAIN + OBS_MAX_SHOP_VOUCHERS
    + OBS_MAX_SHOP_BOOSTERS + OBS_MAX_PACK_CARDS;
static constexpr int BA_CENTER_OCCURRENCES = BA_NUM_CARDS + 3;
static constexpr int BA_EMBED_FEATURES = BA_CENTER_OCCURRENCES * BA_CENTER_EMBED
    + 2 * BA_BLIND_EMBED + OBS_MAX_TAGS * BA_TAG_EMBED;

static constexpr int BA_GLOBAL_OFFSET = 0;
static constexpr int BA_GLOBAL_ID_FEATURES = 3 * BA_CENTER_EMBED + 2 * BA_BLIND_EMBED;
static constexpr int BA_GLOBAL_U8_START = offsetof(ObservationGlobals, stake);
static constexpr int BA_GLOBAL_U8_COUNT = offsetof(ObservationGlobals, ante) - BA_GLOBAL_U8_START;
static constexpr int BA_GLOBAL_FEATURES = BA_GLOBAL_ID_FEATURES + 6 + 2 * HAND_COUNT
    + (BA_GLOBAL_U8_COUNT - 3) + 2 + 13 + 16 + CENTER_COUNT;
static constexpr int BA_VARIANT_OFFSET = BA_GLOBAL_OFFSET + BA_GLOBAL_FEATURES;
static constexpr int BA_VARIANT_SECTION = 1
    + OBS_MAX_PLAYING_VARIANTS * BA_VARIANT_FEATURES;
static constexpr int BA_HAND_OFFSET = BA_VARIANT_OFFSET + BA_VARIANT_SECTION;
static constexpr int BA_HAND_SECTION = 1 + OBS_MAX_HAND * BA_HAND_FEATURES;
static constexpr int BA_DECK_OFFSET = BA_HAND_OFFSET + BA_HAND_SECTION;
static constexpr int BA_DECK_FIELDS = sizeof(ObservationDeckSummary) / sizeof(uint16_t);
static constexpr int BA_DECK_SECTION = 2 * BA_DECK_FIELDS;
static constexpr int BA_CARD_OFFSET = BA_DECK_OFFSET + BA_DECK_SECTION;
static constexpr int BA_CARD_SECTION = 6 + BA_NUM_CARDS * BA_CARD_FEATURES;
static constexpr int BA_TAG_OFFSET = BA_CARD_OFFSET + BA_CARD_SECTION;
static constexpr int BA_TAG_FEATURES = BA_TAG_EMBED + 2;
static constexpr int BA_TAG_SECTION = 1 + OBS_MAX_TAGS * BA_TAG_FEATURES;
static constexpr int BA_POKER_OFFSET = BA_TAG_OFFSET + BA_TAG_SECTION;
static constexpr int BA_FEATURES = BA_POKER_OFFSET + HAND_COUNT * BA_POKER_FEATURES;

static_assert(sizeof(Observation) == 8120,
    "Balatro encoder must be updated for the Observation layout");
static_assert(sizeof(ObservationCard) == 22 && sizeof(ObservationVariant) == 16,
    "Balatro encoder card layout mismatch");
static_assert(BA_NUM_CARDS == 139 && BA_FEATURES == 6583,
    "Balatro encoder feature layout mismatch");

struct BalatroEncoderWeights {
    Prec center_embed, blind_embed, tag_embed, proj_w;
    int obs_size, hidden;
};

struct BalatroEncoderActivations {
    Prec features, out, embed_proj, embed_grad;
    Int center_ids, blind_ids, tag_ids;
    Prec center_embed_wgrad, blind_embed_wgrad, tag_embed_wgrad, proj_wgrad;
    Float center_embed_wgrad_f, blind_embed_wgrad_f, tag_embed_wgrad_f;
};

__device__ __forceinline__ int ba_byte(
        const unsigned char* obs, int64_t base, int offset) {
    return (int)obs[base + offset];
}

__device__ __forceinline__ int ba_u16(
        const unsigned char* obs, int64_t base, int offset) {
    return ba_byte(obs, base, offset)
        | (ba_byte(obs, base, offset + 1) << 8);
}

__device__ __forceinline__ int ba_i16(
        const unsigned char* obs, int64_t base, int offset) {
    int value = ba_u16(obs, base, offset);
    return value >= 32768 ? value - 65536 : value;
}

__device__ __forceinline__ uint32_t ba_u32(
        const unsigned char* obs, int64_t base, int offset) {
    return (uint32_t)ba_byte(obs, base, offset)
        | ((uint32_t)ba_byte(obs, base, offset + 1) << 8)
        | ((uint32_t)ba_byte(obs, base, offset + 2) << 16)
        | ((uint32_t)ba_byte(obs, base, offset + 3) << 24);
}

__device__ __forceinline__ int ba_card_zone(
        int local, int* input_base, int* capacity, int* card_prefix) {
    int cursor = 0;
#define BA_ZONE(member, cap, prefix) \
    if (local < cursor + 1 + (cap) * BA_CARD_FEATURES) { \
        *input_base = offsetof(Observation, member); \
        *capacity = (cap); \
        *card_prefix = (prefix); \
        return local - cursor; \
    } \
    cursor += 1 + (cap) * BA_CARD_FEATURES
    BA_ZONE(jokers, OBS_MAX_JOKERS, 0);
    BA_ZONE(consumables, OBS_MAX_CONSUMABLES, OBS_MAX_JOKERS);
    BA_ZONE(shop, OBS_MAX_SHOP_MAIN, OBS_MAX_JOKERS + OBS_MAX_CONSUMABLES);
    BA_ZONE(shop_vouchers, OBS_MAX_SHOP_VOUCHERS,
        OBS_MAX_JOKERS + OBS_MAX_CONSUMABLES + OBS_MAX_SHOP_MAIN);
    BA_ZONE(shop_boosters, OBS_MAX_SHOP_BOOSTERS,
        OBS_MAX_JOKERS + OBS_MAX_CONSUMABLES + OBS_MAX_SHOP_MAIN
        + OBS_MAX_SHOP_VOUCHERS);
    BA_ZONE(pack, OBS_MAX_PACK_CARDS,
        OBS_MAX_JOKERS + OBS_MAX_CONSUMABLES + OBS_MAX_SHOP_MAIN
        + OBS_MAX_SHOP_VOUCHERS + OBS_MAX_SHOP_BOOSTERS);
#undef BA_ZONE
    return -1;
}

__device__ __forceinline__ int ba_card_feature_offset(int card) {
    int prefix = 0;
    // One zone-count feature appears before each zone. Account for it
    // explicitly because card indices are flattened across zones.
    if (card < OBS_MAX_JOKERS)
        return BA_CARD_OFFSET + 1 + card * BA_CARD_FEATURES;
    prefix = OBS_MAX_JOKERS;
    if (card < prefix + OBS_MAX_CONSUMABLES)
        return BA_CARD_OFFSET + 2 + card * BA_CARD_FEATURES;
    prefix += OBS_MAX_CONSUMABLES;
    if (card < prefix + OBS_MAX_SHOP_MAIN)
        return BA_CARD_OFFSET + 3 + card * BA_CARD_FEATURES;
    prefix += OBS_MAX_SHOP_MAIN;
    if (card < prefix + OBS_MAX_SHOP_VOUCHERS)
        return BA_CARD_OFFSET + 4 + card * BA_CARD_FEATURES;
    prefix += OBS_MAX_SHOP_VOUCHERS;
    if (card < prefix + OBS_MAX_SHOP_BOOSTERS)
        return BA_CARD_OFFSET + 5 + card * BA_CARD_FEATURES;
    return BA_CARD_OFFSET + 6 + card * BA_CARD_FEATURES;
}

__global__ void ba_gather_embed_projection(
        precision_t* __restrict__ compact,
        const precision_t* __restrict__ projection, int hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BA_EMBED_FEATURES * hidden) return;
    int e = idx / hidden;
    int h = idx % hidden;
    int semantic;
    int center_features = BA_CENTER_OCCURRENCES * BA_CENTER_EMBED;
    if (e < center_features) {
        int occurrence = e / BA_CENTER_EMBED;
        int d = e % BA_CENTER_EMBED;
        if (occurrence == 0) semantic = d;
        else if (occurrence == 1) semantic = 16 + d;
        else if (occurrence == 2) semantic = 24 + d;
        else semantic = ba_card_feature_offset(occurrence - 3) + d;
    } else {
        int local = e - center_features;
        if (local < 2 * BA_BLIND_EMBED) {
            int occurrence = local / BA_BLIND_EMBED;
            semantic = (occurrence == 0 ? 8 : 12) + local % BA_BLIND_EMBED;
        } else {
            local -= 2 * BA_BLIND_EMBED;
            int occurrence = local / BA_TAG_EMBED;
            semantic = BA_TAG_OFFSET + 1 + occurrence * BA_TAG_FEATURES
                + local % BA_TAG_EMBED;
        }
    }
    compact[idx] = projection[(int64_t)h * BA_FEATURES + semantic];
}

__global__ void ba_embed_backward_kernel(
        float* __restrict__ center_wgrad,
        float* __restrict__ blind_wgrad,
        float* __restrict__ tag_wgrad,
        const precision_t* __restrict__ compact_grad,
        const int* __restrict__ center_ids,
        const int* __restrict__ blind_ids,
        const int* __restrict__ tag_ids,
        int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * BA_EMBED_FEATURES) return;
    int b = idx / BA_EMBED_FEATURES;
    int local = idx - b * BA_EMBED_FEATURES;
    int center_features = BA_CENTER_OCCURRENCES * BA_CENTER_EMBED;
    float value = to_float(compact_grad[idx]);
    if (local < center_features) {
        int occurrence = local / BA_CENTER_EMBED;
        int d = local % BA_CENTER_EMBED;
        int id = center_ids[b * BA_CENTER_OCCURRENCES + occurrence];
        if ((unsigned)id < CENTER_COUNT) {
            atomicAdd(&center_wgrad[id * BA_CENTER_EMBED + d], value);
        }
    } else if ((local -= center_features) < 2 * BA_BLIND_EMBED) {
        int occurrence = local / BA_BLIND_EMBED;
        int d = local % BA_BLIND_EMBED;
        int id = blind_ids[b * 2 + occurrence];
        if ((unsigned)id < BLIND_COUNT) {
            atomicAdd(&blind_wgrad[id * BA_BLIND_EMBED + d], value);
        }
    } else {
        local -= 2 * BA_BLIND_EMBED;
        int occurrence = local / BA_TAG_EMBED;
        int d = local % BA_TAG_EMBED;
        int id = tag_ids[b * OBS_MAX_TAGS + occurrence];
        if ((unsigned)id < TAG_COUNT) {
            atomicAdd(&tag_wgrad[id * BA_TAG_EMBED + d], value);
        }
    }
}

__global__ void ba_extract_kernel(
        precision_t* __restrict__ out,
        int* __restrict__ center_ids,
        int* __restrict__ blind_ids,
        int* __restrict__ tag_ids,
        const unsigned char* __restrict__ obs,
        const precision_t* __restrict__ center_embed,
        const precision_t* __restrict__ blind_embed,
        const precision_t* __restrict__ tag_embed,
        int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * BA_FEATURES) return;
    int b = idx / BA_FEATURES;
    int feature = idx - b * BA_FEATURES;
    int64_t in = (int64_t)b * obs_size;
    float value = 0.0f;

    if (feature < BA_VARIANT_OFFSET) {
        int local = feature - BA_GLOBAL_OFFSET;
        int base = offsetof(Observation, globals);
        int deck = ba_u16(obs, in, base + offsetof(ObservationGlobals, deck_id));
        int blind = ba_u16(obs, in, base + offsetof(ObservationGlobals, blind_id));
        int boss = ba_u16(obs, in, base + offsetof(ObservationGlobals, next_boss_id));
        int voucher = ba_u16(obs, in, base + offsetof(ObservationGlobals, next_voucher_id));
        int tarot = ba_u16(obs, in, base + offsetof(ObservationGlobals, last_tarot_planet));
        if (local < BA_GLOBAL_ID_FEATURES) {
            int id = -1, d = 0;
            const precision_t* table = center_embed;
            if (local < 8) { id = deck; d = local; }
            else if (local < 12) { id = blind; d = local - 8; table = blind_embed; }
            else if (local < 16) { id = boss; d = local - 12; table = blind_embed; }
            else if (local < 24) { id = voucher; d = local - 16; }
            else { id = tarot; d = local - 24; }
            int limit = table == blind_embed ? BLIND_COUNT : CENTER_COUNT;
            int stored_id = id > 0 && id < limit ? id : -1;
            if (stored_id >= 0) value = to_float(table[stored_id * (table == blind_embed ? 4 : 8) + d]);
            if (d == 0) {
                if (local == 0 && center_ids) center_ids[b * (BA_NUM_CARDS + 3)] = stored_id;
                else if (local == 8 && blind_ids) blind_ids[b * 2] = stored_id;
                else if (local == 12 && blind_ids) blind_ids[b * 2 + 1] = stored_id;
                else if (local == 16 && center_ids) center_ids[b * (BA_NUM_CARDS + 3) + 1] = stored_id;
                else if (local == 24 && center_ids) center_ids[b * (BA_NUM_CARDS + 3) + 2] = stored_id;
            }
        } else {
            local -= BA_GLOBAL_ID_FEATURES;
            int phase = ba_byte(obs, in, base + offsetof(ObservationGlobals, phase));
            int most = ba_byte(obs, in, base + offsetof(ObservationGlobals, most_played_hand));
            int last = ba_byte(obs, in, base + offsetof(ObservationGlobals, last_hand_type));
            if (local < 6) value = local == phase;
            else if ((local -= 6) < HAND_COUNT) value = local == most;
            else if ((local -= HAND_COUNT) < HAND_COUNT) value = local == last;
            else {
                int scalar_u8 = BA_GLOBAL_U8_COUNT - 3;
                if (local < scalar_u8) {
                    int wanted = local;
                    int selected = 0;
                    for (int i = 0; i < BA_GLOBAL_U8_COUNT; ++i) {
                        int off = BA_GLOBAL_U8_START + i;
                        if (off == offsetof(ObservationGlobals, phase) ||
                            off == offsetof(ObservationGlobals, most_played_hand) ||
                            off == offsetof(ObservationGlobals, last_hand_type)) continue;
                        if (selected++ == wanted) {
                            int raw = ba_byte(obs, in, base + off);
                            if (off == offsetof(ObservationGlobals, blind_disabled) ||
                                off == offsetof(ObservationGlobals, hand_sort_suit) ||
                                off == offsetof(ObservationGlobals, boss_rerolled) ||
                                off == offsetof(ObservationGlobals, double_tag) ||
                                off == offsetof(ObservationGlobals, tag_voucher_pending) ||
                                off == offsetof(ObservationGlobals, tag_coupon_pending) ||
                                off == offsetof(ObservationGlobals, tag_coupon_active) ||
                                off == offsetof(ObservationGlobals, tag_investment_pending) ||
                                off == offsetof(ObservationGlobals, tag_d_six_pending) ||
                                off == offsetof(ObservationGlobals, tag_d_six_active) ||
                                off == offsetof(ObservationGlobals, gros_michel_extinct))
                                value = (float)(raw != 0);
                            else if (off == offsetof(ObservationGlobals, stake))
                                value = (float)raw / 8.0f;
                            else if (off == offsetof(ObservationGlobals, blind_on_deck))
                                value = (float)raw / 2.0f;
                            else if (off == offsetof(ObservationGlobals, blind_skipped_mask))
                                value = (float)raw / 7.0f;
                            else if (off == offsetof(ObservationGlobals, discount_percent))
                                value = (float)raw / 100.0f;
                            else if (off == offsetof(ObservationGlobals, active_tag))
                                value = (float)raw / TAG_COUNT;
                            else if (off == offsetof(ObservationGlobals, tag_force_rarity) ||
                                     off == offsetof(ObservationGlobals, tag_force_edition))
                                value = (float)raw / 4.0f;
                            else
                                value = (float)raw / 16.0f;
                            break;
                        }
                    }
                } else if ((local -= scalar_u8) < 2) {
                    int off = local == 0 ? offsetof(ObservationGlobals, ante)
                        : offsetof(ObservationGlobals, run_hands_played);
                    value = (float)ba_u32(obs, in, base + off) / (local == 0 ? 16.0f : 256.0f);
                } else if ((local -= 2) < 13) {
                    value = (float)ba_u16(obs, in,
                        base + offsetof(ObservationGlobals, hands_left) + 2 * local) / 64.0f;
                } else if ((local -= 13) < 16) {
                    value = (float)ba_i16(obs, in,
                        base + offsetof(ObservationGlobals, dollars_q8_8) + 2 * local) / (256.0f * 16.0f);
                } else {
                    local -= 16;
                    int byte = ba_byte(obs, in,
                        base + offsetof(ObservationGlobals, redeemed_vouchers) + local / 8);
                    value = (float)((byte >> (local & 7)) & 1);
                }
            }
        }
    } else if (feature < BA_HAND_OFFSET) {
        int local = feature - BA_VARIANT_OFFSET;
        int base = offsetof(Observation, variants);
        if (local == 0) {
            value = (float)ba_u16(obs, in, base) / OBS_MAX_PLAYING_VARIANTS;
        } else {
            int rec = (local - 1) / BA_VARIANT_FEATURES;
            int f = (local - 1) % BA_VARIANT_FEATURES;
            int record = base + offsetof(ObservationVariants, values)
                + rec * sizeof(ObservationVariant);
            if (f < 5) {
                static constexpr float scales[5] = {
                    1.0f / 14.0f, 1.0f / 3.0f, 1.0f / 8.0f,
                    1.0f / 4.0f, 1.0f / 4.0f};
                value = (float)ba_byte(obs, in, record + f) * scales[f];
            } else if (f == 5) {
                value = (float)ba_byte(obs, in, record + 5) / 255.0f;
            } else if (f == 6) {
                value = (float)ba_i16(obs, in, record + 6) / (256.0f * 16.0f);
            } else {
                value = (float)ba_u16(obs, in, record + 8 + 2 * (f - 7))
                    / OBS_MAX_PLAYING_CARDS;
            }
        }
    } else if (feature < BA_DECK_OFFSET) {
        int local = feature - BA_HAND_OFFSET;
        int base = offsetof(Observation, hand);
        if (local == 0) {
            value = (float)ba_u16(obs, in, base) / OBS_MAX_HAND;
        } else {
            int rec = (local - 1) / BA_HAND_FEATURES;
            int f = (local - 1) % BA_HAND_FEATURES;
            int record = base + offsetof(ObservationHand, values)
                + rec * sizeof(ObservationHandCard);
            value = f == 0
                ? (float)ba_u16(obs, in, record) / OBS_MAX_PLAYING_VARIANTS
                : (float)ba_byte(obs, in, record + 2) / 255.0f;
        }
    } else if (feature < BA_CARD_OFFSET) {
        int local = feature - BA_DECK_OFFSET;
        int summary = local / BA_DECK_FIELDS;
        int field = local % BA_DECK_FIELDS;
        int base = summary == 0
            ? offsetof(Observation, owned_deck)
            : offsetof(Observation, draw_pile);
        value = (float)ba_u16(obs, in, base + 2 * field)
            / OBS_MAX_PLAYING_CARDS;
    } else if (feature < BA_TAG_OFFSET) {
        int input_base, capacity, card_prefix;
        int local = ba_card_zone(feature - BA_CARD_OFFSET,
            &input_base, &capacity, &card_prefix);
        if (local == 0) {
            value = (float)ba_u16(obs, in, input_base) / capacity;
        } else {
            int rec = (local - 1) / BA_CARD_FEATURES;
            int f = (local - 1) % BA_CARD_FEATURES;
            int record = input_base + 2 + rec * sizeof(ObservationCard);
            int center = ba_u16(obs, in, record);
            int active = rec < ba_u16(obs, in, input_base);
            if (!active) {
                if (f == 0 && center_ids)
                    center_ids[b * (BA_NUM_CARDS + 3) + 3 + card_prefix + rec] = -1;
            } else if (f < BA_CENTER_EMBED) {
                if ((unsigned)center < CENTER_COUNT)
                    value = to_float(center_embed[center * BA_CENTER_EMBED + f]);
                if (f == 0 && center_ids)
                    center_ids[b * (BA_NUM_CARDS + 3) + 3 + card_prefix + rec] = center;
            } else if (f < BA_CENTER_EMBED + 5) {
                static constexpr float scales[5] = {
                    1.0f / 14.0f, 1.0f / 3.0f, 1.0f / 8.0f,
                    1.0f / 4.0f, 1.0f / 4.0f};
                int card_field = f - BA_CENTER_EMBED;
                value = (float)ba_byte(obs, in, record + 2 + card_field)
                    * scales[card_field];
            } else if (f == BA_CENTER_EMBED + 5) {
                value = (float)ba_byte(obs, in, record + 7) / 255.0f;
            } else {
                int qfield = f - (BA_CENTER_EMBED + 6);
                value = (float)ba_i16(obs, in, record + 8 + 2 * qfield)
                    / (256.0f * 16.0f);
            }
        }
    } else if (feature < BA_POKER_OFFSET) {
        int local = feature - BA_TAG_OFFSET;
        int base = offsetof(Observation, tags);
        if (local == 0) {
            value = (float)ba_u16(obs, in, base) / OBS_MAX_TAGS;
        } else {
            int rec = (local - 1) / BA_TAG_FEATURES;
            int f = (local - 1) % BA_TAG_FEATURES;
            int id = ba_byte(obs, in, base + 2 + rec);
            int active = rec < ba_u16(obs, in, base);
            if (!active) {
                if (f == 0 && tag_ids) tag_ids[b * OBS_MAX_TAGS + rec] = -1;
            } else if (f < BA_TAG_EMBED) {
                if ((unsigned)id < TAG_COUNT)
                    value = to_float(tag_embed[id * BA_TAG_EMBED + f]);
                if (f == 0 && tag_ids) tag_ids[b * OBS_MAX_TAGS + rec] = id;
            } else {
                int field = f - BA_TAG_EMBED + 1;
                int offset = base + 2 + field * OBS_MAX_TAGS + rec;
                value = (float)ba_byte(obs, in, offset)
                    / (field == 1 ? (float)HAND_COUNT : 255.0f);
            }
        }
    } else {
        int local = feature - BA_POKER_OFFSET;
        int rec = local / BA_POKER_FEATURES;
        int f = local % BA_POKER_FEATURES;
        int record = offsetof(Observation, poker_hands)
            + rec * sizeof(ObservationPokerHand);
        if (f == 0) value = (float)ba_byte(obs, in, record);
        else if (f == 1) value = (float)ba_u32(obs, in, record + 1) / 16.0f;
        else if (f == 2) value = (float)ba_i16(obs, in, record + 5) / (256.0f * 128.0f);
        else if (f == 3) value = (float)ba_i16(obs, in, record + 7) / (256.0f * 128.0f);
        else if (f == 4) value = (float)ba_u32(obs, in, record + 9) / 256.0f;
        else value = (float)ba_u32(obs, in, record + 13) / 64.0f;
    }
    out[idx] = from_float(value);
}


__global__ void ba_float_to_precision_kernel(
        precision_t* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = from_float(src[idx]);
}

static BalatroEncoderWeights* ba_encoder_create(int obs_size, int hidden) {
    assert(obs_size == (int)sizeof(Observation)
        && "Balatro encoder expects packed Observation bytes");
    BalatroEncoderWeights* ew =
        (BalatroEncoderWeights*)calloc(1, sizeof(BalatroEncoderWeights));
    ew->obs_size = obs_size;
    ew->hidden = hidden;
    return ew;
}

static Prec ba_encoder_forward(
        void* w, void* activations, PolicyObs input, cudaStream_t stream) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    BalatroEncoderActivations* a = (BalatroEncoderActivations*)activations;
    int B = input.shape[0];
    ba_extract_kernel<<<grid_size(B * BA_FEATURES), BLOCK_SIZE, 0, stream>>>(
        a->features.data, a->center_ids.data, a->blind_ids.data, a->tag_ids.data,
        input.data, ew->center_embed.data, ew->blind_embed.data,
        ew->tag_embed.data, B, ew->obs_size);
    puf_mm(&a->features, &ew->proj_w, &a->out, stream);
    return a->out;
}

static void ba_encoder_backward(
        void* w, void* activations, Prec grad, cudaStream_t stream) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    BalatroEncoderActivations* a = (BalatroEncoderActivations*)activations;
    int B = grad.shape[0];
    puf_mm_tn(&grad, &a->features, &a->proj_wgrad, stream);
    int embed_n = CENTER_COUNT * BA_CENTER_EMBED;
    int blind_n = BLIND_COUNT * BA_BLIND_EMBED;
    int tag_n = TAG_COUNT * BA_TAG_EMBED;
    cudaMemsetAsync(a->center_embed_wgrad_f.data, 0,
        embed_n * sizeof(float), stream);
    cudaMemsetAsync(a->blind_embed_wgrad_f.data, 0,
        blind_n * sizeof(float), stream);
    cudaMemsetAsync(a->tag_embed_wgrad_f.data, 0,
        tag_n * sizeof(float), stream);
    ba_gather_embed_projection<<<
        grid_size(BA_EMBED_FEATURES * ew->hidden), BLOCK_SIZE, 0, stream>>>(
        a->embed_proj.data, ew->proj_w.data, ew->hidden);
    puf_mm(&grad, &a->embed_proj, &a->embed_grad, stream);
    ba_embed_backward_kernel<<<
        grid_size(B * BA_EMBED_FEATURES), BLOCK_SIZE, 0, stream>>>(
        a->center_embed_wgrad_f.data, a->blind_embed_wgrad_f.data,
        a->tag_embed_wgrad_f.data, a->embed_grad.data,
        a->center_ids.data, a->blind_ids.data, a->tag_ids.data, B);
    ba_float_to_precision_kernel<<<grid_size(embed_n), BLOCK_SIZE, 0, stream>>>(
        a->center_embed_wgrad.data, a->center_embed_wgrad_f.data, embed_n);
    ba_float_to_precision_kernel<<<grid_size(blind_n), BLOCK_SIZE, 0, stream>>>(
        a->blind_embed_wgrad.data, a->blind_embed_wgrad_f.data, blind_n);
    ba_float_to_precision_kernel<<<grid_size(tag_n), BLOCK_SIZE, 0, stream>>>(
        a->tag_embed_wgrad.data, a->tag_embed_wgrad_f.data, tag_n);
}

static void ba_encoder_init_weights(
        void* w, uint64_t* seed, cudaStream_t stream) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    puf_normal_init(&ew->center_embed, 0.25f, (*seed)++, stream);
    puf_normal_init(&ew->blind_embed, 0.25f, (*seed)++, stream);
    puf_normal_init(&ew->tag_embed, 0.25f, (*seed)++, stream);
    Prec projection = {
        .data = ew->proj_w.data,
        .shape = {ew->hidden, BA_FEATURES},
    };
    puf_kaiming_init(&projection, sqrtf(2.0f), (*seed)++, stream);
}

static void ba_encoder_reg_params(void* w, Allocator* alloc) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    ew->center_embed = {.shape = {CENTER_COUNT, BA_CENTER_EMBED}};
    ew->blind_embed = {.shape = {BLIND_COUNT, BA_BLIND_EMBED}};
    ew->tag_embed = {.shape = {TAG_COUNT, BA_TAG_EMBED}};
    ew->proj_w = {.shape = {ew->hidden, BA_FEATURES}};
    alloc_register(alloc, &ew->center_embed);
    alloc_register(alloc, &ew->blind_embed);
    alloc_register(alloc, &ew->tag_embed);
    alloc_register(alloc, &ew->proj_w);
}

static void ba_encoder_reg_train(
        void* w, void* activations, Allocator* acts,
        Allocator* grads, int B_TT) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    BalatroEncoderActivations* a = (BalatroEncoderActivations*)activations;
    *a = {};
    a->features = {.shape = {B_TT, BA_FEATURES}};
    a->out = {.shape = {B_TT, ew->hidden}};
    a->embed_proj = {.shape = {BA_EMBED_FEATURES, ew->hidden}};
    a->embed_grad = {.shape = {B_TT, BA_EMBED_FEATURES}};
    a->center_ids = {.shape = {B_TT, BA_NUM_CARDS + 3}};
    a->blind_ids = {.shape = {B_TT, 2}};
    a->tag_ids = {.shape = {B_TT, OBS_MAX_TAGS}};
    a->center_embed_wgrad = {.shape = {CENTER_COUNT, BA_CENTER_EMBED}};
    a->blind_embed_wgrad = {.shape = {BLIND_COUNT, BA_BLIND_EMBED}};
    a->tag_embed_wgrad = {.shape = {TAG_COUNT, BA_TAG_EMBED}};
    a->proj_wgrad = {.shape = {ew->hidden, BA_FEATURES}};
    a->center_embed_wgrad_f = {.shape = {CENTER_COUNT, BA_CENTER_EMBED}};
    a->blind_embed_wgrad_f = {.shape = {BLIND_COUNT, BA_BLIND_EMBED}};
    a->tag_embed_wgrad_f = {.shape = {TAG_COUNT, BA_TAG_EMBED}};
    alloc_register(acts, &a->features);
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->embed_proj);
    alloc_register(acts, &a->embed_grad);
    alloc_register(acts, &a->center_ids);
    alloc_register(acts, &a->blind_ids);
    alloc_register(acts, &a->tag_ids);
    alloc_register(grads, &a->center_embed_wgrad);
    alloc_register(grads, &a->blind_embed_wgrad);
    alloc_register(grads, &a->tag_embed_wgrad);
    alloc_register(grads, &a->proj_wgrad);
    alloc_register(acts, &a->center_embed_wgrad_f);
    alloc_register(acts, &a->blind_embed_wgrad_f);
    alloc_register(acts, &a->tag_embed_wgrad_f);
}

static void ba_encoder_reg_rollout(
        void* w, void* activations, Allocator* alloc, int B) {
    BalatroEncoderWeights* ew = (BalatroEncoderWeights*)w;
    BalatroEncoderActivations* a = (BalatroEncoderActivations*)activations;
    *a = {};
    a->features = {.shape = {B, BA_FEATURES}};
    a->out = {.shape = {B, ew->hidden}};
    alloc_register(alloc, &a->features);
    alloc_register(alloc, &a->out);
}

static void* ba_encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    return ba_encoder_create(e->in_dim, e->out_dim);
}

static void create_balatro_encoder(Encoder* enc) {
    *enc = Encoder{
        .forward = ba_encoder_forward,
        .backward = ba_encoder_backward,
        .init_weights = ba_encoder_init_weights,
        .reg_params = ba_encoder_reg_params,
        .reg_train = ba_encoder_reg_train,
        .reg_rollout = ba_encoder_reg_rollout,
        .create_weights = ba_encoder_create_weights,
        .in_dim = enc->in_dim,
        .out_dim = enc->out_dim,
        .activation_size = sizeof(BalatroEncoderActivations),
    };
}
