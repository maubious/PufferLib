#include "balatro.h"

#define OBS_SIZE ((int)sizeof(BalatroObservation))
#define NUM_ATNS 9
#define ACT_SIZES {23, 64, 6, 64, 64, 64, 64, 64, 64}
#define OBS_TENSOR_T FloatTensor
#define MY_ACTION_MASK (23 + 64 + 6 + 64 + 64 + 64 + 64 + 64 + 64)
#define Env Env
#include "vecenv.h"
