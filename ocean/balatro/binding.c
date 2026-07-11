#include "balatro.h"

#define OBS_SIZE BALATRO_OBSERVATION_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {BALATRO_MAX_LEGAL_ACTIONS}
#define OBS_TENSOR_T FloatTensor
#define MY_ACTION_MASK BALATRO_MAX_LEGAL_ACTIONS
#define Env Env
#include "vecenv.h"

