// +build arm64

#include "textflag.h"

// func hammingDistance(hash1, hash2 uint64) int
TEXT ·hammingDistance(SB), NOSPLIT, $0-24
    MOVD hash1+0(FP), R0
    MOVD hash2+8(FP), R1
    EOR R0, R1, R0
    FMOVD R0, F0
    CNT V0.8B, V0.8B
    ADDV V0.8B, V0.8B
    FMOVD F0, R0
    MOVD R0, ret+16(FP)
    RET
