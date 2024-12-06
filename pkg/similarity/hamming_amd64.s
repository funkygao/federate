// +build amd64

#include "textflag.h"

// +----------------+
// | return value   |  16(FP)  8 bytes
// +----------------+
// | hash2          |  8(FP)   8 bytes
// +----------------+
// | hash1          |  0(FP)   8 bytes
// +----------------+
//
// func hammingDistance(hash1, hash2 uint64) int
TEXT ·hammingDistance(SB), NOSPLIT, $0-24
    MOVQ hash1+0(FP), SI
    MOVQ hash2+8(FP), DI

    XORQ SI, DI        // DI = hash1 ^ hash2
    POPCNTQ DI, AX     // Use POPCNT instruction to count bits, POPCNTQ 明确指定了这是一个 64 位（Quadword）操作

    MOVQ AX, ret+16(FP)
    RET
