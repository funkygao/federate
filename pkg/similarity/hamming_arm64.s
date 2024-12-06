// +build arm64

#include "textflag.h"

// func hammingDistance(hash1, hash2 uint64) int
TEXT ·hammingDistance(SB), NOSPLIT, $0-24
    MOVD hash1+0(FP), R0
    MOVD hash2+8(FP), R1
    EOR  R0, R1, R2    // R2 = hash1 ^ hash2
    MOVD $0, R0        // Initialize count to 0
    MOVD $64, R3       // Set up loop counter

loop:
    ANDS $1, R2, R4    // Check least significant bit
    CBNZ R4, increment // If bit is set, increment count: Compare and Branch if Non-Zero
    B    next

increment:
    ADD  $1, R0        // Increment count

next:
    LSR  $1, R2        // Logical shift right by 1
    SUBS $1, R3        // Decrement loop counter
    CBNZ R3, loop      // Continue if counter is not zero

    MOVD R0, ret+16(FP)
    RET
