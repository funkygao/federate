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
    XORQ SI, DI
    //POPCNT DI, AX 
    BYTE $0xf3; BYTE $0x48; BYTE $0x0f; BYTE $0xb8; BYTE $0xc7 // Go 汇编器不直接支持 POPCNT 助记符，但可插入机器码，这行对应于POPCNT DI, AX
    MOVQ AX, ret+16(FP)
    RET
