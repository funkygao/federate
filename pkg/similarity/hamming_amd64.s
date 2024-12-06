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
    XORQ SI, DI             // DI = hash1 ^ hash2

    // POPCNT DI, AX 指令的机器码表示(Go 汇编器不直接支持 POPCNT 助记符)
    BYTE $0xf3 // REPZ 前缀：标识这是 POPCNT 指令
    BYTE $0x48 // REX.W 前缀：指示这是 64 位操作
    BYTE $0x0f // 两字节操作码的第一个字节
    BYTE $0xb8 // POPCNT 指令的主操作码
    BYTE $0xc7 // ModR/M 字节：11 000 111
                    // 11    - 寄存器对寄存器操作
                    // 000   - 操作码扩展（POPCNT 固定为 000）
                    // 111   - 源寄存器是 DI
                    // (目标寄存器 AX 是隐含的)

    MOVQ AX, ret+16(FP)
    RET
