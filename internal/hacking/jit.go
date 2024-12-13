package hacking

import (
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
)

func jit() {
	// x86 汇编指令，用于打印 "Hello World!"
	printFunction := []uint16{
		0x48c7, 0xc001, 0x0, // mov %rax,$0x1
		0x48, 0xc7c7, 0x100, 0x0, // mov %rdi,$0x1
		0x48c7, 0xc20c, 0x0, // mov 0x13, %rdx
		0x48, 0x8d35, 0x400, 0x0, // lea 0x4(%rip), %rsi
		0xf05,                  // syscall
		0xc3cc,                 // ret
		0x4865, 0x6c6c, 0x6f20, // Hello_(whitespace)
		0x576f, 0x726c, 0x6421, 0xa, // World!
	}

	// 分配可执行内存
	executablePrintFunc, _ := syscall.Mmap(
		-1,  // 文件描述符，-1 表示不映射文件
		0,   // 起始地址，0 表示让系统选择
		128, // 分配的内存大小
		syscall.PROT_READ|syscall.PROT_WRITE|syscall.PROT_EXEC, // PROT_EXEC 确保新分配的内存地址是可执行的
		syscall.MAP_PRIVATE|unix.MAP_ANON)                      // 内存映射标志

	// 将 printFunction 中的指令复制到可执行内存中
	j := 0
	for i := range printFunction {
		executablePrintFunc[j] = byte(printFunction[i] >> 8) // 高8位
		executablePrintFunc[j+1] = byte(printFunction[i])    // 低8位
		j = j + 2
	}

	type printFunc func()

	// 将可执行内存转换为函数指针
	unsafePrintFunc := (uintptr)(unsafe.Pointer(&executablePrintFunc))
	printer := *(*printFunc)(unsafe.Pointer(&unsafePrintFunc))

	// 执行生成的函数
	printer()
}
