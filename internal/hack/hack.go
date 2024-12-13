package hack

import (
	"reflect"
	"unsafe"
)

// []byte -> string
func B2s(b []byte) string {
	return *(*string)(unsafe.Pointer(&b))
}

// string -> []byte
func S2b(s string) []byte {
	strHeader := (*reflect.StringHeader)(unsafe.Pointer(&s))
	bsliceHeader := reflect.SliceHeader{
		Data: strHeader.Data,
		Len:  strHeader.Len,
		Cap:  strHeader.Len,
	}
	return *(*[]byte)(unsafe.Pointer(&bsliceHeader))
}
