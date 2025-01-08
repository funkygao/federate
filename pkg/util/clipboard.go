package util

import "github.com/atotto/clipboard"

func ClipboardPut(content string) error {
	return clipboard.WriteAll(content)
}
