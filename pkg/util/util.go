package util

func Contains(s string, l []string) bool {
	for _, data := range l {
		if data == s {
			return true
		}
	}
	return false
}

func Truncate(s string, maxLen int) string {
	if len(s) > maxLen {
		return s[:maxLen] + "..."
	}
	return s
}
