package javast

import (
	"strconv"

	"federate/pkg/manifest"
)

// HITS: Hyperlink-Induced Topic Search
func AnalyzeHITS(component manifest.ComponentInfo, topK int) error {
	d := NewJavastDriver(component)
	return d.Invoke("hits-analysis", strconv.Itoa(topK))
}
