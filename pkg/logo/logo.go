package logo

import (
	"github.com/fatih/color"
)

func Federate() string {
	return color.New(color.Bold).Add(color.Underline).Add(color.FgCyan).Sprintf("federate")
}
