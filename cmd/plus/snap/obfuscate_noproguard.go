//go:build !proguard

package snap

import (
	"federate/pkg/step"
	"github.com/fatih/color"
)

func obfuscateJars(bar step.Bar) {
	color.Red("make install INCLUDE_PROGUARD=1 to Enable JAR Obfuscate")
}
