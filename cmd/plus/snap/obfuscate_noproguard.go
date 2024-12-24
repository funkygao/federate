//go:build !proguard

package snap

import (
	"federate/pkg/manifest"
	"federate/pkg/step"
	"github.com/fatih/color"
)

func obfuscateJars(m *manifest.Manifest, bar step.Bar) {
	color.Red("'federate', make install INCLUDE_PROGUARD=1 to Enable JAR Obfuscate")
}
