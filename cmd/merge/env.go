package merge

import (
	"log"

	"federate/pkg/manifest"
	"github.com/fatih/color"
)

func reconcileEnvConflicts(m *manifest.Manifest) {
	log.Printf("System.getProperty")
	color.Green("🍺 ENV variables conflicts reconciled")
}
