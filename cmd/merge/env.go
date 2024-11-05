package merge

import (
	"federate/pkg/manifest"
	"federate/pkg/merge"
	"github.com/fatih/color"
)

func reconcileEnvConflicts(m *manifest.Manifest) {
	merge.ReconcileEnvConflicts(m)
	color.Green("🍺 ENV variables conflicts reconciled")
}
