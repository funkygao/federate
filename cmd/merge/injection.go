package merge

import (
	"log"

	"federate/pkg/manifest"
	"federate/pkg/merge"
	"github.com/fatih/color"
)

func reconcileBeanInjectionConflicts(m *manifest.Manifest, manager *merge.SpringBeanInjectionManager) {
	if err := manager.ReconcileResourceToAutowired(m, dryRunMerge); err != nil {
		log.Fatalf("%v", err)
	}
	color.Green("🍺 Java code Spring Bean injection conflicts reconciled")
}
