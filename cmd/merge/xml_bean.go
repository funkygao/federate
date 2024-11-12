package merge

import (
	"log"

	"federate/pkg/manifest"
	"federate/pkg/merge"
	"github.com/fatih/color"
)

func reconcileTargetXmlBeanConflicts(m *manifest.Manifest, manager *merge.XmlBeanManager) {
	manager.ReconcileTargetConflicts(dryRunMerge)
	plan := manager.ReconcilePlan()
	log.Printf("Found bean id conflicts: %d", plan.ConflictCount())
	color.Green("🍺 Spring XML Beans conflicts reconciled")
}
