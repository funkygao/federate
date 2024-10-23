package merge

import (
	"federate/pkg/manifest"
	"federate/pkg/merge"
	"github.com/fatih/color"
)

func reconcileTargetXmlBeanConflicts(m *manifest.Manifest, manager *merge.XmlBeanManager) {
	manager.ReconcileTargetConflicts(dryRunMerge)
	plan := manager.ReconcilePlan()
	color.Yellow("Found bean id conflicts: %d", plan.ConflictCount())
	color.Green("🍺 Spring XML Beans conflicts reconciled")
}
