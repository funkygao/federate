package merge

import (
	"federate/pkg/manifest"
	"federate/pkg/merge"
	"github.com/fatih/color"
)

func reconcileTargetXmlBeanConflicts(m *manifest.Manifest, manager *merge.XmlBeanManager) {
	manager.ReconcileTargetConflicts(dryRunMerge)
	color.Green("🍺 Spring XML Beans conflicts reconciled")
}
