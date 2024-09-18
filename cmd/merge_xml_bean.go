package cmd

import (
	"federate/pkg/manifest"
	"federate/pkg/merge"
	"github.com/fatih/color"
)

func reconcileTargetXmlBeanConflicts(m *manifest.Manifest, manager *merge.XmlBeanManager, dryRun bool) {
	manager.ReconcileTargetConflicts(dryRun)
	color.Green("🍺 Spring XML Beans conflicts reconciled")
}
