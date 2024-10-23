package merge

import (
	"log"

	"federate/pkg/manifest"
	"federate/pkg/merge"
	"github.com/fatih/color"
)

func reconcileBeanInjectionConflicts(m *manifest.Manifest, manager *merge.SpringBeanInjectionManager) {
	result, err := manager.ReconcileResourceToAutowired(m, dryRunMerge)
	if err != nil {
		log.Fatalf("%v", err)
	}

	if result.Updated > 0 {
		color.Cyan("Source Code Rewritten, @Resource -> @Autowired: %d", result.Updated)
	}
	color.Green("🍺 Java code Spring Bean Injection conflicts reconciled")
}
