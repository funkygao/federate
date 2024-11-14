package merge

import (
	"log"

	"federate/pkg/manifest"
	"federate/pkg/merge"
	"github.com/fatih/color"
)

func reconcileBeanInjectionConflicts(m *manifest.Manifest, manager *merge.SpringBeanInjectionManager) {
	if err := manager.Reconcile(m, dryRunMerge); err != nil {
		log.Fatalf("%v", err)
	}

	if manager.Updated > 0 {
		log.Printf("Source Code Rewritten, @Resource -> @Autowired: %d", manager.Updated)
	}
	color.Green("🍺 Java code Spring Bean Injection conflicts reconciled")
}
