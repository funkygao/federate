package merge

import (
	"log"

	"federate/pkg/merge"
	"github.com/fatih/color"
)

func mergeRpcConsumerXml(managers []*merge.RpcConsumerManager) {
	for _, manager := range managers {
		if err := manager.Reconcile(dryRunMerge); err != nil {
			log.Fatalf("Error merging %s consumer xml: %v", manager.RPC(), err)
		}

		if manager.ScannedBeansCount == 0 {
			return
		}

		log.Printf("[%s] ScannedBeans:%d, IgnoredInterface:%d, GeneratedBeans:%d, InterComponentConflicts:%d", manager.RPC(), manager.ScannedBeansCount,
			manager.IgnoredInterfaceN, manager.GeneratedBeansCount, len(manager.InterComponentConflicts))
		if len(manager.InterComponentConflicts) > 0 {
			log.Printf("[%s] InterComponentConflicts: %v", manager.RPC(), manager.InterComponentConflicts)
		}
		for component, conflicts := range manager.IntraComponentConflicts {
			if len(conflicts) > 0 {
				color.Yellow("[%s:%s] IntraComponentConflicts: %v", manager.RPC(), component, conflicts)
			}
		}
		color.Green("🍺 Merged into %s", manager.TargetFile)

		manager.Reset()
	}
}
