package merge

import (
	"log"

	"federate/pkg/manifest"
	"federate/pkg/merge"
	"github.com/fatih/color"
)

func mergeRpcConsumerXml(m *manifest.Manifest, managers []*merge.RpcConsumerManager) {
	for _, manager := range managers {
		mergeConsumerXml(m, manager)
		manager.Reset()
	}
}

func mergeConsumerXml(m *manifest.Manifest, manager *merge.RpcConsumerManager) {
	if err := manager.MergeConsumerXmlFiles(m); err != nil {
		log.Fatalf("Error merging %s consumer xml: %v", manager.RPC(), err)
	}

	if manager.ScannedBeansCount == 0 {
		return
	}

	color.Cyan("[%s] ScannedBeans:%d, IgnoredInterface:%d, GeneratedBeans:%d, InterComponentConflicts:%d", manager.RPC(), manager.ScannedBeansCount,
		manager.IgnoredInterfaceN, manager.GeneratedBeansCount, len(manager.InterComponentConflicts))
	if len(manager.InterComponentConflicts) > 0 {
		log.Printf("[%s] InterComponentConflicts: %v", manager.RPC(), manager.InterComponentConflicts)
	}
	for component, conflicts := range manager.IntraComponentConflicts {
		if len(conflicts) > 0 {
			color.Yellow("[%s:%s] IntraComponentConflicts: %v", manager.RPC(), component, conflicts)
		}
	}
	color.Green("🍺 Consolidated into %s", manager.TargetFile)
}
