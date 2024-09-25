package merge

import (
	"log"

	"federate/pkg/manifest"
	"federate/pkg/merge"
	"github.com/fatih/color"
)

func mergeRpcConsumerXml(m *manifest.Manifest, manager *merge.RpcConsumerManager) {
	rpcs := []string{merge.RpcDubbo, merge.RpcJsf}
	for _, rpc := range rpcs {
		mergeConsumerXml(m, manager, rpc)
		manager.Reset()
	}
}

func mergeConsumerXml(m *manifest.Manifest, manager *merge.RpcConsumerManager, rpc string) {
	if err := manager.MergeConsumerXmlFiles(m, rpc); err != nil {
		log.Fatalf("Error merging %s consumer xml: %v", rpc, err)
	}

	if manager.ScannedBeansCount == 0 {
		return
	}

	color.Cyan("[%s] ScannedBeans:%d, GeneratedBeans:%d, InterComponentConflicts:%d", rpc, manager.ScannedBeansCount,
		manager.GeneratedBeansCount, len(manager.InterComponentConflicts))
	if len(manager.InterComponentConflicts) > 0 {
		log.Printf("[%s] InterComponentConflicts: %v", rpc, manager.InterComponentConflicts)
	}
	for component, conflicts := range manager.IntraComponentConflicts {
		if len(conflicts) > 0 {
			color.Yellow("[%s] IntraComponentConflicts[%s]: %v", rpc, component, conflicts)
		}
	}
	color.Green("🍺 Merged into %s", manager.TargetFile)
}
