package merge

import (
	"federate/pkg/merge"
	"github.com/fatih/color"
)

func reconcileRpcAliasConflict(m *merge.RpcAliasManager) {
	m.Reconcile()
	color.Green("🍺 RPC alias/group conflicts reconciled")
}
