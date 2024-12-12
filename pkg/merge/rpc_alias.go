package merge

import (
	"log"
	"sort"

	"federate/pkg/manifest"
	"federate/pkg/merge/property"
	"federate/pkg/spring"
	"federate/pkg/util"
)

// 处理 JSF/Dubbo provider alias/group 冲突
// 对于同一个 interface，它只能有1个alias/group
type RpcAliasManager struct {
	m  *manifest.Manifest
	pm *property.PropertyManager
}

func NewRpcAliasManager(pm *property.PropertyManager) Reconciler {
	return &RpcAliasManager{pm: pm, m: pm.M()}
}

func (m *RpcAliasManager) Name() string {
	return "Detect RPC Provider alias/group conflicts by Rewriting XML"
}

func (m *RpcAliasManager) DetectOnly() bool {
	return true
}

func (m *RpcAliasManager) Reconcile() error {
	// pass 1: search the conflicting alias/group
	springMgr := spring.New(false)
	beans := springMgr.ListBeans(m.m.SpringXmlPath(), spring.QueryRpcAlias())

	aliasMap := make(map[string][]string)
	for _, b := range beans {
		iface := b.Bean.SelectAttrValue("interface", "")
		if iface != "" {
			if aliasMap[iface] == nil {
				aliasMap[iface] = make([]string, 0)
			}

			alias := m.pm.ResolveLine(b.Value)
			aliasMap[iface] = append(aliasMap[iface], alias)
		}
	}

	conflicted := false
	for iface, aliases := range aliasMap {
		if len(aliases) > 1 && len(util.UniqueStrings(aliases)) != len(aliases) {
			conflicted = true

			// sort aliases
			sort.Strings(aliases)

			log.Printf("Interface: %s", iface)
			for _, alias := range aliases {
				log.Printf("  - %s", alias)
			}
		}
	}

	if !conflicted {
		log.Printf("RPC Provider alias found no conflicts!")
		return nil
	} else if FailFast {
		log.Fatal("You need to solve RPC Provider Alias Conflict before proceeding!")
	}

	// pass 2: fix the conflicts

	return nil
}
