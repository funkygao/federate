package merge

import (
	"fmt"
	"log"
	"sort"

	"federate/pkg/manifest"
	"federate/pkg/spring"
)

// 处理 JSF/Dubbo provider alias/group 冲突
// 对于同一个 interface，它只能有1个alias/group
type RpcAliasManager struct {
	m  *manifest.Manifest
	pm *PropertyManager
}

func NewRpcAliasManager(pm *PropertyManager) *RpcAliasManager {
	return &RpcAliasManager{pm: pm, m: pm.m}
}

func (m *RpcAliasManager) Reconcile() error {
	// pass 1: search the conflicting alias/group
	springMgr := spring.New(false)
	beans := springMgr.ListBeans(m.m.SpringXmlPath(), spring.SearchByAlias)

	aliasMap := make(map[string][]string)
	for _, b := range beans {
		iface := b.Bean.SelectAttrValue("interface", "")
		alias := m.pm.ResolveLine(b.Identifier)
		if alias != b.Identifier {
			alias = fmt.Sprintf("%60s  %s", alias, b.Identifier)
		}
		if iface != "" {
			if aliasMap[iface] == nil {
				aliasMap[iface] = make([]string, 0)
			}

			aliasMap[iface] = append(aliasMap[iface], alias)
		}
	}

	for iface, aliases := range aliasMap {
		// sort aliases
		sort.Strings(aliases)

		if len(aliases) > 1 {
			log.Printf("Interface: %s", iface)
			for _, alias := range aliases {
				log.Printf("  - %s", alias)
			}
		}
	}

	// pass 2: fix the conflicts

	return nil
}
