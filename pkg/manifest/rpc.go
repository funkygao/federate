package manifest

import (
	"strings"
)

type RpcSpec struct {
	Provider RpcProviderSpec `yaml:"provider"`
	Consumer RpcConsumerSpec `yaml:"consumer"`
}

type RpcConsumerSpec struct {
	IgnoreRules []IgnoreRuleSpec `yaml:"ignoreRules"`
}

type IgnoreRuleSpec struct {
	Package string   `yaml:"package"`
	Except  []string `yaml:"except"`
}

func (s *RpcConsumerSpec) IgnoreInterface(interfaceName string) bool {
	for _, rule := range s.IgnoreRules {
		if strings.HasPrefix(interfaceName, rule.Package) {
			// 检查是否在例外列表中
			for _, except := range rule.Except {
				if interfaceName == except {
					return false
				}
			}
			return true
		}
	}
	return false
}

type RpcProviderSpec struct {
}
