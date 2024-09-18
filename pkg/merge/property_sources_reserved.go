package merge

import (
	"strings"

	"federate/pkg/manifest"
)

// ComponentKeyValue represents a key-value pair for a specific component.
type ComponentKeyValue struct {
	Component manifest.ComponentInfo
	Value     interface{}
}

// ValueOverride is a type alias for functions that calculate special key values.
type ValueOverride func([]ComponentKeyValue) interface{}

func M(values []ComponentKeyValue) *manifest.MainSystem {
	return values[0].Component.M
}

var reservedKeyHandlers = map[string]ValueOverride{
	"spring.application.name": func(values []ComponentKeyValue) interface{} {
		return M(values).Name
	},
	"spring.profiles.active": func(values []ComponentKeyValue) interface{} {
		return M(values).SpringProfile
	},
	"spring.profiles.include": func(values []ComponentKeyValue) interface{} {
		return nil
	},
	"spring.messages.basename": func(values []ComponentKeyValue) interface{} {
		basenameSet := make(map[string]struct{})
		for _, v := range values {
			if basenameStr, ok := v.Value.(string); ok {
				basenames := strings.Split(basenameStr, ",")
				for _, basename := range basenames {
					trimmedBasename := strings.TrimSpace(basename)
					if trimmedBasename != "" {
						basenameSet[trimmedBasename] = struct{}{}
					}
				}
			}
		}

		var uniqueBasenames []string
		for basename := range basenameSet {
			uniqueBasenames = append(uniqueBasenames, basename)
		}

		return strings.Join(uniqueBasenames, ",")
	},
	"logging.config": func(values []ComponentKeyValue) interface{} {
		return nil
	},
	"mybatis.config-location": func(values []ComponentKeyValue) interface{} {
		return nil
	},
	"server.servlet.context-path#": func(values []ComponentKeyValue) interface{} {
		return nil
	},
}
