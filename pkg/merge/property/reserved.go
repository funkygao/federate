package property

import (
	"fmt"
	"log"
	"strings"

	"federate/pkg/manifest"
	"federate/pkg/tablerender"
	"github.com/fatih/color"
)

// ComponentKeyValue represents a key-value pair for a specific component.
type ComponentKeyValue struct {
	Component manifest.ComponentInfo
	Value     interface{}
}

// ValueOverride is a type alias for functions that calculate special key values.
// If return value is nil, the key will be deleted.
type ValueOverride func(*PropertyManager, []ComponentKeyValue) interface{}

func M(values []ComponentKeyValue) *manifest.MainSystem {
	return values[0].Component.M
}

func (cm *PropertyManager) isReservedProperty(key string) bool {
	_, exists := cm.reservedYamlKeys[key]
	return exists
}

var reservedKeyHandlers = map[string]ValueOverride{
	"spring.application.name": func(m *PropertyManager, values []ComponentKeyValue) interface{} {
		return M(values).Name
	},
	"spring.profiles.active": func(m *PropertyManager, values []ComponentKeyValue) interface{} {
		return M(values).SpringProfile
	},
	"spring.profiles.include": func(m *PropertyManager, values []ComponentKeyValue) interface{} {
		return nil
	},
	"spring.messages.basename": func(m *PropertyManager, values []ComponentKeyValue) interface{} {
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
	"logging.config": func(m *PropertyManager, values []ComponentKeyValue) interface{} {
		return nil
	},
	"mybatis.config-location": func(m *PropertyManager, values []ComponentKeyValue) interface{} {
		return nil
	},
	"server.servlet.context-path": func(m *PropertyManager, values []ComponentKeyValue) interface{} {
		for _, v := range values {
			if contextPath, ok := v.Value.(string); ok {
				m.recordServletContextPath(v.Component, contextPath)
			}
		}
		return "/"
	},
	"server.tomcat.accesslog.directory": func(m *PropertyManager, values []ComponentKeyValue) interface{} {
		return "${LOG_HOME}"
	},
}

func (m *PropertyManager) recordServletContextPath(c manifest.ComponentInfo, contextPath string) {
	m.servletContextPath[c.Name] = contextPath
}

func (cm *PropertyManager) registerReservedProperty(key string, component manifest.ComponentInfo, value interface{}) {
	if _, exists := cm.reservedValues[key]; !exists {
		cm.reservedValues[key] = []ComponentKeyValue{}
	}
	cm.reservedValues[key] = append(cm.reservedValues[key], ComponentKeyValue{Component: component, Value: value})
}

func (cm *PropertyManager) applyReservedPropertyRules() {
	var cellData [][]string
	for key, values := range cm.reservedValues {
		if handler, exists := cm.reservedYamlKeys[key]; exists {
			if cm.m.Main.Reconcile.PropertySettled(key) {
				color.Yellow("key:%s reserved, but used directive: propertySettled, skipped", key)
				continue
			}

			if value := handler(cm, values); value != nil {
				for _, componentProps := range cm.resolvedProperties {
					componentProps[key] = PropertySource{
						Value:    value,
						FilePath: "reserved.yml",
					}
				}

				cellData = append(cellData, []string{key, fmt.Sprintf("%v", value)})
			} else {
				for _, componentProps := range cm.resolvedProperties {
					delete(componentProps, key)
				}
				cellData = append(cellData, []string{key, color.New(color.FgRed).Add(color.CrossedOut).Sprintf("deleted")})
			}
		}
	}

	if !cm.silent {
		log.Printf("Reserved keys processed:")
		header := []string{"Reserved Key", "Value"}
		tablerender.DisplayTable(header, cellData, false, -1)
	}
}
