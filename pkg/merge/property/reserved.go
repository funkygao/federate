package property

import (
	"fmt"
	"log"
	"strings"

	"federate/pkg/manifest"
	"federate/pkg/tablerender"
	"github.com/fatih/color"
)

type ComponentPropertyValue struct {
	Component manifest.ComponentInfo
	Value     interface{}
}

// If return value is nil, the key will be deleted.
type ReservedPropertyHandler func(*PropertyManager, []ComponentPropertyValue) interface{}

func M(values []ComponentPropertyValue) *manifest.MainSystem {
	return values[0].Component.M
}

var reservedKeyHandlers = map[string]ReservedPropertyHandler{
	"spring.application.name": func(m *PropertyManager, values []ComponentPropertyValue) interface{} {
		return M(values).Name
	},
	"spring.profiles.active": func(m *PropertyManager, values []ComponentPropertyValue) interface{} {
		return M(values).SpringProfile
	},
	"spring.profiles.include": func(m *PropertyManager, values []ComponentPropertyValue) interface{} {
		return nil
	},
	"spring.messages.basename": func(m *PropertyManager, values []ComponentPropertyValue) interface{} {
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
	"logging.config": func(m *PropertyManager, values []ComponentPropertyValue) interface{} {
		return nil
	},
	"mybatis.config-location": func(m *PropertyManager, values []ComponentPropertyValue) interface{} {
		return nil
	},
	"server.servlet.context-path": func(m *PropertyManager, values []ComponentPropertyValue) interface{} {
		for _, v := range values {
			if contextPath, ok := v.Value.(string); ok {
				m.recordServletContextPath(v.Component, contextPath)
			}
		}
		return "/"
	},
	"server.tomcat.accesslog.directory": func(m *PropertyManager, values []ComponentPropertyValue) interface{} {
		return "${LOG_HOME}"
	},
}

func (m *PropertyManager) recordServletContextPath(c manifest.ComponentInfo, contextPath string) {
	m.servletContextPath[c.Name] = contextPath
}

func (cm *PropertyManager) applyReservedPropertyRules() {
	var cellData [][]string
	for key, values := range cm.r.GetReservedPropertyValues() {
		if handler, exists := reservedKeyHandlers[key]; exists {
			if cm.m.Main.Reconcile.PropertySettled(key) {
				color.Yellow("key:%s reserved, but used directive: propertySettled, skipped", key)
				continue
			}

			if value := handler(cm, values); value != nil {
				for _, entries := range cm.r.GetAllResolvableEntries() {
					entries[key] = PropertyEntry{
						Value:    value,
						FilePath: "reserved.yml",
					}
				}

				cellData = append(cellData, []string{key, fmt.Sprintf("%v", value)})
			} else {
				for _, entries := range cm.r.GetAllResolvableEntries() {
					delete(entries, key)
				}
				cellData = append(cellData, []string{key, color.New(color.FgRed).Add(color.CrossedOut).Sprintf("deleted")})
			}
		}
	}

	if !cm.silent && len(cellData) > 0 {
		log.Printf("Reserved keys processed:")
		header := []string{"Reserved Key", "Value"}
		tablerender.DisplayTable(header, cellData, false, -1)
	}
}
