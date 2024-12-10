package property

import (
	"fmt"
	"log"
	"path/filepath"
	"strings"

	"federate/pkg/federated"
	"federate/pkg/manifest"
	"federate/pkg/primitive"
	"federate/pkg/tablerender"
	"github.com/fatih/color"
)

type ComponentPropertyValue struct {
	Component manifest.ComponentInfo
	Value     any
}

// If return value is nil, the key will be deleted.
type ReservedPropertyHandler func(*PropertyManager, []ComponentPropertyValue) any

func M(values []ComponentPropertyValue) *manifest.MainSystem {
	return values[0].Component.M
}

var reservedKeyHandlers = map[string]ReservedPropertyHandler{
	"spring.application.name": func(m *PropertyManager, values []ComponentPropertyValue) any {
		return M(values).Name
	},
	"spring.profiles.active": func(m *PropertyManager, values []ComponentPropertyValue) any {
		return M(values).SpringProfile
	},
	"spring.profiles.include": func(m *PropertyManager, values []ComponentPropertyValue) any {
		return nil
	},
	"spring.messages.basename": func(m *PropertyManager, values []ComponentPropertyValue) any {
		// i18n message bundles path
		basenameSet := primitive.NewStringSet()
		for _, v := range values {
			if basenameStr, ok := v.Value.(string); ok {
				basenames := strings.Split(basenameStr, ",")
				for _, basename := range basenames {
					trimmedBasename := strings.TrimSpace(basename)
					if trimmedBasename != "" {
						federatedBasename := filepath.Join(federated.FederatedDir, v.Component.Name, trimmedBasename)
						basenameSet.Add(federatedBasename)

					}
				}
			}
		}

		return strings.Join(basenameSet.Values(), ",")
	},
	"logging.config": func(m *PropertyManager, values []ComponentPropertyValue) any {
		return nil
	},
	"mybatis.config-location": func(m *PropertyManager, values []ComponentPropertyValue) any {
		return nil
	},
	"server.port": func(m *PropertyManager, values []ComponentPropertyValue) any {
		return 8080
	},
	"server.servlet.context-path": func(m *PropertyManager, values []ComponentPropertyValue) any {
		for _, v := range values {
			if contextPath, ok := v.Value.(string); ok {
				m.recordServletContextPath(v.Component, contextPath)
			}
		}
		return "/"
	},
	"server.tomcat.accesslog.directory": func(m *PropertyManager, values []ComponentPropertyValue) any {
		return "${LOG_HOME}"
	},
}

func (m *PropertyManager) recordServletContextPath(c manifest.ComponentInfo, contextPath string) {
	m.servletContextPath[c.Name] = contextPath
}

func (pm *PropertyManager) applyReservedPropertyRules() {
	var cellData [][]string
	for key, values := range pm.r.GetReservedPropertyValues() {
		if handler, exists := reservedKeyHandlers[key]; exists {
			if pm.m.Main.Reconcile.PropertySettled(key) {
				color.Yellow("key:%s reserved, but used directive: propertySettled, skipped", key)
				continue
			}

			if value := handler(pm, values); value != nil {
				for _, entries := range pm.r.GetAllResolvableEntries() {
					entries[key] = PropertyEntry{
						Value:    value,
						FilePath: "reserved.yml",
					}
				}

				cellData = append(cellData, []string{key, fmt.Sprintf("%v", value)})
			} else {
				for _, entries := range pm.r.GetAllResolvableEntries() {
					delete(entries, key)
				}
				cellData = append(cellData, []string{key, color.New(color.FgRed).Add(color.CrossedOut).Sprintf("deleted")})
			}
		}
	}

	if !pm.silent && len(cellData) > 0 {
		log.Printf("Reserved keys processed:")
		header := []string{"Reserved Key", "Value"}
		tablerender.DisplayTable(header, cellData, false, -1)
	}
}
