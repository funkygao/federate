package merge

import (
	"log"

	"federate/pkg/merge"
	"github.com/fatih/color"
)

func transformServiceValue(m *merge.ServiceManager) {
	if err := m.Reconcile(); err != nil {
		log.Fatalf("%v", err)

	}
	color.Green("🍺 Java @Service and corresponding spring.xml ref transformed")
}
