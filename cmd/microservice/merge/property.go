package merge

import (
	"fmt"
	"log"
	"path/filepath"
	"reflect"
	"sort"

	"federate/pkg/federated"
	"federate/pkg/merge/property"
	"federate/pkg/tablerender"
	"federate/pkg/util"
	"github.com/fatih/color"
)

func identifyPropertyConflicts(manager *property.PropertyManager) {
	if err := manager.Analyze(); err != nil {
		log.Fatalf("%v, Error type: %s", err, reflect.TypeOf(err))
	}
	showPropertiesConflicts(manager)
	showYamlConflicts(manager)
}

func reconcilePropertiesConflicts(manager *property.PropertyManager) {
	if err := manager.Reconcile(dryRunMerge); err != nil {
		log.Fatalf("%v, Error type: %s", err, reflect.TypeOf(err))
	}

	result := manager.Result()

	pn := filepath.Join(federated.GeneratedResourceBaseDir(manager.M().Main.Name), "application.properties")
	if err := manager.GenerateMergedPropertiesFile(pn); err != nil {
		log.Fatalf("%v", err)
	} else {
		log.Printf("Generated %s", pn)
	}

	an := filepath.Join(federated.GeneratedResourceBaseDir(manager.M().Main.Name), "application.yml")
	if err := manager.GenerateMergedYamlFile(an); err != nil {
		log.Fatalf("%v", err)
	} else {
		log.Printf("Generated %s", pn)
	}
	log.Printf("Source code rewritten, @RequestMapping: %d, @Value: %d, @ConfigurationProperties: %d",
		result.RequestMapping, result.KeyPrefixed, result.ConfigurationProperties)
	color.Green("🍺 Reconciled property conflicts: %s, %s", an, pn)
}

func showPropertiesConflicts(manager *property.PropertyManager) {
	conflictKeys := manager.IdentifyPropertiesFileConflicts()
	if len(conflictKeys) == 0 {
		log.Printf("Bingo! .properties files found no conflicts")
		return
	}

	log.Printf("Found .properties conflicts: %d", len(conflictKeys))
	keys := make([]string, 0, len(conflictKeys))
	for key := range conflictKeys {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	header := []string{"Conflicting Key"}
	for _, component := range manager.M().Components {
		header = append(header, component.Name)
	}

	var rows [][]string
	for _, key := range keys {
		row := []string{key}
		for _, componentName := range header[1:] {
			if val, exists := conflictKeys[key][componentName]; exists {
				row = append(row, util.Truncate(fmt.Sprintf("%v", val), yamlConflictCellMaxWidth))
			} else {
				row = append(row, "")
			}
		}
		rows = append(rows, row)
	}
	tablerender.DisplayTable(header, rows, false, 0)

}
func showYamlConflicts(manager *property.PropertyManager) {
	conflictKeys := manager.IdentifyYamlFileConflicts()
	if len(conflictKeys) == 0 {
		log.Printf("application.yml files found no conflicts, bingo!")
		return
	}

	log.Printf("Found application.yml conflicts: %d", len(conflictKeys))
	keys := make([]string, 0, len(conflictKeys))
	for key := range conflictKeys {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	header := []string{"Conflicting Key"}
	for _, component := range manager.M().Components {
		header = append(header, component.Name)
	}
	var rows [][]string

	for _, key := range keys {
		row := []string{key}
		for _, componentName := range header[1:] { // Skip the first header: "Conflicting Key"
			if val, exists := conflictKeys[key][componentName]; exists {
				row = append(row, util.Truncate(fmt.Sprintf("%v", val), yamlConflictCellMaxWidth))
			} else {
				row = append(row, "")
			}
		}
		rows = append(rows, row)
	}
	tablerender.DisplayTable(header, rows, false, 0)
}
