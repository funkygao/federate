package merge

import (
	"fmt"
	"log"
	"path/filepath"
	"reflect"
	"sort"

	"federate/pkg/federated"
	"federate/pkg/manifest"
	"federate/pkg/merge"
	"federate/pkg/tablerender"
	"federate/pkg/util"
	"github.com/fatih/color"
)

func prepareMergePropertiesFiles(m *manifest.Manifest, manager *merge.PropertySourcesManager) {
	if err := manager.PrepareMergePropertiesFiles(); err != nil {
		log.Fatalf("%v, Error type: %s", err, reflect.TypeOf(err))
	}
	showPropertiesConflicts(m, manager)
}

func prepareMergeApplicationYaml(m *manifest.Manifest, manager *merge.PropertySourcesManager) {
	if err := manager.PrepareMergeApplicationYaml(); err != nil {
		log.Fatalf("%v, Error type: %s", err, reflect.TypeOf(err))
	}
	showYamlConflicts(m, manager)
}

func reconcilePropertiesConflicts(m *manifest.Manifest, manager *merge.PropertySourcesManager) {
	result, err := manager.ReconcileConflicts(dryRunMerge)
	if err != nil {
		log.Fatalf("%v, Error type: %s", err, reflect.TypeOf(err))
	}

	pn := filepath.Join(federated.GeneratedResourceBaseDir(m.Main.Name), "federated.properties")
	manager.WriteMergedProperties(pn)
	an := filepath.Join(federated.GeneratedResourceBaseDir(m.Main.Name), "application.yml")
	manager.WriteMergedYaml(an)
	color.Cyan("Source code rewritten, @RequestMapping: %d, KeyReferencePrefixed: %d", result.RequestMapping, result.KeyPrefixed)
	color.Green("🍺 Reconciled placeholder conflicts: %s, %s", an, pn)
}

func showPropertiesConflicts(m *manifest.Manifest, manager *merge.PropertySourcesManager) {
	conflictKeys := manager.GetPropertiesConflicts()
	if len(conflictKeys) == 0 {
		color.Cyan("Bingo! .properties files found no conflicts")
		return
	}

	color.Red("Found .properties conflicts: %d", len(conflictKeys))
	keys := make([]string, 0, len(conflictKeys))
	for key := range conflictKeys {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	header := []string{"Conflicting Key"}
	for _, component := range m.Components {
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
func showYamlConflicts(m *manifest.Manifest, manager *merge.PropertySourcesManager) {
	conflictKeys := manager.GetYamlConflicts()
	if len(conflictKeys) == 0 {
		color.Cyan("application.yml files found no conflicts, bingo!")
		return
	}

	color.Yellow("Found application.yml conflicts: %d", len(conflictKeys))
	keys := make([]string, 0, len(conflictKeys))
	for key := range conflictKeys {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	header := []string{"Conflicting Key"}
	for _, component := range m.Components {
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
