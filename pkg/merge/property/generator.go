package property

import (
	"fmt"

	"federate/pkg/ds"
)

func (cm *PropertyManager) GenerateMergedYamlFile(targetFile string) error {
	parser, supported := ParserByFile(targetFile)
	if !supported {
		return fmt.Errorf("unsupported file type: %s", targetFile)
	}

	entries := make(map[string]PropertyEntry)
	processedKeys := ds.NewStringSet()

	for _, component := range cm.m.Components {
		for key, entry := range cm.r.ComponentYamlEntries(component) {
			if !processedKeys.Contains(key) {
				entries[key] = entry
				processedKeys.Add(key)
			}
		}
	}

	rawKeys := cm.m.Main.Reconcile.Resources.Property.RawKeys
	return parser.Generate(entries, rawKeys, targetFile)
}

func (cm *PropertyManager) GenerateMergedPropertiesFile(targetFile string) error {
	parser, supported := ParserByFile(targetFile)
	if !supported {
		return fmt.Errorf("unsupported file type: %s", targetFile)
	}

	entries := make(map[string]PropertyEntry)
	processedKeys := ds.NewStringSet()
	for _, component := range cm.m.Components {
		for key, entry := range cm.r.ComponentYamlEntries(component) {
			if !processedKeys.Contains(key) {
				entries[key] = entry
				processedKeys.Add(key)
			}
		}
	}

	return parser.Generate(entries, nil, targetFile)
}
