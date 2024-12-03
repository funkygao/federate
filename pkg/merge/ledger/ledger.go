package ledger

import (
	"log"
	"sort"
	"strings"

	"federate/pkg/tablerender"
	"federate/pkg/util"
	"github.com/fatih/color"
)

type Ledger struct {
	transactions            map[string]string
	importResource          map[string]map[string]string
	configurationProperties map[string]map[string]string
	regularProperties       map[string]map[string]string
	propertyPlaceholders    map[string]map[string]string
	requestMappings         map[string]map[string]string
	envKeys                 map[string]map[string]struct{}
}

var instance = newLedger()

func newLedger() *Ledger {
	return &Ledger{
		transactions:            make(map[string]string),
		importResource:          make(map[string]map[string]string),
		configurationProperties: make(map[string]map[string]string),
		propertyPlaceholders:    make(map[string]map[string]string),
		regularProperties:       make(map[string]map[string]string),
		requestMappings:         make(map[string]map[string]string),
		envKeys:                 make(map[string]map[string]struct{}),
	}
}

func Get() *Ledger {
	return instance
}

func (l *Ledger) TransformRequestMapping(componentName, value, newValue string) {
	l.transform(l.requestMappings, componentName, value, newValue)
}

func (l *Ledger) TransformRegularProperty(componentName, value, newValue string) {
	l.transform(l.regularProperties, componentName, value, newValue)
}

func (l *Ledger) TransformConfigurationProperties(componentName, value, newValue string) {
	l.transform(l.configurationProperties, componentName, value, newValue)
}

func (l *Ledger) TransformPlaceholder(componentName, value, newValue string) {
	l.transform(l.propertyPlaceholders, componentName, value, newValue)
}

func (l *Ledger) TransformImportResource(componentName, value, newValue string) {
	l.transform(l.importResource, componentName, value, newValue)
}

func (l *Ledger) TransformTransaction(component, txManagerName string) {
	l.transactions[component] = txManagerName
}

func (l *Ledger) transform(m map[string]map[string]string, componentName, value, newValue string) {
	if _, exists := m[componentName]; !exists {
		m[componentName] = make(map[string]string)
	}
	m[componentName][value] = newValue
}

func (l *Ledger) RegisterEnvKey(componentName, key string) {
	if _, exists := l.envKeys[componentName]; !exists {
		l.envKeys[componentName] = make(map[string]struct{})
	}
	l.envKeys[componentName][key] = struct{}{}
}

func (l *Ledger) ShowSummary() {
	indent := strings.Repeat(" ", 4)

	l.printMapSection("Transformed Regular Properties:", indent, l.regularProperties)
	l.printMapSection("Transformed Property Placeholders:", indent, l.propertyPlaceholders)
	l.printMapSection("Transformed @ConfigurationProperties:", indent, l.configurationProperties)
	l.printMapSection("Transformed @RequestMapping:", indent, l.requestMappings)
	l.printMapSection("Transformed @ImportResource:", indent, l.importResource)
	l.printSection("Detected ENV keys referenced:", indent, l.envKeys)
	l.printSimpleMapSection("Transformed Transaction Manager:", indent, l.transactions)
}

func (l *Ledger) printMapSection(title, indent string, m map[string]map[string]string) {
	color.Cyan(title)
	components := sortedKeys(m)
	header := []string{"Component", "Old", "New"}
	var cellData [][]string
	for _, component := range components {
		keys := sortedKeys(m[component])
		for _, key := range keys {
			cellData = append(cellData, []string{component, key, util.Truncate(m[component][key], 60)})
		}
	}
	tablerender.DisplayTable(header, cellData, true, -1)
}

func (l *Ledger) printSection(title, indent string, m map[string]map[string]struct{}) {
	color.Cyan(title)
	components := sortedKeys(m)
	for _, component := range components {
		keys := sortedKeysFromSet(m[component])
		for _, key := range keys {
			log.Printf("%s%s: %s", indent, component, key)
		}
	}
}

func (l *Ledger) printSimpleMapSection(title, indent string, m map[string]string) {
	color.Cyan(title)
	keys := sortedKeys(m)
	for _, key := range keys {
		log.Printf("%s%s: %s", indent, key, m[key])
	}
}

func sortedKeys(m interface{}) []string {
	var keys []string
	switch v := m.(type) {
	case map[string]map[string]string:
		for k := range v {
			keys = append(keys, k)
		}
	case map[string]map[string]struct{}:
		for k := range v {
			keys = append(keys, k)
		}
	case map[string]string:
		for k := range v {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	return keys
}

func sortedKeysFromSet(m map[string]struct{}) []string {
	var keys []string
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
