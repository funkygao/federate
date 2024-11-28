package transformer

import (
	"log"
	"sort"
	"strings"

	"federate/pkg/tablerender"
	"federate/pkg/util"
	"github.com/fatih/color"
)

type Transformer struct {
	transactions            map[string]string
	importResource          map[string]map[string]string
	configurationProperties map[string]map[string]string
	regularProperties       map[string]map[string]string
	requestMappings         map[string]map[string]string
	envKeys                 map[string]map[string]struct{}
}

var tf = &Transformer{
	transactions:            make(map[string]string),
	importResource:          make(map[string]map[string]string),
	configurationProperties: make(map[string]map[string]string),
	regularProperties:       make(map[string]map[string]string),
	requestMappings:         make(map[string]map[string]string),
	envKeys:                 make(map[string]map[string]struct{}),
}

func Get() *Transformer {
	return tf
}

func (t *Transformer) TransformRequestMapping(componentName, value, newValue string) {
	t.transform(t.requestMappings, componentName, value, newValue)
}

func (t *Transformer) TransformRegularProperty(componentName, value, newValue string) {
	t.transform(t.regularProperties, componentName, value, newValue)
}

func (t *Transformer) TransformConfigurationProperties(componentName, value, newValue string) {
	t.transform(t.configurationProperties, componentName, value, newValue)
}

func (t *Transformer) TransformImportResource(componentName, value, newValue string) {
	t.transform(t.importResource, componentName, value, newValue)
}

func (t *Transformer) TransformTransaction(component, txManagerName string) {
	t.transactions[component] = txManagerName
}

func (t *Transformer) transform(m map[string]map[string]string, componentName, value, newValue string) {
	if _, exists := m[componentName]; !exists {
		m[componentName] = make(map[string]string)
	}
	m[componentName][value] = newValue
}

func (t *Transformer) RegisterEnvKey(componentName, key string) {
	if _, exists := t.envKeys[componentName]; !exists {
		t.envKeys[componentName] = make(map[string]struct{})
	}
	t.envKeys[componentName][key] = struct{}{}
}

func (t *Transformer) ShowSummary() {
	indent := strings.Repeat(" ", 4)

	t.printMapSection("Transformed Regular Properties:", indent, t.regularProperties)
	t.printMapSection("Transformed @ConfigurationProperties:", indent, t.configurationProperties)
	t.printMapSection("Transformed @RequestMapping:", indent, t.requestMappings)
	t.printMapSection("Transformed @ImportResource:", indent, t.importResource)
	t.printSection("Detected ENV keys referenced:", indent, t.envKeys)
	t.printSimpleMapSection("Transformed Transaction Manager:", indent, t.transactions)
}

func (t *Transformer) printMapSection(title, indent string, m map[string]map[string]string) {
	color.Cyan(title)
	components := sortedKeys(m)
	header := []string{"Component", "Old", "New"}
	var cellData [][]string
	for _, component := range components {
		keys := sortedKeys(m[component])
		for _, key := range keys {
			cellData = append(cellData, []string{component, key, util.Truncate(m[component][key], 40)})
		}
	}
	tablerender.DisplayTable(header, cellData, true, -1)
}

func (t *Transformer) printSection(title, indent string, m map[string]map[string]struct{}) {
	color.Cyan(title)
	components := sortedKeys(m)
	for _, component := range components {
		keys := sortedKeysFromSet(m[component])
		for _, key := range keys {
			log.Printf("%s%s: %s", indent, component, key)
		}
	}
}

func (t *Transformer) printSimpleMapSection(title, indent string, m map[string]string) {
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
