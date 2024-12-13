package ledger

import (
	"encoding/json"
	"log"
	"os"
	"sort"
	"strings"

	"federate/pkg/tabular"
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

func (l *Ledger) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Guide                   string                         `json:"帮助,omitempty"`
		RegularProperties       map[string]map[string]string   `json:"属性Key变化"`
		PropertyPlaceholders    map[string]map[string]string   `json:"属性的引用值变化"`
		EnvKeys                 map[string]map[string]struct{} `json:"目前使用的环境变量"`
		Transactions            map[string]string              `json:"@Transactional 相关源代码改动"`
		ImportResource          map[string]map[string]string   `json:"@ImportResource 相关源代码改动"`
		ConfigurationProperties map[string]map[string]string   `json:"@ConfigurationProperties 相关源代码改动"`
		RequestMappings         map[string]map[string]string   `json:"@RequestMapping 相关源代码改动"`
	}{
		Guide:                   "汇总代码和资源文件变更：按照模块分组，左侧是旧值，右侧是新值",
		Transactions:            l.transactions,
		ImportResource:          l.importResource,
		ConfigurationProperties: l.configurationProperties,
		RegularProperties:       l.regularProperties,
		PropertyPlaceholders:    l.propertyPlaceholders,
		RequestMappings:         l.requestMappings,
		EnvKeys:                 l.envKeys,
	})
}

func (l *Ledger) SaveToFile(filename string) error {
	data, err := json.MarshalIndent(l, "", "    ")
	if err != nil {
		return err
	}

	color.Cyan("Summary Writen to: %s", filename)
	return os.WriteFile(filename, data, 0644)
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
	if len(m) == 0 {
		return
	}

	components := sortedKeys(m)
	header := []string{"Component", "Old", "New"}
	var cellData [][]string
	n := 0
	for _, component := range components {
		keys := sortedKeys(m[component])
		n += len(keys)
		for _, key := range keys {
			cellData = append(cellData, []string{component, key, util.Truncate(m[component][key], 60)})
		}
	}
	color.Yellow("%s: %d", title, n)
	tabular.Display(header, cellData, true, -1)
}

func (l *Ledger) printSection(title, indent string, m map[string]map[string]struct{}) {
	if len(m) == 0 {
		return
	}

	color.Yellow("%s %d", title, len(m))
	components := sortedKeys(m)
	for _, component := range components {
		keys := sortedKeysFromSet(m[component])
		for _, key := range keys {
			log.Printf("%s%s: %s", indent, component, key)
		}
	}
}

func (l *Ledger) printSimpleMapSection(title, indent string, m map[string]string) {
	if len(m) == 0 {
		return
	}

	color.Yellow("%s %d", title, len(m))
	keys := sortedKeys(m)
	for _, key := range keys {
		log.Printf("%s%s: %s", indent, key, m[key])
	}
}

func sortedKeys(m any) []string {
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
