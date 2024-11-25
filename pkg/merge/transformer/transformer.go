package transformer

import (
	"log"
	"strings"
)

func Get() *Transformer {
	return tf
}

var tf = &Transformer{
	transactions:            make(map[string]string),
	configurationProperties: make(map[string]string),
	importResource:          make(map[string]string),
	envKeys:                 make([]string, 0),
}

type Transformer struct {
	transactions            map[string]string // component: txManager
	configurationProperties map[string]string
	importResource          map[string]string
	envKeys                 []string
}

func (t *Transformer) TransformConfigurationProperties(value, newValue string) {
	t.configurationProperties[value] = newValue
}

func (t *Transformer) TransformImportResource(value, newValue string) {
	t.importResource[value] = newValue
}

func (t *Transformer) TransformTransaction(component, txManagerName string) {
	t.transactions[component] = txManagerName
}

func (t *Transformer) RegisterEnvKey(key string) {
	t.envKeys = append(t.envKeys, key)
}

func (t *Transformer) ShowSummary() {
	indent := strings.Repeat(" ", 2)

	log.Printf("System.getProperty keys:")
	for _, key := range t.envKeys {
		log.Printf("%s%s", indent, key)
	}

	log.Printf("@ConfigurationProperties:")
	for key, value := range t.configurationProperties {
		log.Printf("%s%s => %s", indent, key, value)
	}

	log.Printf("@RequestMapping:")

	log.Printf("@ImportResource:")
	for key, value := range t.importResource {
		log.Printf("%s%s => %s", indent, key, value)
	}

	log.Printf("Transaction Manager Transformed:")
	for key, value := range t.transactions {
		log.Printf("%s%s => %s", indent, key, value)
	}

}
