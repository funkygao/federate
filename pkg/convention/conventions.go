package convention

import (
	"embed"
	"log"

	"gopkg.in/yaml.v2"
)

type Conventions struct {
	Items map[string]map[string]string `yaml:"contents"`
}

// GetAllConventions returns the singleton instance of Conventions
func GetAll(f embed.FS, fn string) *Conventions {
	data, err := f.ReadFile(fn)
	if err != nil {
		log.Fatalf("error reading embedded file: %v", err)
	}

	var allConventions Conventions
	err = yaml.Unmarshal(data, &allConventions)
	if err != nil {
		log.Fatalf("error unmarshalling yaml: %v", err)
	}
	return &allConventions
}

// GetKeys returns all keys for a given kind
func (c *Conventions) GetKeys(kind string) []string {
	keys := make([]string, 0, len(c.Items[kind]))
	for key := range c.Items[kind] {
		keys = append(keys, key)
	}
	return keys
}

// GetExample returns the example value for a given kind and key
func (c *Conventions) GetExample(kind, key string) string {
	return c.Items[kind][key]
}

// Kinds returns all kinds
func (c *Conventions) Kinds() []string {
	kinds := make([]string, 0, len(c.Items))
	for kind := range c.Items {
		kinds = append(kinds, kind)
	}
	return kinds
}
