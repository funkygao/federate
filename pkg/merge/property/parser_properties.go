package property

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"federate/pkg/ds"
	"federate/pkg/manifest"
)

type propertiesParser struct{}

func (p *propertiesParser) Parse(filePath string, component manifest.ComponentInfo, cm *PropertyManager) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	if !cm.silent || cm.debug {
		log.Printf("[%s] Processing %s", component.Name, filePath)
	}

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			value := strings.TrimSpace(parts[1])

			cm.r.AddProperty(component, key, value, filePath)
		}
	}

	if err = scanner.Err(); err != nil {
		return err
	}

	return nil
}

func (p *propertiesParser) Generate(entries map[string]PropertyEntry, rawKeys []string, targetFile string) error {
	var builder strings.Builder
	processedKeys := ds.NewStringSet()

	for key, entry := range entries {
		if !processedKeys.Contains(key) {
			builder.WriteString(fmt.Sprintf("%s=%v\n", key, entry.Value))
			processedKeys.Add(key)
		}
	}

	if err := os.MkdirAll(filepath.Dir(targetFile), 0755); err != nil {
		return fmt.Errorf("error creating directory for merged properties: %v", err)
	}

	if err := os.WriteFile(targetFile, []byte(builder.String()), 0644); err != nil {
		return fmt.Errorf("error writing merged properties to %s: %v", targetFile, err)
	}

	return nil
}
