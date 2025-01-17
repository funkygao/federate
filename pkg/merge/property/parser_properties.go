package property

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"federate/pkg/manifest"
)

type propertiesParser struct{}

func (p *propertiesParser) Parse(filePath string, component manifest.ComponentInfo, pm *PropertyManager) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	if !pm.silent || pm.debug {
		log.Printf("[%s] Parsing %s", component.Name, filePath)
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

			pm.r.AddProperty(component, key, value, filePath)
		}
	}

	if err = scanner.Err(); err != nil {
		return err
	}

	return nil
}

func (p *propertiesParser) Generate(entries map[string]PropertyEntry, targetFile string) error {
	var builder strings.Builder

	for key, entry := range entries {
		builder.WriteString(fmt.Sprintf("%s=%v\n", key, entry.Value))
	}

	if err := os.MkdirAll(filepath.Dir(targetFile), 0755); err != nil {
		return fmt.Errorf("error creating directory for merged properties: %v", err)
	}

	if err := os.WriteFile(targetFile, []byte(builder.String()), 0644); err != nil {
		return fmt.Errorf("error writing merged properties to %s: %v", targetFile, err)
	}

	return nil
}
