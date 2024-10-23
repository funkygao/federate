package merge

import (
	"log"
	"os"
	"path/filepath"

	"federate/internal/fs"
	"federate/pkg/federated"
	"federate/pkg/manifest"
	"github.com/fatih/color"
)

func generateSpringBootstrapXML(m *manifest.Manifest) {
	targetDir := federated.GeneratedResourceBaseDir(m.Main.Name)
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		log.Fatalf("Error creating directory: %v", err)
	}

	targetFile := filepath.Join(targetDir, targetSpringXml)
	fs.GenerateFileFromTmpl("templates/spring.xml", targetFile, m)

	color.Green("🍺 Generated %s", targetFile)
}
