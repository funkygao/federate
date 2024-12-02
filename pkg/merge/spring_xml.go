package merge

import (
	"os"
	"path/filepath"

	"federate/internal/fs"
	"federate/pkg/federated"
	"federate/pkg/manifest"
	"github.com/fatih/color"
)

type springXmlMerger struct {
	m *manifest.Manifest
}

func NewSpringXmlMerger(m *manifest.Manifest) Reconciler {
	return &springXmlMerger{m: m}
}

func (m *springXmlMerger) Name() string {
	return "Generating Federated Spring Bootstrap XML"
}

func (m *springXmlMerger) Reconcile() error {
	targetDir := federated.GeneratedResourceBaseDir(m.m.Main.Name)
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return err
	}

	targetFile := filepath.Join(targetDir, "spring.xml")
	fs.GenerateFileFromTmpl("templates/spring.xml", targetFile, m)

	color.Green("🍺 Generated %s", targetFile)
	return nil
}
