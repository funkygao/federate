package merge

import (
	"fmt"
	"path/filepath"
	"plugin"

	"federate/pkg/manifest"
)

type PluginReconciler interface {
	Reconciler
	Init(m *manifest.Manifest) error
}

func LoadPluginReconcilers(pluginDir string, m *manifest.Manifest) ([]Reconciler, error) {
	var reconcilers []Reconciler

	plugins, err := filepath.Glob(filepath.Join(pluginDir, "*.so"))
	if err != nil {
		return nil, fmt.Errorf("error loading plugins: %v", err)
	}

	for _, pluginPath := range plugins {
		p, err := plugin.Open(pluginPath)
		if err != nil {
			return nil, fmt.Errorf("error opening plugin %s: %v", pluginPath, err)
		}

		symReconciler, err := p.Lookup("Reconciler")
		if err != nil {
			return nil, fmt.Errorf("Reconciler symbol not found in plugin %s: %v", pluginPath, err)
		}

		reconciler, ok := symReconciler.(PluginReconciler)
		if !ok {
			return nil, fmt.Errorf("invalid Reconciler type in plugin %s", pluginPath)
		}

		if err := reconciler.Init(m); err != nil {
			return nil, fmt.Errorf("error initializing plugin %s: %v", pluginPath, err)
		}

		reconcilers = append(reconcilers, reconciler)
	}

	return reconcilers, nil
}
