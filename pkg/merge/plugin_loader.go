package merge

import (
	"fmt"
	"path/filepath"
	"plugin"

	"federate/pkg/manifest"
)

// Load PluginReconcilers from specified dir.
func LoadPluginReconcilers(m *manifest.Manifest) (reconcilers []Reconciler, err error) {
	plugins, err := filepath.Glob(filepath.Join(m.Main.Reconcile.PluginDir, "*.so"))
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

		// 执行初始化，传入 manifest
		if err := reconciler.Init(m); err != nil {
			return nil, fmt.Errorf("error initializing plugin %s: %v", pluginPath, err)
		}

		reconcilers = append(reconcilers, reconciler)
	}

	return
}
