package main

import (
	"federate/pkg/manifest"
	"federate/pkg/merge"
)

var (
	// This is the exported symbol that the plugin loader will look for: plugin.Lookup("Reconciler")
	Reconciler merge.PluginReconciler = &DemoReconciler{}
)

type DemoReconciler struct {
	manifest *manifest.Manifest
}

func (r *DemoReconciler) Name() string {
	return "Demo"
}

func (r *DemoReconciler) Reconcile() error {
	return nil
}

func (r *DemoReconciler) Init(m *manifest.Manifest) error {
	r.manifest = m
	return nil
}

// This main function is required for building the plugin
func main() {}
