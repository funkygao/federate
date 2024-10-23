package merge

import (
	"log"

	"federate/pkg/manifest"
	"federate/pkg/merge"
	"github.com/fatih/color"
)

func recursiveFederatedCopyResources(m *manifest.Manifest, resourceManager *merge.ResourceManager) {
	if err := resourceManager.RecursiveFederatedCopyResources(m); err != nil {
		log.Fatalf("Error copying resources: %v", err)
	}

	color.Green("🍺 Resources recursively federated copied")
}

func recursiveFlatCopyResources(m *manifest.Manifest, resourceManager *merge.ResourceManager) {
	if err := resourceManager.RecursiveFlatCopyResources(m); err != nil {
		log.Fatalf("Error merging reconcile.flatCopyResources: %v", err)
	}
	color.Green("🍺 Resources recursively flat copied")
}
