package merge

import (
	"log"

	"federate/pkg/manifest"
	"federate/pkg/merge"
	"github.com/fatih/color"
)

func recursiveCopyResources(m *manifest.Manifest, resourceManager *merge.ResourceManager) {
	if err := resourceManager.RecursiveCopyResources(m); err != nil {
		log.Fatalf("Error copying resources: %v", err)
	}

	color.Green("🍺 Resources recursively copied")
}

func recursiveFlatCopyResources(m *manifest.Manifest, resourceManager *merge.ResourceManager) {
	if err := resourceManager.RecursiveFlatCopyResources(m); err != nil {
		log.Fatalf("Error merging reconcile.flatCopyResources: %v", err)
	}
	color.Green("🍺 Resources recursively flat copied")
}
