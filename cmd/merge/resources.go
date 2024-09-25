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

func recursiveMergeResources(m *manifest.Manifest, resourceManager *merge.ResourceManager) {
	if err := resourceManager.RecursiveMergeResources(m); err != nil {
		log.Fatalf("Error merging reconcile.mergeResources: %v", err)
	}
	color.Green("🍺 Resources recursively merged")
}
