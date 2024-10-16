package snap

import (
	"log"
	"path/filepath"

	"federate/pkg/manifest"
)

func runSnap(m *manifest.Manifest) {
	for _, component := range m.Components {
		log.Printf("Processing %s", component.RootDir())

		rootPom := filepath.Join(component.RootDir(), "pom.xml")
		processPom(rootPom, 1)

		for _, module := range component.MavenModules() {
			processModule(module, 1)
			break
		}

		// Additional processing for the component
		removeUnnecessaryFiles(component.RootDir(), 2)
		removeSensitiveInformation(component.RootDir(), 2)
		updateConfigurations(component.RootDir(), 2)
		break
	}

	finalChecks(0)

	log.Println()
	log.Println("Snapshot creation complete. Please review changes with 'git diff'.")
	log.Println("IMPORTANT: Perform a final code review to ensure all sensitive information has been removed and the code is appropriate for customer delivery.")
}
