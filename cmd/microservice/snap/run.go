package snap

import (
	"log"
	"path/filepath"

	"federate/pkg/manifest"
)

var hinted = false

func runSnap(m *manifest.Manifest) {
	createLocalMavenRepo()

	for _, component := range m.Components {
		log.Printf("Component: %s", component.RootDir())

		rootPom := filepath.Join(component.RootDir(), "pom.xml")
		processPom(rootPom, 1)

		for _, module := range component.MavenModules() {
			processModule(module, 1)
		}

		// Additional processing for the component
		removeUnnecessaryFiles(component.RootDir(), 1)
		removeSensitiveInformation(component.RootDir(), 1)
		updateConfigurations(component.RootDir(), 1)

		hinted = true
	}

	updatePomFilesForLocalRepo(m)
	copyDependenciesToLocalRepo()

	log.Println("Snapshot creation complete. Please review changes with 'git diff'.")
	log.Println("IMPORTANT: Perform a final code review to ensure all sensitive information has been removed and the code is appropriate for customer delivery.")
	finalChecks(0)
}
