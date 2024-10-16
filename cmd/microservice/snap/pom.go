package snap

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"federate/pkg/manifest"
	"github.com/beevik/etree"
)

func updatePomFilesForLocalRepo(m *manifest.Manifest) {
	for _, component := range m.Components {
		rootPom := filepath.Join(component.RootDir(), "pom.xml")
		if err := updatePom(rootPom); err != nil {
			log.Fatalf("%s: %v", rootPom, err)
		}

		for _, module := range component.ChildDirs() {
			pomPath := filepath.Join(component.RootDir(), module, "pom.xml")
			if _, err := os.Stat(pomPath); os.IsNotExist(err) {
				continue
			}
			if err := updatePom(pomPath); err != nil {
				log.Fatalf("%s: %v", pomPath, err)
			}
		}
	}
	log.Println("🍺 All pom.xml files updated to use local Maven repository")
}

func updatePom(pomPath string) error {
	doc := etree.NewDocument()
	if err := doc.ReadFromFile(pomPath); err != nil {
		return err
	}

	project := doc.SelectElement("project")
	if project == nil {
		log.Printf("Warning: %s is not a valid pom.xml", pomPath)
		return nil
	}

	// Add or update repositories section
	repositories := project.SelectElement("repositories")
	if repositories == nil {
		repositories = project.CreateElement("repositories")
	}

	// Check if local repository already exists
	localRepoExists := false
	for _, repo := range repositories.SelectElements("repository") {
		if id := repo.SelectElement("id"); id != nil && id.Text() == "local-maven-repo" {
			localRepoExists = true
			break
		}
	}

	if !localRepoExists {
		repo := repositories.CreateElement("repository")
		repo.CreateElement("id").SetText("local-maven-repo")
		repo.CreateElement("url").SetText(fmt.Sprintf("file://${project.basedir}/%s", filepath.Base(localRepoPath)))
		log.Printf("Added local Maven repository reference to %s", pomPath)
	}

	// Remove internal repository references
	for _, repo := range repositories.SelectElements("repository") {
		if id := repo.SelectElement("id"); id != nil && (id.Text() == "internal-repo" || id.Text() == "company-repo") {
			repositories.RemoveChild(repo)
			log.Printf("Removed internal repository reference from %s", pomPath)
		}
	}

	doc.Indent(4)

	// Save the updated pom.xml
	f, err := os.Create(pomPath)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = doc.WriteTo(f)
	return err
}
