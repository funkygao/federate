package snap

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"federate/pkg/manifest"
	"github.com/beevik/etree"
)

func createLocalMavenRepo() {
	err := os.MkdirAll(localRepoPath, 0755)
	if err != nil {
		log.Fatalf("Failed to create local Maven repository: %v", err)
	}
	log.Printf("🍺 Local Maven repository created at: %s", localRepoPath)
}

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

func copyDependenciesToLocalRepo(m *manifest.Manifest) {
	for _, component := range m.Components {
		log.Printf("Processing dependencies for component: %s", component.Name)

		// Change to component directory
		err := os.Chdir(component.RootDir())
		if err != nil {
			log.Fatalf("Failed to change directory to %s: %v", component.RootDir(), err)
		}

		// Run Maven command
		cmd := exec.Command("mvn", "dependency:copy-dependencies", "-DoutputDirectory="+localRepoPath, "-DincludeScope=runtime")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		err = cmd.Run()
		if err != nil {
			log.Fatalf("Failed to copy dependencies for %s: %v", component.Name, err)
		}

		log.Printf("🍺 Dependencies copied for component: %s", component.Name)

		// Change back to original directory
		err = os.Chdir("..")
		if err != nil {
			log.Fatalf("Failed to change back to original directory: %v", err)
		}
	}

	// Organize the local repository structure
	organizeLocalRepo()

	log.Println("🍺 All dependencies copied and organized in local Maven repository")
}

func organizeLocalRepo() {
	err := filepath.Walk(localRepoPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && filepath.Ext(path) == ".jar" {
			// Extract groupId, artifactId, and version from file name
			fileName := filepath.Base(path)
			parts := strings.Split(fileName, "-")
			if len(parts) < 2 {
				return nil // Skip files that don't match expected format
			}
			version := parts[len(parts)-1]
			version = strings.TrimSuffix(version, ".jar")
			artifactId := strings.Join(parts[:len(parts)-1], "-")

			// Create directory structure
			newDir := filepath.Join(localRepoPath, strings.Replace(artifactId, ".", "/", -1), version)
			err = os.MkdirAll(newDir, 0755)
			if err != nil {
				return err
			}

			// Move JAR file
			newPath := filepath.Join(newDir, fileName)
			err = os.Rename(path, newPath)
			if err != nil {
				return err
			}
		}
		return nil
	})

	if err != nil {
		log.Fatalf("Failed to organize local repository: %v", err)
	}

	log.Printf("🍺 Local Maven repository organized at: %s", localRepoPath)
}
