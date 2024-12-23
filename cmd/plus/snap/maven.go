package snap

import (
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"federate/pkg/manifest"
	"federate/pkg/primitive"
	"federate/pkg/step"
	"github.com/fatih/color"
)

func createLocalMavenRepo(bar step.Bar) {
	err := os.MkdirAll(absLocalRepoPath, 0755)
	if err != nil {
		log.Fatalf("Failed to create local Maven repository: %v", err)
	}
}

func copyDependenciesToLocalRepo(m *manifest.Manifest, bar step.Bar) {
	bar.ChangeMax(len(m.Components))
	for _, component := range m.Components {

		// Change to component directory
		err := os.Chdir(component.RootDir())
		if err != nil {
			log.Fatalf("Failed to change directory to %s: %v", component.RootDir(), err)
		}

		// 构建 Maven 命令
		mvnArgs := []string{
			"dependency:copy-dependencies",
			"-DoutputDirectory=" + absLocalRepoPath,
			"-DincludeScope=runtime",
			"-q", // quiet 模式
		}

		// Run Maven command
		cmd := exec.Command("mvn", mvnArgs...)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		color.Cyan(strings.Join(cmd.Args, " "))
		err = cmd.Run()
		if err != nil {
			log.Fatalf("Failed to copy dependencies for %s: %v", component.Name, err)
		}

		// Change back to original directory
		err = os.Chdir("..")
		if err != nil {
			log.Fatalf("Failed to change back to original directory: %v", err)
		}

		bar.Add(1)
	}
}

func detectVersionConflicts(bar step.Bar) {
	artifactVersions := make(map[string]*primitive.StringSet)

	err := filepath.Walk(absLocalRepoPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(info.Name(), ".jar") {
			parts := strings.Split(info.Name(), "-")
			if len(parts) < 2 {
				return nil
			}
			version := parts[len(parts)-1]
			version = strings.TrimSuffix(version, ".jar")
			artifactId := strings.Join(parts[:len(parts)-1], "-")
			if _, ok := artifactVersions[artifactId]; !ok {
				artifactVersions[artifactId] = primitive.NewStringSet()
			}

			artifactVersions[artifactId].Add(version)
		}
		return nil
	})

	if err != nil {
		log.Fatalf("Failed to walk local repository: %v", err)
	}

	for artifact, versions := range artifactVersions {
		if versions.Cardinality() > 1 {
			color.Yellow("Version conflict detected for %s: %v", artifact, versions.Values())
		}
	}
}

func organizeLocalRepo(bar step.Bar) {
	err := filepath.Walk(absLocalRepoPath, func(path string, info os.FileInfo, err error) error {
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
			newDir := filepath.Join(absLocalRepoPath, strings.Replace(artifactId, ".", "/", -1), version)
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
}
