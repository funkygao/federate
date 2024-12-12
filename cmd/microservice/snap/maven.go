package snap

import (
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"federate/pkg/manifest"
	"federate/pkg/step"
	"github.com/fatih/color"
)

func prepareMavenRepo(m *manifest.Manifest) {
	currentDir, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get current working directory: %v", err)
	}

	// 设置 localRepoPath 为绝对路径
	localRepoPath = filepath.Join(currentDir, "generated", "artifactory")

	steps := []step.Step{
		{
			Name: "Create local Maven repository",
			Fn:   createLocalMavenRepo,
		},
		{
			Name: "Update component pom.xml to use the local Maven repository",
			Fn:   func(bar step.Bar) { updatePomFilesForLocalRepo(m) },
		},
		{
			Name: "Copy dependencies to local Maven repository",
			Fn:   func(bar step.Bar) { copyDependenciesToLocalRepo(m) },
		},
		{
			Name: "Detect version conflicts in local Maven repository",
			Fn:   detectVersionConflicts,
		},
		{
			Name: "Organize the local Maven repository",
			Fn:   organizeLocalRepo,
		},
	}

	step.Run(steps)
}

func createLocalMavenRepo(bar step.Bar) {
	err := os.MkdirAll(localRepoPath, 0755)
	if err != nil {
		log.Fatalf("Failed to create local Maven repository: %v", err)
	}
	log.Printf("🍺 Local Maven repository created at: %s", localRepoPath)
}

func copyDependenciesToLocalRepo(m *manifest.Manifest) {
	for _, component := range m.Components {
		log.Printf("Processing dependencies for component: %s", component.Name)

		// Change to component directory
		err := os.Chdir(component.RootDir())
		if err != nil {
			log.Fatalf("Failed to change directory to %s: %v", component.RootDir(), err)
		}

		// 构建 Maven 命令
		mvnArgs := []string{
			"dependency:copy-dependencies",
			"-DoutputDirectory=" + localRepoPath,
			"-DincludeScope=runtime",
			"-q", // quiet 模式
		}

		// Run Maven command
		cmd := exec.Command("mvn", mvnArgs...)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		color.Blue(strings.Join(cmd.Args, " "))
		err = cmd.Run()
		if err != nil {
			log.Fatalf("Failed to copy dependencies for %s: %v", component.Name, err)
		}

		// Change back to original directory
		err = os.Chdir("..")
		if err != nil {
			log.Fatalf("Failed to change back to original directory: %v", err)
		}
	}

	log.Println("🍺 All dependencies copied to local Maven repository")
}

func detectVersionConflicts(bar step.Bar) {
	artifactVersions := make(map[string][]string)

	err := filepath.Walk(localRepoPath, func(path string, info os.FileInfo, err error) error {
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

			artifactVersions[artifactId] = append(artifactVersions[artifactId], version)
		}
		return nil
	})

	if err != nil {
		log.Fatalf("Failed to walk local repository: %v", err)
	}

	for artifact, versions := range artifactVersions {
		if len(versions) > 1 {
			color.Yellow("Version conflict detected for %s: %v", artifact, versions)
		}
	}
}

func organizeLocalRepo(bar step.Bar) {
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
