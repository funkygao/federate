package microservice

import (
	"log"
	"os"
	"path/filepath"
	"strings"

	"federate/cmd/image"
	"federate/internal/fs"
	"federate/pkg/manifest"
	"federate/pkg/util"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var (
	cleanFlag bool

	runtimeClasses = []string{
		"FederatedAnnotationBeanNameGenerator",
		"FederatedMybatisConfig",
		"FederatedExcludedTypeFilter",
		"FederatedResourceLoader",

		// Inspect
		"FederatedIndirectRiskDetector",
		"RiskDetector",
		"RiskDetectorConditional",
		"RiskDetectorRequestMapping",

		// TODO 目前没有实质性意义
		"FederatedApplicationContextInitializer",
		"FederatedBeanDefinitionConflictProcessor",
		"FederatedEnvironmentPostProcessor",

		"package-info",
	}
)

var fusionStartCmd = &cobra.Command{
	Use:   "fusion-start",
	Short: "Scaffold a new fusion-starter project for the target system",
	Long: `The create command scaffolds a new fusion-starter project for the target system.
It provides runtime support for the target system.

Example usage:
  federate microservice fusion-start -i manifest.yaml`,
	Run: func(cmd *cobra.Command, args []string) {
		m := manifest.LoadManifest()
		if cleanFlag {
			cleanFusionStarterProject(m)
		} else {
			fusionStartProject(m)
		}
	},
}

func fusionStartProject(m *manifest.Manifest) {
	generatePomFile(m)
	generateMakefile(m)
	generateFederateRuntimeJavaClasses(m)
	generateJdosDockerfile(m)
	color.Green("🍺 %s-fusion-starter project scaffold generated for target: %s", m.Main.Name, m.Main.Name)
}

func generatePomFile(m *manifest.Manifest) {
	data := struct {
		Name                  string
		ComponentDependencies []manifest.DependencyInfo
		Dependencies          []manifest.DependencyInfo
	}{
		Name:                  m.Main.Name,
		ComponentDependencies: m.ComponentDependencies(),
		Dependencies:          m.Starter.Dependencies,
	}
	fn := filepath.Join(m.Dir, "pom.xml")
	overwrite := fs.GenerateFileFromTmpl("templates/fusion-starter/pom.xml", fn, data)
	if !overwrite {
		color.Cyan("Generated %s", fn)
	} else {
		color.Yellow("Overwrite %s", fn)
	}
}

func generateJdosDockerfile(m *manifest.Manifest) {
	fn := filepath.Join(m.Dir, "Dockerfile")
	overwrite := image.GenerateJdosDockerfile(m, fn)
	if !overwrite {
		color.Cyan("Generated %s", fn)
	} else {
		color.Yellow("Overwrite %s", fn)
	}
}

func generateMakefile(m *manifest.Manifest) {
	data := struct {
		AppName string
	}{
		AppName: m.Main.Name,
	}
	fn := filepath.Join(m.Dir, "Makefile")
	overwrite := fs.GenerateFileFromTmpl("templates/fusion-starter/Makefile", fn, data)
	if !overwrite {
		color.Cyan("Generated %s", fn)
	} else {
		color.Yellow("Overwrite %s", fn)
	}
}

func generateFederateRuntimeJavaClasses(m *manifest.Manifest) {
	for _, cls := range runtimeClasses {
		generateJava(m, cls)
	}
}

func generateJava(m *manifest.Manifest, simpleClassName string) {
	packageName := m.Main.FederatedRuntimePackage()
	data := struct {
		Package               string
		MapperScanBasePackage string
		SingletonClasses      []string
		AddOns                []string
		ExcludedBeanPatterns  []string
	}{
		Package:               packageName,
		MapperScanBasePackage: "com.jdwl.wms", // TODO
		SingletonClasses:      m.Main.Runtime.SingletonComponents,
		AddOns:                m.Starter.Inspect.AddOn,
		ExcludedBeanPatterns:  m.Starter.BeanNameGenerator.ExcludedBeanPatterns,
	}
	mainClassDir := filepath.Join(m.Dir, "src", "main", "java", filepath.FromSlash(strings.ReplaceAll(packageName, ".", "/")))
	javaFile := filepath.Join(mainClassDir, simpleClassName+".java")
	overwrite := fs.GenerateFileFromTmpl("templates/fusion-starter/"+simpleClassName+".java", javaFile, data)
	if !overwrite {
		color.Cyan("Generated %s", javaFile)
	} else {
		color.Yellow("Overwrite %s", javaFile)
	}
}

func cleanFusionStarterProject(m *manifest.Manifest) {
	filesToRemove := []string{
		filepath.Join(m.Dir, "pom.xml"),
		filepath.Join(m.Dir, "Makefile"),
	}

	packagePath := filepath.Join(m.Dir, "src", "main", "java", filepath.FromSlash(strings.ReplaceAll(m.Main.FederatedRuntimePackage(), ".", "/")))
	for _, cls := range runtimeClasses {
		filesToRemove = append(filesToRemove, filepath.Join(packagePath, cls+".java"))
	}

	n := 0
	for _, file := range filesToRemove {
		if util.FileExists(file) {
			os.Remove(file)
			log.Printf("Removed %s", file)
			n++
		}
	}

	if n > 0 {
		color.Green("🧹 Cleaned up %s-fusion-starter project files", m.Main.Name)
	}
}

func init() {
	manifest.RequiredManifestFileFlag(fusionStartCmd)
	fusionStartCmd.Flags().BoolVarP(&cleanFlag, "clean", "c", false, "Remove all generated files")
}
