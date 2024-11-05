package microservice

import (
	"log"
	"os"
	"path/filepath"

	"federate/internal/fs"
	"federate/pkg/java"
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
		m := manifest.Load()
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
	color.Green("🍺 %s-starter project scaffold generated for: %s", m.Main.Name, m.Main.Name)
}

func generatePomFile(m *manifest.Manifest) {
	data := struct {
		Name                  string
		ComponentDependencies []java.DependencyInfo
		Dependencies          []java.DependencyInfo
		GroupId               string
	}{
		Name:                  m.Main.Name,
		ComponentDependencies: m.ComponentModules(),
		Dependencies:          m.Starter.Dependencies,
		GroupId:               m.Main.GroupId,
	}
	fn := filepath.Join(m.StarterBaseDir(), "pom.xml")
	overwrite := fs.GenerateFileFromTmpl("templates/fusion-starter/pom.xml", fn, data)
	if !overwrite {
		log.Printf("Generated %s", fn)
	} else {
		log.Printf("Overwrite %s", fn)
	}
}

func generateMakefile(m *manifest.Manifest) {
	data := struct {
		AppName string
	}{
		AppName: m.Main.Name,
	}
	fn := filepath.Join(m.StarterBaseDir(), "Makefile")
	overwrite := fs.GenerateFileFromTmpl("templates/fusion-starter/Makefile", fn, data)
	if !overwrite {
		log.Printf("Generated %s", fn)
	} else {
		log.Printf("Overwrite %s", fn)
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
	mainClassDir := filepath.Join(m.StarterBaseDir(), "src", "main", "java", java.Pkg2Path(packageName))
	javaFile := filepath.Join(mainClassDir, simpleClassName+".java")
	overwrite := fs.GenerateFileFromTmpl("templates/fusion-starter/"+simpleClassName+".java", javaFile, data)
	if !overwrite {
		log.Printf("Generated %s", javaFile)
	} else {
		log.Printf("Overwrite %s", javaFile)
	}
}

func cleanFusionStarterProject(m *manifest.Manifest) {
	filesToRemove := []string{
		filepath.Join(m.StarterBaseDir(), "pom.xml"),
		filepath.Join(m.StarterBaseDir(), "Makefile"),
	}

	packagePath := filepath.Join(m.StarterBaseDir(), "src", "main", "java", java.Pkg2Path(m.Main.FederatedRuntimePackage()))
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
