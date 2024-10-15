package microservice

import (
	"path/filepath"
	"strings"

	"federate/internal/fs"
	"federate/pkg/manifest"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
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
		scaffoldProject(m)
	},
}

func scaffoldProject(m *manifest.Manifest) {
	generatePomFile(m)
	generateMakefile(m)
	generateFederateRuntimeJavaClasses(m)
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
	runtimeClasses := []string{
		"FederatedAnnotationBeanNameGenerator",
		"FederatedApplicationContextInitializer",
		"FederatedBeanDefinitionConflictProcessor",
		"FederatedDefaultBeanNameGenerator",
		"FederatedMybatisConfig",
		"FederatedEnvironmentPostProcessor",
		"FederatedExcludedTypeFilter",
		"FederatedResourceLoader",

		"FederatedIndirectRiskDetector",
		"RiskDetector",
		"RiskDetectorConditional",
		"RiskDetectorRequestMapping",
	}

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
	}{
		Package:               packageName,
		MapperScanBasePackage: "com.jdwl.wms", // TODO
		SingletonClasses:      m.Main.Runtime.SingletonComponents,
		AddOns:                m.Starter.Inspect.AddOn,
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

func init() {
	manifest.RequiredManifestFileFlag(fusionStartCmd)
}
