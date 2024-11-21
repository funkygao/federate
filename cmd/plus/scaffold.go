package plus

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"federate/cmd/microservice/merge"
	"federate/internal/fs"
	"federate/pkg/git"
	"federate/pkg/java"
	"federate/pkg/manifest"
	"federate/pkg/util"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var scaffoldCmd = &cobra.Command{
	Use:   "scaffold",
	Short: "Generate a WMS6 Plus Project with standard structure and boilerplate code",
	Run: func(cmd *cobra.Command, args []string) {
		m := manifest.Load()
		validatePlusSpec(m.Main.Plus)
		doCreate(m)
	},
}

func doCreate(m *manifest.Manifest) {
	// 添加 git submodules
	if err := git.AddSubmodules(m); err != nil {
		log.Fatalf("Error adding git submodules: %v", err)
	}

	// 脚手架
	log.Printf("Scaffolding %s project structure ...", m.Main.Name)
	generatePlusProjectFiles(m)

	// 让WMS6.0代码安装后可以被依赖
	log.Println("Instrumenting submodule pom.xml for JAR dependency ...")
	merge.InstrumentPomForFederatePackaging(m)

	color.Green("🍺 Congrat, %s scaffolded!", m.Main.Name)
}

func generatePlusProjectFiles(m *manifest.Manifest) {
	basePackage := m.Main.PlusBasePackage()
	data := struct {
		ArtifactId            string
		ComponentDependencies []java.DependencyInfo
		BasePackage           string
		BasePackageDir        string
		SpringXml             string
		AppName               string
		AppSrc                string
		ClassName             string
		JvmSize               string
		TomcatPort            int16
	}{
		ArtifactId:            m.Main.Name,
		ComponentDependencies: m.ComponentModules(),
		BasePackage:           basePackage,
		BasePackageDir:        java.Pkg2Path(basePackage),
		SpringXml:             m.Main.Plus.SpringXml,
		AppName:               m.Main.Name,
		AppSrc:                fmt.Sprintf("target/%s-%s-package", m.Main.Name, m.Main.Version),
		ClassName:             m.Main.Plus.EntryPointClass,
		JvmSize:               m.RpmByEnv("on-premise").JvmSize,
		TomcatPort:            m.RpmByEnv("on-premise").TomcatPort,
	}
	generateFile("gitignore", ".gitignore", data)
	generateFile("pom.xml", "pom.xml", data)
	generateFile("Makefile", "Makefile", data)

	generatePackageInfo(m)
	paths := [][]string{
		{"src", "main", "java", java.Pkg2Path(basePackage), "configuration"},
		{"src", "main", "java", java.Pkg2Path(basePackage), "controller"},
		{"src", "main", "java", java.Pkg2Path(basePackage), "device"},
		{"src", "main", "java", java.Pkg2Path(basePackage), "extension", "aspect"},
		{"src", "main", "java", java.Pkg2Path(basePackage), "repository"},
		{"src", "main", "java", java.Pkg2Path(basePackage), "application"},
		{"src", "main", "java", java.Pkg2Path(basePackage), "entity"},
		{"src", "main", "java", java.Pkg2Path(basePackage), "service"},
		{"src", "main", "java", "unsafe", "hack"},
		{"src", "main", "resources"},
		{"src", "test", "java"},
		{"src", "test", "resources"},
	}
	for _, p := range paths {
		mkdir(filepath.Join(p...))
	}

	generateFile("package.xml", filepath.Join("src", "main", "assembly", "package.xml"), data)

	if m.Main.Plus.SpringXml != "" {
		// 自动引导用户提供的 spring xml
		generateFile("spring.xml", filepath.Join("src", "main", "resources", m.Main.Plus.SpringXml), data)
		generateFile("spring.factories", filepath.Join("src", "main", "resources", "META-INF", "spring.factories"), data)
		generateFile("SpringResourcePlusLoader.java", filepath.Join("src", "main", "java", java.Pkg2Path(basePackage), "configuration", "SpringResourcePlusLoader.java"), data)
		generateFile("ExtensionPolicyRoutingEnhancementAspect.java", filepath.Join("src", "main", "java", "unsafe", "hack", "ExtensionPolicyRoutingEnhancementAspect.java"), data)
	}
}

func generatePackageInfo(m *manifest.Manifest) {
	pkg := m.Main.PlusBasePackage()
	data := struct {
		Package string
	}{
		Package: pkg,
	}
	mainClassDir := filepath.Join("src", "main", "java", java.Pkg2Path(pkg))
	fn := filepath.Join(mainClassDir, "package-info.java")
	generateFile("package-info.java", fn, data)
}

func generateFile(fromTemplateFile, targetFile string, data interface{}) {
	if util.FileExists(targetFile) {
		log.Printf("Skipped existing %s", targetFile)
		return
	}

	log.Printf("Generating %s", targetFile)
	fs.GenerateFileFromTmpl("templates/plus/"+fromTemplateFile, targetFile, data)
}

func mkdir(path string) {
	if util.DirExists(path) {
		return
	}

	log.Printf("Generating %s", path)
	if err := os.MkdirAll(path, 0755); err != nil {
		log.Fatalf("mkdir: %v", err)
	}
}

func validatePlusSpec(p manifest.PlusSpec) {
	if p.SpringXml != "" && !strings.HasSuffix(p.SpringXml, ".xml") {
		log.Fatalf("manifest.yaml federated.plus.springXml MUST be xml file instead of directory")
	}
}

func init() {
	manifest.RequiredManifestFileFlag(scaffoldCmd)
}
