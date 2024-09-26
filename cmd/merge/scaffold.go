package merge

import (
	"fmt"
	"log"
	"path/filepath"
	"strings"

	"federate/internal/fs"
	"federate/pkg/manifest"
	"federate/pkg/merge"
	"github.com/fatih/color"
)

func createFederatedSystem(m *manifest.Manifest) {
	packageName, className := m.ParseMainClass()
	generatePomFile(m)
	generateMainClassFile(packageName, className, m)
	generateMetaInf(m)
	generateMakefile(m)
	generatePackageXml(m)
	generateStartStopScripts(m)
	copyTaint(m)

	color.Green("🍺 Scaffold generated for federated system: %s", m.Main.Name)
}

func generatePomFile(m *manifest.Manifest) {
	pomData := struct {
		ArtifactId            string
		GroupId               string
		Parent                manifest.DependencyInfo
		ComponentDependencies []manifest.DependencyInfo
		MainDependencies      []manifest.DependencyInfo
	}{
		ArtifactId:            m.Main.Name,
		GroupId:               m.Main.GroupId(),
		Parent:                m.Main.Parent,
		ComponentDependencies: m.ComponentDependencies(),
		MainDependencies:      m.Main.Dependencies,
	}
	root, err := m.CreateTargetSystemDir()
	if err != nil {
		log.Fatalf("%v", err)
	}
	fn := filepath.Join(root, "pom.xml")
	fs.GenerateFileFromTmpl("templates/pom.xml", fn, pomData)
	color.Cyan("Generated %s", fn)
}

func generateMetaInf(m *manifest.Manifest) {
	root, err := m.CreateTargetSystemDir()
	if err != nil {
		log.Fatalf("%v", err)
	}

	data := struct {
		FederatedRuntimePackage string
	}{
		FederatedRuntimePackage: m.Main.FederatedRuntimePackage(),
	}
	fn := filepath.Join(root, "src", "main", "resources", "META-INF", "spring.factories")
	fs.GenerateFileFromTmpl("templates/spring.factories", fn, data)
	color.Cyan("Generated %s", fn)
}

func generateMainClassFile(packageName, className string, m *manifest.Manifest) {
	mainClassData := struct {
		Package                 string
		ClassName               string
		FederatedRuntimePackage string
		Profile                 string
		BasePackages            []string
		ExcludedTypes           []string
		Imports                 []string
	}{
		Package:                 packageName,
		ClassName:               className,
		Profile:                 m.Main.SpringProfile,
		FederatedRuntimePackage: m.Main.FederatedRuntimePackage(),
		BasePackages:            m.Main.MainClass.ComponentScan.BasePackages,
		ExcludedTypes:           m.Main.MainClass.ComponentScan.ExcludedTypes,
		Imports:                 m.Main.MainClass.Imports,
	}
	root, err := m.CreateTargetSystemDir()
	if err != nil {
		log.Fatalf("%v", err)
	}
	mainClassDir := filepath.Join(root, "src", "main", "java", filepath.FromSlash(strings.ReplaceAll(packageName, ".", "/")))
	fn := filepath.Join(mainClassDir, className+".java")
	fs.GenerateFileFromTmpl("templates/mainClass.java", fn, mainClassData)
	color.Cyan("Generated %s", fn)
}

func generateMakefile(m *manifest.Manifest) {
	data := struct {
		AppName    string
		AppSrc     string
		JvmSize    string
		ClassName  string
		TomcatPort int16
	}{
		AppName:    m.Main.Name,
		AppSrc:     fmt.Sprintf("target/%s-%s-package", m.Main.Name, m.Main.Version),
		JvmSize:    m.Main.JvmSize,
		ClassName:  m.Main.MainClass.Name,
		TomcatPort: m.Main.TomcatPort,
	}
	root, err := m.CreateTargetSystemDir()
	if err != nil {
		log.Fatalf("%v", err)
	}
	fn := filepath.Join(root, "Makefile")
	fs.GenerateFileFromTmpl("templates/Makefile", fn, data)
	color.Cyan("Generated %s", fn)
}

func generatePackageXml(m *manifest.Manifest) {
	root, err := m.CreateTargetSystemDir()
	if err != nil {
		log.Fatalf("%v", err)
	}
	fn := filepath.Join(root, "src", "main", "assembly", "package.xml")
	fs.GenerateFileFromTmpl("templates/package.xml", fn, nil)
	color.Cyan("Generated %s", fn)
}

func generateStartStopScripts(m *manifest.Manifest) {
	root, err := m.CreateTargetSystemDir()
	if err != nil {
		log.Fatalf("%v", err)
	}

	data := struct {
		AppName   string
		MainClass string
	}{
		AppName:   m.Main.Name,
		MainClass: m.Main.MainClass.Name,
	}
	start := filepath.Join(root, "src", "main", "assembly", "bin", "start.sh")
	fs.GenerateExecutableFileFromTmpl("templates/start.sh", start, data)
	stop := filepath.Join(root, "src", "main", "assembly", "bin", "stop.sh")
	fs.GenerateExecutableFileFromTmpl("templates/stop.sh", stop, data)
	color.Cyan("Generated %s, %s", start, filepath.Base(stop))
}

func copyTaint(m *manifest.Manifest) {
	root, err := m.CreateTargetSystemDir()
	if err != nil {
		log.Fatalf("%v", err)
	}

	for _, f := range m.Main.Reconcile.Taint.ResourceFiles() {
		src := filepath.Join(m.Dir, f)
		target := filepath.Join(root, "src", "main", "resources", f)
		if err := merge.CopyFile(src, target); err != nil {
			log.Fatalf("%v", err)
		}
		color.Cyan("Copied %s -> %s", src, target)
	}

}
