package merge

import (
	"fmt"
	"log"
	"path/filepath"
	"strings"

	"federate/internal/fs"
	"federate/pkg/java"
	"federate/pkg/manifest"
	"federate/pkg/util"
	"github.com/fatih/color"
)

func scaffoldTargetSystem(m *manifest.Manifest) {
	packageName, className := m.ParseMainClass()
	generatePomFile(m)
	generateMainClassFile(packageName, className, m)
	generateMetaInf(m)
	generateMakefile(m)
	generatePackageXml(m)
	generateStartStopScripts(m)
	copyTaint(m)

	color.Green("🍺 Scaffold generated for target system: %s", m.Main.Name)
}

func generatePomFile(m *manifest.Manifest) {
	pomData := struct {
		ArtifactId          string
		GroupId             string
		IncludeDependencies []java.DependencyInfo
		ExcludeDependencies []java.DependencyInfo
	}{
		ArtifactId:          m.Main.Name,
		GroupId:             m.Main.GroupId,
		IncludeDependencies: m.Main.Dependency.Includes,
		ExcludeDependencies: m.Main.Dependency.Excludes,
	}
	root, err := m.CreateTargetSystemDir()
	if err != nil {
		log.Fatalf("%v", err)
	}
	fn := filepath.Join(root, "pom.xml")
	overwrite := fs.GenerateFileFromTmpl("templates/pom.xml", fn, pomData)
	if overwrite {
		color.Yellow("Overwrite %s", fn)
	} else {
		color.Cyan("Generated %s", fn)
	}
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
	overwrite := fs.GenerateFileFromTmpl("templates/spring.factories", fn, data)
	if overwrite {
		color.Yellow("Overwrite %s", fn)
	} else {
		color.Cyan("Generated %s", fn)
	}
}

func generateMainClassFile(packageName, className string, m *manifest.Manifest) {
	mainClassData := struct {
		App                     string
		Package                 string
		ClassName               string
		FederatedRuntimePackage string
		Profile                 string
		BasePackages            []string
		ExcludedTypes           []string
		Imports                 []string
		Excludes                []string
	}{
		App:                     m.Main.Name,
		Package:                 packageName,
		ClassName:               className,
		Profile:                 m.Main.SpringProfile,
		FederatedRuntimePackage: m.Main.FederatedRuntimePackage(),
		BasePackages:            m.Main.MainClass.ComponentScan.BasePackages,
		ExcludedTypes:           m.Main.MainClass.ComponentScan.ExcludedTypes,
		Imports:                 m.Main.MainClass.Imports,
		Excludes:                m.Main.MainClass.Excludes,
	}
	root, err := m.CreateTargetSystemDir()
	if err != nil {
		log.Fatalf("%v", err)
	}
	mainClassDir := filepath.Join(root, "src", "main", "java", filepath.FromSlash(strings.ReplaceAll(packageName, ".", "/")))
	fn := filepath.Join(mainClassDir, className+".java")
	overwrite := fs.GenerateFileFromTmpl("templates/mainClass.java", fn, mainClassData)
	if overwrite {
		color.Yellow("Overwrite %s", fn)
	} else {
		color.Cyan("Generated %s", fn)
	}
}

func generateMakefile(m *manifest.Manifest) {
	data := struct {
		AppName    string
		AppSrc     string
		ClassName  string
		JvmSize    string
		TomcatPort int16
		Env        string
	}{
		AppName:    m.Main.Name,
		AppSrc:     fmt.Sprintf("target/%s-%s-package", m.Main.Name, m.Main.Version),
		ClassName:  m.Main.MainClass.Name,
		JvmSize:    m.RpmByEnv("on-premise").JvmSize,
		TomcatPort: m.RpmByEnv("on-premise").TomcatPort,
		Env:        m.Main.Runtime.Env,
	}
	if data.JvmSize == "" {
		data.JvmSize = "large"
	}
	root, err := m.CreateTargetSystemDir()
	if err != nil {
		log.Fatalf("%v", err)
	}
	fn := filepath.Join(root, "Makefile")
	overwrite := fs.GenerateFileFromTmpl("templates/Makefile", fn, data)
	if overwrite {
		color.Yellow("Overwrite %s", fn)
	} else {
		color.Cyan("Generated %s", fn)
	}
}

func generatePackageXml(m *manifest.Manifest) {
	root, err := m.CreateTargetSystemDir()
	if err != nil {
		log.Fatalf("%v", err)
	}
	fn := filepath.Join(root, "src", "main", "assembly", "package.xml")
	overwrite := fs.GenerateFileFromTmpl("templates/package.xml", fn, nil) // TODO 动态指定哪些资源文件拷贝的目标包
	if overwrite {
		color.Yellow("Overwrite %s", fn)
	} else {
		color.Cyan("Generated %s", fn)
	}
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
	overwrite := fs.GenerateExecutableFileFromTmpl("templates/stop.sh", stop, data)
	if overwrite {
		color.Yellow("Overwrite %s, %s", start, filepath.Base(stop))
	} else {
		color.Cyan("Generated %s, %s", start, filepath.Base(stop))
	}
}

func copyTaint(m *manifest.Manifest) {
	root, err := m.CreateTargetSystemDir()
	if err != nil {
		log.Fatalf("%v", err)
	}

	for _, f := range m.Main.Reconcile.Taint.ResourceFiles() {
		src := filepath.Join(m.StarterBaseDir(), f)
		target := filepath.Join(root, "src", "main", "resources", f)
		if err := util.CopyFile(src, target); err != nil {
			log.Fatalf("%v", err)
		}
		color.Cyan("Taint Copied %s -> %s", src, target)
	}

}
