package addon

import (
	"fmt"
	"log"
	"path/filepath"

	"federate/internal/fs"
	"federate/pkg/java"
	"federate/pkg/manifest"
	"federate/pkg/util"
)

type FusionProjectGenerator struct {
	m *manifest.Manifest
}

func NewFusionProjectGenerator(m *manifest.Manifest) *FusionProjectGenerator {
	return &FusionProjectGenerator{m: m}
}

func (f *FusionProjectGenerator) Name() string {
	return "Generating federated system scaffold"
}

func (f *FusionProjectGenerator) Reconcile() error {
	f.generatePomFile()
	f.generateMainClassFile()
	f.generateMetaInf()
	f.generateMakefile()
	f.generatePackageXml()
	f.generateStartStopScripts()
	f.copyTaint()
	return nil
}

func (f *FusionProjectGenerator) generatePomFile() {
	m := f.m
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
	fs.GenerateFileFromTmpl("templates/pom.xml", fn, pomData)
}

func (f *FusionProjectGenerator) generateMetaInf() {
	m := f.m
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
}

func (f *FusionProjectGenerator) generateMainClassFile() {
	packageName, className := f.m.ParseMainClass()
	m := f.m
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
	mainClassDir := filepath.Join(root, "src", "main", "java", java.Pkg2Path(packageName))
	fn := filepath.Join(mainClassDir, className+".java")
	fs.GenerateFileFromTmpl("templates/mainClass.java", fn, mainClassData)
}

func (f *FusionProjectGenerator) generateMakefile() {
	m := f.m
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
	fs.GenerateFileFromTmpl("templates/Makefile", fn, data)
}

func (f *FusionProjectGenerator) generatePackageXml() {
	root, err := f.m.CreateTargetSystemDir()
	if err != nil {
		log.Fatalf("%v", err)
	}
	fn := filepath.Join(root, "src", "main", "assembly", "package.xml")
	fs.GenerateFileFromTmpl("templates/package.xml", fn, nil) // TODO 动态指定哪些资源文件拷贝的目标包
}

func (f *FusionProjectGenerator) generateStartStopScripts() {
	m := f.m
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
		log.Printf("Overwrite %s, %s", start, filepath.Base(stop))
	} else {
		log.Printf("Generated %s, %s", start, filepath.Base(stop))
	}
}

func (f *FusionProjectGenerator) copyTaint() {
	m := f.m
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
		log.Printf("Taint Copied %s -> %s", src, target)
	}
}
