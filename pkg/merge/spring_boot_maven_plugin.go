package merge

import (
	"log"
	"os"
	"path/filepath"

	"federate/pkg/manifest"
	"federate/pkg/util"
	"github.com/beevik/etree"
)

type springBootMavenPluginManager struct {
	m *manifest.Manifest
}

func NewSpringBootMavenPluginManager(m *manifest.Manifest) Reconciler {
	return &springBootMavenPluginManager{m: m}
}

func (s *springBootMavenPluginManager) Name() string {
	return "Instrument pom.xml spring-boot-maven-plugin"
}

func (s *springBootMavenPluginManager) Reconcile() error {
	for _, c := range s.m.Components {
		rootPom := filepath.Join(c.RootDir(), "pom.xml")
		if err := s.instrumentPom(c, rootPom); err != nil {
			return err
		}

		for _, module := range c.ChildDirs() {
			pomPath := filepath.Join(c.RootDir(), module, "pom.xml")
			if !util.FileExists(pomPath) {
				continue
			}
			if err := s.instrumentPom(c, pomPath); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *springBootMavenPluginManager) instrumentPom(c manifest.ComponentInfo, pomPath string) error {
	doc := etree.NewDocument()
	if err := doc.ReadFromFile(pomPath); err != nil {
		return err
	}

	project := doc.SelectElement("project")
	if project == nil {
		log.Printf("Warning: %s is not a valid pom.xml", pomPath)
		return nil
	}

	// Find spring-boot-maven-plugin
	build := project.SelectElement("build")
	if build == nil {
		return nil
	}

	plugins := build.SelectElement("plugins")
	if plugins == nil {
		return nil
	}

	springBootPluginFound := false
	for _, plugin := range plugins.SelectElements("plugin") {
		artifactId := plugin.SelectElement("artifactId")
		if artifactId != nil && artifactId.Text() == "spring-boot-maven-plugin" {
			springBootPluginFound = true
			break
		}
	}

	if !springBootPluginFound {
		return nil
	}

	// Add federate.packaging property if not exists
	properties := project.SelectElement("properties")
	if properties == nil {
		properties = project.CreateElement("properties")
	}
	if properties.SelectElement("federate.packaging") == nil {
		federatePackaging := properties.CreateElement("federate.packaging")
		federatePackaging.SetText("false")
	}

	// Update spring-boot-maven-plugin configuration
	for _, plugin := range plugins.SelectElements("plugin") {
		artifactId := plugin.SelectElement("artifactId")
		if artifactId != nil && artifactId.Text() == "spring-boot-maven-plugin" {
			configuration := plugin.SelectElement("configuration")
			if configuration == nil {
				configuration = plugin.CreateElement("configuration")
			}
			skip := configuration.SelectElement("skip")
			if skip == nil {
				skip = configuration.CreateElement("skip")
			}
			skip.SetText("${federate.packaging}")
			log.Printf("[%s] Instrumented spring-boot-maven-plugin configuration: %s", c.Name, pomPath)
			break
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
