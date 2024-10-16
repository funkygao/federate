package merge

import (
	"log"
	"os"
	"path/filepath"

	"federate/pkg/manifest"
	"federate/pkg/util"
	"github.com/beevik/etree"
	"github.com/fatih/color"
)

func instrumentPomForFederatePackaging(m *manifest.Manifest) {
	for _, c := range m.Components {
		rootPom := filepath.Join(c.RootDir(), "pom.xml")
		if err := instrumentPom(rootPom); err != nil {
			log.Fatalf("%s: %v", rootPom, err)
		}

		for _, module := range c.ChildDirs() {
			pomPath := filepath.Join(c.RootDir(), module, "pom.xml")
			if !util.FileExists(pomPath) {
				continue
			}
			if err := instrumentPom(pomPath); err != nil {
				log.Fatalf("%s: %v", pomPath, err)
			}
		}
	}
	color.Green("🍺 pom.xml Instrumented for federate packaging")
}

func instrumentPom(pomPath string) error {
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
		log.Printf("Added federate.packaging property to %s", pomPath)
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
			log.Printf("Updated spring-boot-maven-plugin configuration in %s", pomPath)
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
