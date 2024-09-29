package merge

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"federate/pkg/java"
	"federate/pkg/manifest"
)

type SpringBeanInjectionManager struct {
	resourcePattern         *regexp.Regexp
	autowiredImportPattern  *regexp.Regexp
	qualifierImportPattern  *regexp.Regexp
	importPattern           *regexp.Regexp
	resourceImportPattern   *regexp.Regexp
	wildcardImportPattern   *regexp.Regexp
	resourceWithNamePattern *regexp.Regexp
}

func NewSpringBeanInjectionManager() *SpringBeanInjectionManager {
	return &SpringBeanInjectionManager{
		resourcePattern:         regexp.MustCompile(`@Resource(\s*\([^)]*\))?`),
		autowiredImportPattern:  regexp.MustCompile(`import\s+org\.springframework\.beans\.factory\.annotation\.Autowired;`),
		qualifierImportPattern:  regexp.MustCompile(`import\s+org\.springframework\.beans\.factory\.annotation\.Qualifier;`),
		importPattern:           regexp.MustCompile(`(package\s+[\w.]+;\s*(?:import\s+[\w.]+;\s*)*)`),
		resourceImportPattern:   regexp.MustCompile(`import\s+javax\.annotation\.Resource;\s*`),
		wildcardImportPattern:   regexp.MustCompile(`import\s+javax\.annotation\.\*;\s*`),
		resourceWithNamePattern: regexp.MustCompile(`@Resource\s*\(\s*name\s*=\s*"([^"]*)"\s*\)`),
	}
}

func (m *SpringBeanInjectionManager) ReconcileResourceToAutowired(manifest *manifest.Manifest, dryRun bool) error {
	for _, component := range manifest.Components {
		if err := m.reconcileComponentInjections(component, dryRun); err != nil {
			return err
		}
	}
	return nil
}

func (m *SpringBeanInjectionManager) reconcileComponentInjections(component manifest.ComponentInfo, dryRun bool) error {
	return filepath.Walk(component.RootDir(), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !java.IsJavaMainSource(info, path) {
			return nil
		}

		content, err := ioutil.ReadFile(path)
		if err != nil {
			return err
		}

		oldContent := string(content)
		newContent := m.replaceResourceWithAutowired(oldContent)

		if newContent != oldContent {
			log.Printf("[%s] Reconciling @Resource to @Autowired in %s", component.Name, path)
			if !dryRun {
				err = ioutil.WriteFile(path, []byte(newContent), info.Mode())
				if err != nil {
					return err
				}
				log.Printf("Updated %s", path)
			}
		}
		return nil
	})
}

func (m *SpringBeanInjectionManager) replaceResourceWithAutowired(content string) string {
	// Quick check if @Resource is present
	if !m.resourcePattern.MatchString(content) {
		return content // No changes needed
	}

	// Replace @Resource with @Autowired and @Qualifier if necessary
	content = m.resourceWithNamePattern.ReplaceAllStringFunc(content, func(match string) string {
		name := m.resourceWithNamePattern.FindStringSubmatch(match)[1]
		return fmt.Sprintf("@Autowired\n    @Qualifier(\"%s\")", name)
	})
	content = m.resourcePattern.ReplaceAllString(content, "@Autowired")

	// Check if @Autowired is already imported
	if !m.autowiredImportPattern.MatchString(content) {
		content = m.importPattern.ReplaceAllString(content, "${1}import org.springframework.beans.factory.annotation.Autowired;\n")
	}

	// Check if @Qualifier is needed and not already imported
	if strings.Contains(content, "@Qualifier") && !m.qualifierImportPattern.MatchString(content) {
		content = m.importPattern.ReplaceAllString(content, "${1}import org.springframework.beans.factory.annotation.Qualifier;\n")
	}

	// Handle import statements
	if m.wildcardImportPattern.MatchString(content) {
		// Check if there are any other javax.annotation.* annotations still in use
		otherAnnotationsPattern := regexp.MustCompile(`@(?:PostConstruct|PreDestroy|Resource)\b`)
		if otherAnnotationsPattern.MatchString(content) {
			// Keep the wildcard import
		} else {
			// Remove the wildcard import if no other javax.annotation.* annotations are used
			content = m.wildcardImportPattern.ReplaceAllString(content, "")
		}
	} else {
		// Remove only the Resource import, preserving the line break
		lines := strings.Split(content, "\n")
		var newLines []string
		for i, line := range lines {
			if !m.resourceImportPattern.MatchString(line) {
				newLines = append(newLines, line)
			} else if i+1 < len(lines) && lines[i+1] == "" {
				// If the next line is empty, keep it
				newLines = append(newLines, "")
			}
		}
		content = strings.Join(newLines, "\n")
	}

	return content
}
