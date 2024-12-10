package property

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
)

// global
var P pattern

type pattern struct {
	// @RequestMapping
	requestMappingRegex *regexp.Regexp

	// 属性的引用：${xxx}
	placeholderRegex *regexp.Regexp

	parsers map[string]PropertyParser // key is filename ext
}

func (p *pattern) createAnnotationReferenceRegex(key string) *regexp.Regexp {
	annotations := []string{
		"@Value",
		"@ConditionalOnProperty",
		"@PostMapping",
		"@GetMapping",
		"@DeleteMapping",
		"@PutMapping",
		//"@RequestMapping",
		//"@ConfigurationProperties",
	}

	annotationPattern := `(` + strings.Join(annotations, `|`) + `)`
	keyPattern := regexp.QuoteMeta(key)

	// Build a regex pattern that matches annotations referencing the key
	pattern := fmt.Sprintf(`(?s)%s\s*\(\s*[^)]*%s[^)]*\)`, annotationPattern, keyPattern)

	return regexp.MustCompile(pattern)
}

func (p *pattern) createXMLPropertyReferenceRegex(key string) *regexp.Regexp {
	return regexp.MustCompile(`(\w+\s*=\s*"\$\{)` + regexp.QuoteMeta(key) + `(:[^}]*)?\}"`)
}

func (p *pattern) createConfigurationPropertiesPrefixRegex(key string) *regexp.Regexp {
	return regexp.MustCompile(`@ConfigurationProperties\s*\(\s*"` + regexp.QuoteMeta(key) + `"\s*\)`)
}

func ParserByFile(file string) (parser PropertyParser, supported bool) {
	parser, supported = P.parsers[filepath.Ext(file)]
	return
}

func init() {
	P = pattern{
		requestMappingRegex: regexp.MustCompile(`(@RequestMapping\s*\(\s*(?:value\s*=)?\s*")([^"]+)("\s*\))`),
		placeholderRegex:    regexp.MustCompile(`\$\{([^}]+)\}`),
		parsers: map[string]PropertyParser{
			".yml":        &yamlParser{},
			".yaml":       &yamlParser{},
			".properties": &propertiesParser{},
		},
	}
}
