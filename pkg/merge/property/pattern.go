package property

import (
	"path/filepath"
	"regexp"
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

func createJavaRegex(key string) *regexp.Regexp {
	return regexp.MustCompile(`@Value\s*\(\s*"([^"]*\$\{` + regexp.QuoteMeta(key) + `(:[^}]*)?\}[^"]*)"\s*\)`)
}

func createXmlRegex(key string) *regexp.Regexp {
	return regexp.MustCompile(`(\w+\s*=\s*"\$\{)` + regexp.QuoteMeta(key) + `(:[^}]*)?\}"`)
}

func createConfigurationPropertiesRegex(key string) *regexp.Regexp {
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
