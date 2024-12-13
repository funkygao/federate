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

func (p *pattern) createXMLPropertyReferenceRegex(key string) *regexp.Regexp {
	return regexp.MustCompile(`(\w+\s*=\s*"\$\{)` + regexp.QuoteMeta(key) + `(:[^}]*)?\}"`)
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
			".yml":        newYamlParser(),
			".yaml":       newYamlParser(),
			".properties": &propertiesParser{},
		},
	}
}
