package property

import (
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

func (p *pattern) createJavaRegex(key string) *regexp.Regexp {
	return regexp.MustCompile(`@Value\s*\(\s*"\$\{` + regexp.QuoteMeta(key) + `(:[^}]*)?\}"\s*\)`)
}

func (p *pattern) createXmlRegex(key string) *regexp.Regexp {
	return regexp.MustCompile(`(value|key)="\$\{` + regexp.QuoteMeta(key) + `(:[^}]*)?\}"`)
}

func (p *pattern) createConfigurationPropertiesRegex(key string) *regexp.Regexp {
	return regexp.MustCompile(`@ConfigurationProperties\s*\(\s*"` + regexp.QuoteMeta(key) + `"\s*\)`)
}

func (p *pattern) ParserByFileExt(ext string) (parser PropertyParser, supported bool) {
	parser, supported = p.parsers[ext]
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
