package merge

import (
	"regexp"
)

// global
var P pattern

type pattern struct {
	// @Resource @Autowired @Qualifier
	resourcePattern         *regexp.Regexp
	resourceWithNamePattern *regexp.Regexp
	methodResourcePattern   *regexp.Regexp

	autowiredPattern       *regexp.Regexp
	methodAutowiredPattern *regexp.Regexp
	qualifierPattern       *regexp.Regexp

	genericTypePattern *regexp.Regexp

	// @RequestMapping
	requestMappingRegex *regexp.Regexp

	// System.getProperty
	systemGetPropertyRegex *regexp.Regexp

	// Xxx.getBean
	getBeanPattern *regexp.Regexp

	importRegex *regexp.Regexp

	importResourcePattern *regexp.Regexp
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

func (p *pattern) IsInjectionAnnotatedLine(line string) bool {
	return p.resourcePattern.MatchString(line) || p.autowiredPattern.MatchString(line)
}

func init() {
	P = pattern{
		resourcePattern:         regexp.MustCompile(`@Resource\b(\s*\([^)]*\))?`),
		resourceWithNamePattern: regexp.MustCompile(`@Resource\s*\(\s*name\s*=\s*"([^"]*)"\s*\)`),
		methodResourcePattern:   regexp.MustCompile(`@Resource\b(\s*\([^)]*\))?\s*\n\s*public\s+void\s+(set\w+)\s*\(`),

		autowiredPattern:       regexp.MustCompile(`@Autowired\b(\s*\([^)]*\))?`),
		methodAutowiredPattern: regexp.MustCompile(`@Autowired\b(\s*\([^)]*\))?\s*\n\s*public\s+void\s+(set\w+)\s*\(`),
		qualifierPattern:       regexp.MustCompile(`@Qualifier\s*\(\s*"([^"]*)"\s*\)`),

		genericTypePattern: regexp.MustCompile(`(Map|List)<.*>`),

		requestMappingRegex: regexp.MustCompile(`(@RequestMapping\s*\(\s*(?:value\s*=)?\s*")([^"]+)("\s*\))`),

		systemGetPropertyRegex: regexp.MustCompile(`System\.getProperty\s*\(\s*([^)]+)\s*\)`),

		getBeanPattern: regexp.MustCompile(`\bgetBean\s*\(\s*"([^"]+)"\s*\)`),

		importRegex:           regexp.MustCompile(`^\s*import\s+(?:static\s+)?[\w.]+(?:\s*\*\s*)?;?\s*`),
		importResourcePattern: regexp.MustCompile(`@ImportResource\s*\(\s*(locations\s*=\s*)?\{?\s*("([^"]+)"|'([^']+)')\s*(,\s*("([^"]+)"|'([^']+)')\s*)*\}?\s*\)`),
	}
}
