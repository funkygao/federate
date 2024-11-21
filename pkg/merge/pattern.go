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

	// System.getProperty
	systemGetPropertyRegex *regexp.Regexp

	importRegex           *regexp.Regexp
	importResourcePattern *regexp.Regexp
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

		systemGetPropertyRegex: regexp.MustCompile(`System\.getProperty\s*\(\s*([^)]+)\s*\)`),

		importRegex:           regexp.MustCompile(`^\s*import\s+(?:static\s+)?[\w.]+(?:\s*\*\s*)?;?\s*`),
		importResourcePattern: regexp.MustCompile(`@ImportResource\s*\(\s*(locations\s*=\s*)?\{?\s*("([^"]+)"|'([^']+)')\s*(,\s*("([^"]+)"|'([^']+)')\s*)*\}?\s*\)`),
	}
}
