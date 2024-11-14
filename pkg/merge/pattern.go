package merge

import (
	"regexp"
)

// global
var P pattern

type pattern struct {
	resourcePattern         *regexp.Regexp
	resourceWithNamePattern *regexp.Regexp
	methodResourcePattern   *regexp.Regexp
	genericTypePattern      *regexp.Regexp
	autowiredPattern        *regexp.Regexp
	qualifierPattern        *regexp.Regexp
}

func init() {
	P = pattern{
		resourcePattern:         regexp.MustCompile(`@Resource(\s*\([^)]*\))?`),
		resourceWithNamePattern: regexp.MustCompile(`@Resource\s*\(\s*name\s*=\s*"([^"]*)"\s*\)`),
		methodResourcePattern:   regexp.MustCompile(`@Resource(\s*\([^)]*\))?\s*\n\s*public\s+void\s+(set\w+)\s*\(`),
		genericTypePattern:      regexp.MustCompile(`(Map|List)<.*>`),
		autowiredPattern:        regexp.MustCompile(`@Autowired(\s*\([^)]*\))?`),
		qualifierPattern:        regexp.MustCompile(`@Qualifier\s*\(\s*"([^"]*)"\s*\)`),
	}
}
