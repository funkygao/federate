package code

import (
	"regexp"
)

// global
var P pattern

type pattern struct {
	// @Resource
	resourcePattern *regexp.Regexp

	// @Resource(name = "foo")
	resourceWithNamePattern *regexp.Regexp

	// @Resource
	// public void setFoo(Foo foo) {
	methodResourcePattern *regexp.Regexp

	// @Autowired
	autowiredPattern *regexp.Regexp

	// @Autowired
	// public void setFoo(Foo foo) {
	methodAutowiredPattern *regexp.Regexp

	// @Qualifier
	qualifierPattern *regexp.Regexp

	// private Map Hash List beans;
	collectionTypePattern *regexp.Regexp

	// 环境变量在 java 源代码和资源文件的引用
	SystemGetPropertyRegex *regexp.Regexp
	XmlEnvRef              *regexp.Regexp

	importRegex *regexp.Regexp

	ImportResourcePattern *regexp.Regexp
}

func init() {
	P = pattern{
		resourcePattern:         regexp.MustCompile(`@Resource\b(\s*\([^)]*\))?`),
		resourceWithNamePattern: regexp.MustCompile(`@Resource\s*\(\s*name\s*=\s*"([^"]*)"\s*\)`),
		methodResourcePattern:   regexp.MustCompile(`@Resource\b(\s*\([^)]*\))?\s*\n\s*public\s+void\s+(set\w+)\s*\(`),

		autowiredPattern:       regexp.MustCompile(`@Autowired\b(\s*\([^)]*\))?`),
		methodAutowiredPattern: regexp.MustCompile(`@Autowired\b(\s*\([^)]*\))?\s*\n\s*public\s+void\s+(set\w+)\s*\(`),
		qualifierPattern:       regexp.MustCompile(`@Qualifier\s*\(\s*"([^"]*)"\s*\)`),

		collectionTypePattern: regexp.MustCompile(`(Map|List)<.*>`),

		SystemGetPropertyRegex: regexp.MustCompile(`System\.getProperty\s*\(\s*([^)]+)\s*\)`),
		XmlEnvRef:              regexp.MustCompile(`\$\{([^:}]+)(?::[^}]+)?\}`),

		importRegex:           regexp.MustCompile(`^\s*import\s+(?:static\s+)?[\w.]+(?:\s*\*\s*)?;?\s*`),
		ImportResourcePattern: regexp.MustCompile(`@ImportResource\s*\(\s*(locations\s*=\s*)?\{?\s*("([^"]+)"|'([^']+)')\s*(,\s*("([^"]+)"|'([^']+)')\s*)*\}?\s*\)`),
	}
}
