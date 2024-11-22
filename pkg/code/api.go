package code

// import com.foo.FooService;
func IsImportStatement(line string) bool {
	return P.importRegex.MatchString(line)
}

// @Resource
func IsResourceAnnotatedLine(line string) bool {
	return P.resourcePattern.MatchString(line)
}

// @Resource(name = "foo")
func IsResourceAnnotatedWithNameLine(line string) bool {
	return P.resourceWithNamePattern.MatchString(line)
}

// 获取 @Resource(name = "foo") 其中的 name 值
func GetResourceAnnotationName(line string) string {
	matches := P.resourceWithNamePattern.FindStringSubmatch(line)
	if len(matches) > 1 {
		return matches[1]
	}
	return ""
}

// @Resource
// public setFoo(Foo foo) {
// line2 = line + "\n" + nextLine
func IsMethodResourceAnnotatedLines(line2 string) bool {
	return P.methodResourcePattern.MatchString(line2)
}

// @Autowired
// public setFoo(Foo foo) {
// line2 = line + "\n" + nextLine
func IsMethodAutowiredAnnotatedLines(line2 string) bool {
	return P.methodAutowiredPattern.MatchString(line2)
}

// @Resource | @Autowired
func IsInjectionAnnotatedLine(line string) bool {
	return P.resourcePattern.MatchString(line) || P.autowiredPattern.MatchString(line)
}

// 检查是否为 Map, HashMap 或 List 类型
func IsCollectionTypeLine(line string) bool {
	return P.collectionTypePattern.MatchString(line)
}
