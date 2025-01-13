package ast

import "federate/pkg/primitive"

var (
	TopK int

	ignoredInterfaces       = primitive.NewStringSet()
	ignoredAnnotations      = primitive.NewStringSet()
	ignoredCompositionTypes = primitive.NewStringSet()

	significantInheritanceChildren = 2
	significantInheritanceDepth    = 2
)

func init() {
	ignoredInterfaces.Add("Serializable")
	ignoredAnnotations.Add("Override", "Data", "NoArgsConstructor", "AllArgsConstructor", "Builder", "SuppressWarnings", "Slf4j")
	ignoredTypes := []string{
		"String", "Integer", "Long", "Float", "Double", "Boolean",
		"int", "long", "float", "double", "boolean",
		"BigDecimal", "Date", "List", "Map", "Set",
		"ArrayList", "HashMap", "HashSet",
		"Object", "Enum",
	}
	for _, t := range ignoredTypes {
		ignoredCompositionTypes.Add(t)
	}
}
