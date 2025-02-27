package ast

import "federate/pkg/primitive"

var (
	TopK           int
	Verbosity      int
	Web            bool
	GeneratePrompt bool
	ShowException  bool

	ignoredInterfaces       = primitive.NewStringSet()
	ignoredAnnotations      = primitive.NewStringSet()
	ignoredCompositionTypes = primitive.NewStringSet()

	significantInheritanceChildren = 2
	significantInheritanceDepth    = 2
)

func init() {
	ignoredInterfaces.Add("Serializable")

	ignoredAnnotations.Add("Override", "SuppressWarnings",
		"Data", "NoArgsConstructor", "AllArgsConstructor", "Builder")

	ignoredCompositionTypes.Add("String", "Integer", "Long", "Float", "Double", "Boolean",
		"int", "long", "float", "double", "boolean",
		"BigDecimal", "Date", "List", "Map", "Set",
		"ArrayList", "HashMap", "HashSet",
		"Object", "Enum")
}
