package ast

import "federate/pkg/primitive"

var (
	TopK int

	ignoredInterfaces  = primitive.NewStringSet()
	ignoredAnnotations = primitive.NewStringSet()

	significantInheritanceChildren = 2
	significantInheritanceDepth    = 2
)

func init() {
	ignoredInterfaces.Add("Serializable")
	ignoredAnnotations.Add("Override", "Data", "NoArgsConstructor", "AllArgsConstructor", "Builder", "SuppressWarnings", "Slf4j")
}
