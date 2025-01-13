package ast

import "federate/pkg/primitive"

var (
	TopK int

	ignoredInterfaces  = primitive.NewStringSet()
	ignoredAnnotations = primitive.NewStringSet()
)

func init() {
	ignoredInterfaces.Add("Serializable")
	ignoredAnnotations.Add("Override", "Data", "NoArgsConstructor", "AllArgsConstructor", "Builder", "SuppressWarnings", "Slf4j")
}
