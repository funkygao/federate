package merge

import (
	"federate/pkg/javast"
	"federate/pkg/manifest"
	"federate/pkg/step"
)

type javaAstTransformer struct {
	m *manifest.Manifest
}

func NewJavaAstTransformer(m *manifest.Manifest) Reconciler {
	return &javaAstTransformer{m: m}
}

func (m *javaAstTransformer) Name() string {
	return "Batch Instrument Java AST Transformers"
}

func (m *javaAstTransformer) Reconcile(bar step.Bar) error {
	return javast.Instrument(bar)
}
