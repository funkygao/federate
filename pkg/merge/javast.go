package merge

import (
	"federate/pkg/javast"
	"federate/pkg/manifest"
)

type javaAstTransformer struct {
	m *manifest.Manifest
}

func NewJavaAstTransformer(m *manifest.Manifest) Reconciler {
	return &javaAstTransformer{m: m}
}

func (m *javaAstTransformer) Name() string {
	return "Instrument Java AST Transformers to Execute Plans"
}

func (m *javaAstTransformer) Reconcile() error {
	return javast.Instrument()
}
