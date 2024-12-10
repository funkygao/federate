package merge

import (
	"os"
	"path/filepath"

	"federate/internal/fs"
	"federate/pkg/federated"
	"federate/pkg/manifest"
)

type javaAstTransformer struct {
	m *manifest.Manifest
}

func NewJavaAstTransformer(m *manifest.Manifest) Reconciler {
	return &javastTransformer{m: m}
}

func (m *javaAstTransformer) Name() string {
	return "Instrument Java AST for all conflicts"
}

func (m *javaAstTransformer) Reconcile() error {
	return javast.Instrument(m.m)
}
