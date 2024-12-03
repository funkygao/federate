package code

import (
	"os"
	"path/filepath"

	"federate/pkg/java"
	"federate/pkg/manifest"
)

type JavaWalker struct {
	c        manifest.ComponentInfo
	rootDir  string
	visitors []JavaFileVisitor
}

func NewComponentJavaWalker(c manifest.ComponentInfo) *JavaWalker {
	w := NewJavaWalker(c.RootDir())
	w.c = c
	return w
}

func NewJavaWalker(rootDir string) *JavaWalker {
	return &JavaWalker{
		rootDir:  rootDir,
		visitors: make([]JavaFileVisitor, 0),
	}
}

func (w *JavaWalker) AddVisitor(visitors ...JavaFileVisitor) *JavaWalker {
	w.visitors = append(w.visitors, visitors...)
	return w
}

func (w *JavaWalker) Walk(opts ...AcceptOption) error {
	return filepath.Walk(w.rootDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !java.IsJavaMainSource(info, path) {
			return nil
		}
		if w.c.M != nil && w.c.M.MainClass.ExcludeJavaFile(info.Name()) {
			return nil
		}

		return w.processJavaFile(path, info, opts...)
	})
}

func (w *JavaWalker) processJavaFile(path string, info os.FileInfo, opts ...AcceptOption) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	javaFile := NewJavaFile(path, &w.c, content).withInfo(info)
	javaFile.AddVisitor(w.visitors...)
	javaFile.Accept(opts...)

	return nil
}
