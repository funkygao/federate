package code

import (
	"log"
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

func (w *JavaWalker) AddVisitor(visitor JavaFileVisitor) *JavaWalker {
	w.visitors = append(w.visitors, visitor)
	return w
}

func (w *JavaWalker) Walk() error {
	return filepath.Walk(w.rootDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !java.IsJavaMainSource(info, path) {
			return nil
		}
		if w.c.M != nil && w.c.M.MainClass.ExcludeJavaFile(info.Name()) {
			log.Printf("[%s] Excluded %s", w.c.Name, info.Name())
			return nil
		}

		return w.processJavaFile(path, info)
	})
}

func (w *JavaWalker) processJavaFile(path string, info os.FileInfo) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	javaFile := NewJavaFile(path, &w.c, string(content)).WithInfo(info)
	for _, visitor := range w.visitors {
		javaFile.AddVisitor(visitor)
	}
	javaFile.Accept()

	return nil
}
