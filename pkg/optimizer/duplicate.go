package optimizer

import (
	"os"
	"path/filepath"
	"strings"

	"federate/pkg/manifest"
	"federate/pkg/similarity"
)

type DupJavaInfo struct {
	ClassName  string
	Paths      []string
	Similarity float64
}

// DetectDuplicateJava detects highly similar java files across components.
func DetectDuplicateJava(manifest *manifest.Manifest) ([]DupJavaInfo, error) {
	classMap := make(map[string][]string)

	for _, component := range manifest.Components {
		err := filepath.Walk(component.Name, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if info.IsDir() && info.Name() == "test" {
				return filepath.SkipDir
			}
			if !info.IsDir() && strings.HasSuffix(info.Name(), ".java") {
				className := strings.TrimSuffix(info.Name(), ".java")
				classMap[className] = append(classMap[className], path)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}

	var dups []DupJavaInfo
	for className, paths := range classMap {
		if len(paths) > 1 {
			similarityScore, err := similarity.BetweenFiles(paths)
			if err != nil {
				return nil, err
			}
			dups = append(dups, DupJavaInfo{
				ClassName:  className,
				Paths:      paths,
				Similarity: similarityScore,
			})
		}
	}

	if len(dups) == 0 {
		return nil, nil
	}

	return dups, nil
}
