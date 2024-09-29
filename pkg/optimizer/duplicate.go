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

// 定义需要忽略的路径特征
var ignoredPaths = []string{
	"src/test",
}

// DetectDuplicateJava detects highly similar java files across components.
func DetectDuplicateJava(manifest *manifest.Manifest) ([]DupJavaInfo, error) {
	classMap := make(map[string][]string)

	for _, component := range manifest.Components {
		err := filepath.Walk(component.Name, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
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

	return checkDup(classMap)
}

func checkDup(classMap map[string][]string) ([]DupJavaInfo, error) {
	var dups []DupJavaInfo

	for className, paths := range classMap {
		if len(paths) > 1 {
			filteredPaths := filterIgnoredPaths(paths)
			if len(filteredPaths) > 1 {
				similarityScore, err := similarity.CalculateAverageSimilarity(filteredPaths)
				if err != nil {
					return nil, err
				}
				dups = append(dups, DupJavaInfo{
					ClassName:  className,
					Paths:      filteredPaths,
					Similarity: similarityScore,
				})
			}
		}
	}

	if len(dups) == 0 {
		return nil, nil
	}

	return dups, nil
}

func filterIgnoredPaths(paths []string) []string {
	var filteredPaths []string
	for _, path := range paths {
		ignore := false
		for _, ignoredPath := range ignoredPaths {
			if strings.Contains(path, ignoredPath) {
				ignore = true
				break
			}
		}
		if !ignore {
			filteredPaths = append(filteredPaths, path)
		}
	}
	return filteredPaths
}
