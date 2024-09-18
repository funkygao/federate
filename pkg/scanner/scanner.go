package scanner

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"federate/pkg/manifest"
	"federate/pkg/similarity"
)

type DupInfo struct {
	ClassName  string
	Paths      []string
	Similarity float64
}

// 定义需要忽略的路径特征
var ignoredPaths = []string{
	"src/test",
}

func ScanComponents(manifest *manifest.Manifest) (map[string][]string, error) {
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

	return classMap, nil
}

func CheckDup(classMap map[string][]string) ([]DupInfo, error) {
	var dups []DupInfo

	for className, paths := range classMap {
		if len(paths) > 1 {
			filteredPaths := filterIgnoredPaths(paths)
			if len(filteredPaths) > 1 {
				similarityScore, err := similarity.CalculateAverageSimilarity(filteredPaths)
				if err != nil {
					return nil, err
				}
				dups = append(dups, DupInfo{
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

	return dups, fmt.Errorf("duplicates detected")
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
