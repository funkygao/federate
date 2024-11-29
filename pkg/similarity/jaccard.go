package similarity

import (
	"io/ioutil"
	"strings"
)

func jaccardAvg(paths []string) (float64, error) {
	if len(paths) < 2 {
		return 0, nil
	}

	totalSimilarity := 0.0
	count := 0

	for i := 0; i < len(paths); i++ {
		for j := i + 1; j < len(paths); j++ {
			similarityScore, err := jaccard(paths[i], paths[j])
			if err != nil {
				return 0, err
			}
			totalSimilarity += similarityScore
			count++
		}
	}

	return totalSimilarity / float64(count), nil
}

func jaccard(file1, file2 string) (float64, error) {
	content1, err := ioutil.ReadFile(file1)
	if err != nil {
		return 0, err
	}

	content2, err := ioutil.ReadFile(file2)
	if err != nil {
		return 0, err
	}

	words1 := strings.Fields(string(content1))
	words2 := strings.Fields(string(content2))

	set1 := make(map[string]struct{})
	set2 := make(map[string]struct{})

	for _, word := range words1 {
		set1[word] = struct{}{}
	}

	for _, word := range words2 {
		set2[word] = struct{}{}
	}

	intersection := 0
	for word := range set1 {
		if _, found := set2[word]; found {
			intersection++
		}
	}

	union := len(set1) + len(set2) - intersection

	if union == 0 {
		return 1, nil
	}

	return float64(intersection) / float64(union), nil
}
