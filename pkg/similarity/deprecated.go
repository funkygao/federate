package similarity

import (
	"strings"
)

type FileSimilarityFunc func(f1, f2 string) (float64, error)

var simFuncs = map[string]FileSimilarityFunc{
	"simhash": simHashFileSimilarity,
	"jaccard": jaccardFileSimilarity,
}

func BetweenFiles(f1, f2 string, algo string) (float64, error) {
	return simFuncs[algo](f1, f2)
}

func simHashFileSimilarity(file1, file2 string) (float64, error) {
	jf1, jf2, err := loadJavaFiles(file1, file2)
	if err != nil {
		return 0, err
	}

	return simHashJavaFileSimilarity(jf1, jf2), nil
}

func jaccardFileSimilarity(file1, file2 string) (float64, error) {
	jf1, jf2, err := loadJavaFiles(file1, file2)
	if err != nil {
		return 0, err
	}

	words1 := strings.Fields(jf1.Content())
	words2 := strings.Fields(jf2.Content())

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
