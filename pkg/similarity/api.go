package similarity

type FileSimilarityFunc func(f1, f2 string) (float64, error)

var simFuncs = map[string]FileSimilarityFunc{
	"simhash": simHashFileSimilarity,
	"jaccard": jaccardFileSimilarity,
}

func BetweenFiles(f1, f2 string, algo string) (float64, error) {
	return simFuncs[algo](f1, f2)
}
