package similarity

type SimilarityFunc func(paths []string) (float64, error)

var distanceFuncs = map[string]SimilarityFunc{
	"simhash": simhashAvg,
	"jaccard": jaccardAvg,
}

func BetweenFiles(paths []string, algo string) (float64, error) {
	return distanceFuncs[algo](paths)
}
