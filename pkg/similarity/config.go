package similarity

type SimilarityFunc func(paths []string) (float64, error)

var (
	f SimilarityFunc
)

func UseSimhash() {
	f = simhashAvg
}

func UseJaccard() {
	f = jaccardAvg
}

func BetweenFiles(paths []string) (float64, error) {
	return f(paths)
}
