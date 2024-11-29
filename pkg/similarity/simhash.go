package similarity

import (
	"io/ioutil"
	"math/bits"

	"github.com/mfonda/simhash"
)

func simhashAvg(paths []string) (float64, error) {
	if len(paths) < 2 {
		return 0, nil
	}

	hashes := make([]uint64, len(paths))
	for i, path := range paths {
		content, err := ioutil.ReadFile(path)
		if err != nil {
			return 0, err
		}
		hashes[i] = simhash.Simhash(simhash.NewWordFeatureSet(content))
	}

	totalSimilarity := 0.0
	count := 0

	for i := 0; i < len(hashes); i++ {
		for j := i + 1; j < len(hashes); j++ {
			similarity := calculateSimhashSimilarity(hashes[i], hashes[j])
			totalSimilarity += similarity
			count++
		}
	}

	return totalSimilarity / float64(count), nil
}

func calculateSimhashSimilarity(hash1, hash2 uint64) float64 {
	distance := bits.OnesCount64(hash1 ^ hash2)
	return 1 - float64(distance)/64
}
