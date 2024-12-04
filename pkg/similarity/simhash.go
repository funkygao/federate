package similarity

import (
	"crypto/md5"
	"math/bits"

	"federate/pkg/code"
)

const (
	SimHashSize = 64
)

func simHashFileSimilarity(file1, file2 string) (float64, error) {
	jf1, jf2, err := loadJavaFiles(file1, file2)
	if err != nil {
		return 0, err
	}

	return simHashJavaFileSimilarity(jf1, jf2), nil
}

func simHashJavaFileSimilarity(jf1, jf2 *code.JavaFile) float64 {
	feature1 := simHash(jf1.CompactCode())
	feature2 := simHash(jf2.CompactCode())

	return simHashSimilarity(feature1, feature2)
}

// simhash is a LSH: Locality-Sensitive Hashing.
func simHash(content string) (feature uint64) {
	var vector [SimHashSize]int

	for i := 0; i < len(content)-3; i++ {
		trigram := content[i : i+3]
		hash := md5.Sum([]byte(trigram))
		for j := 0; j < SimHashSize; j++ {
			if hash[j/8]&(1<<(uint(j)%8)) != 0 {
				vector[j]++
			} else {
				vector[j]--
			}
		}
	}

	for i := 0; i < SimHashSize; i++ {
		if vector[i] > 0 {
			feature |= 1 << uint(i)
		}
	}

	return
}

func simHashSimilarity(hash1, hash2 uint64) float64 {
	hammingDistance := bits.OnesCount64(hash1 ^ hash2)
	similarity := 1 - float64(hammingDistance)/SimHashSize
	return similarity
}
