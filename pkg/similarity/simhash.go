package similarity

import (
	"context"
	"crypto/md5"

	"federate/internal/hacking"
	"federate/pkg/code"
)

func simHashJavaFileSimilarity(jf1, jf2 *code.JavaFile) float64 {
	feature1, ok := jf1.Context.Value(simhashCacheKey).(uint64)
	if !ok {
		feature1 = simHash(jf1.CompactCode())
		jf1.Context = context.WithValue(context.Background(), simhashCacheKey, feature1)
	}
	feature2, ok := jf2.Context.Value(simhashCacheKey).(uint64)
	if !ok {
		feature2 := simHash(jf2.CompactCode())
		jf2.Context = context.WithValue(context.Background(), simhashCacheKey, feature2)
	}

	return hammingSimilarity(feature1, feature2)
}

func simHash(doc string) (feature uint64) {
	var vector [SimHashBits]int64

	for i := 0; i < len(doc)-NGramLength; i++ {
		processNgram(doc[i:i+NGramLength], &vector)
	}

	for i := 0; i < SimHashBits; i++ {
		if vector[i] > 0 {
			feature |= 1 << uint(i)
		}
	}

	return
}

func processNgram(ngram string, vectorCounts *[SimHashBits]int64) {
	ngramHash := md5.Sum(hacking.S2b(ngram))

	for bitIndex := 0; bitIndex < SimHashBits; bitIndex++ {
		if isBitSet(ngramHash, bitIndex) {
			vectorCounts[bitIndex]++
		} else {
			vectorCounts[bitIndex]--
		}
	}
}

func isBitSet(hash [16]byte, bitIndex int) bool {
	byteIndex := bitIndex / 8
	bitOffset := uint(bitIndex % 8)
	return hash[byteIndex]&(1<<bitOffset) != 0
}

func hammingSimilarity(hash1, hash2 uint64) float64 {
	distance := hammingDistance(hash1, hash2)
	similarity := 1 - float64(distance)/SimHashBits
	return similarity
}
