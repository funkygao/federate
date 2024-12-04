package similarity

import (
	"crypto/md5"
	"math/bits"

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

// LSH: Locality-Sensitive Hashing.
func simHash(content string) (feature uint64) {
	var vector [SimHashBits]int

	// 对内容进行滑动窗口处理，生成trigram
	for i := 0; i < len(content)-3; i++ {
		trigram := content[i : i+3]
		// 对每个trigram计算hash值
		hash := md5.Sum([]byte(trigram))
		for j := 0; j < SimHashBits; j++ {
			// 检查 hash 的第 j 位是否为 1
			if hash[j/8]&(1<<(uint(j)%8)) != 0 {
				vector[j]++
			} else {
				vector[j]--
			}
		}
	}

	for i := 0; i < SimHashBits; i++ {
		if vector[i] > 0 {
			// 如果向量中第 i 位大于 0，则在 feature 的第 i 位设置为 1
			feature |= 1 << uint(i)
		}
	}

	return
}

func hammingSimilarity(hash1, hash2 uint64) float64 {
	// 汉明距离
	hammingDistance := bits.OnesCount64(hash1 ^ hash2)
	similarity := 1 - float64(hammingDistance)/SimHashBits
	return similarity
}
