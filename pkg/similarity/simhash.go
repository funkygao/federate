package similarity

import (
	"crypto/md5"
	"math/bits"

	"federate/pkg/code"
)

const (
	// 特征向量的位数，通常使用64位
	SimHashBits = 64
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
	var vector [SimHashBits]int

	// 对内容进行滑动窗口处理,生成trigram
	for i := 0; i < len(content)-3; i++ {
		trigram := content[i : i+3]
		// 对每个trigram计算hash值
		hash := md5.Sum([]byte(trigram))
		for j := 0; j < SimHashBits; j++ {
			// 检查 hash 的第 j 位是否为 1
			// hash[j/8] 选择字节，(1<<(uint(j)%8)) 创建掩码
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
			// 1 << uint(i) 创建一个只有第 i 位为 1 的掩码
			// |= 操作将这一位设置到 feature 中
			feature |= 1 << uint(i)
		}
	}

	return
}

func simHashSimilarity(hash1, hash2 uint64) float64 {
	// 汉明距离
	hammingDistance := bits.OnesCount64(hash1 ^ hash2)
	similarity := 1 - float64(hammingDistance)/SimHashBits
	return similarity
}
