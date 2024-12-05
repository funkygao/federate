package similarity

import (
	"context"

	"federate/pkg/code"
)

type BandHash uint16

type Indexer struct {
	Buckets map[BandHash][]*code.JavaFile
}

func NewIndexer() *Indexer {
	return &Indexer{
		Buckets: make(map[BandHash][]*code.JavaFile),
	}
}

func (l *Indexer) Insert(jf *code.JavaFile) {
	// 对于每个文档，我们计算其 SimHash 值：特征，并缓存
	simhash := simHash(jf.CompactCode())
	jf.Context = context.WithValue(context.Background(), simhashCacheKey, simhash)

	// 文档分桶过程
	for i := 0; i < NumBands; i++ {
		// 对每个 band，我们使用其 16 位值作为一个 bucket 的 key
		bandHash := l.bandHash(simhash, i)

		// 文档被添加到这个 bucket 中
		l.Buckets[bandHash] = append(l.Buckets[bandHash], jf)
	}
}

// GetCandidates 查找可能相似的文档
func (l *Indexer) GetCandidates(simhash uint64) []*code.JavaFile {
	uniqueCandidates := make(map[*code.JavaFile]struct{})
	// 如果两个文档在任何一个 band 上完全匹配，它们就会在至少一个 bucket 中相遇
	for i := 0; i < NumBands; i++ {
		bandHash := l.bandHash(simhash, i)
		for _, jf := range l.Buckets[bandHash] {
			uniqueCandidates[jf] = struct{}{}
		}
	}

	// 结果转换
	result := make([]*code.JavaFile, 0, len(uniqueCandidates))
	for jf := range uniqueCandidates {
		result = append(result, jf)
	}
	return result
}

func (l *Indexer) bandHash(simhash uint64, i int) BandHash {
	return BandHash((simhash >> uint(i*BandBits)) & ((1 << BandBits) - 1))
}
