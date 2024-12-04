package similarity

import (
	"federate/pkg/code"
)

const (
	NumBands = 4
	BandSize = SimHashSize / NumBands
)

type LSH struct {
	Buckets map[uint64][]*code.JavaFile
}

func NewLSH() *LSH {
	return &LSH{
		Buckets: make(map[uint64][]*code.JavaFile),
	}
}

func (l *LSH) Size() int {
	return len(l.Buckets)
}

func (l *LSH) Insert(jf *code.JavaFile, simhash uint64) {
	for i := 0; i < NumBands; i++ {
		bandHash := (simhash >> uint(i*BandSize)) & ((1 << BandSize) - 1)
		l.Buckets[bandHash] = append(l.Buckets[bandHash], jf)
	}
}

func (l *LSH) GetCandidates(simhash uint64) []*code.JavaFile {
	candidates := make(map[*code.JavaFile]struct{})
	for i := 0; i < NumBands; i++ {
		bandHash := (simhash >> uint(i*BandSize)) & ((1 << BandSize) - 1)
		for _, jf := range l.Buckets[bandHash] {
			candidates[jf] = struct{}{}
		}
	}

	result := make([]*code.JavaFile, 0, len(candidates))
	for jf := range candidates {
		result = append(result, jf)
	}
	return result
}
