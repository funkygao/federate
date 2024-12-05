package similarity

import (
	"context"
	"runtime"
	"sync"
	"sync/atomic"

	"federate/pkg/code"
)

type BandHash uint16

type Indexer struct {
	Buckets map[BandHash][]*code.JavaFile
	mu      sync.RWMutex
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

	l.mu.Lock()
	defer l.mu.Unlock()

	// 文档分桶过程
	for i := 0; i < NumBands; i++ {
		bandHash := l.bandHash(simhash, i)
		l.Buckets[bandHash] = append(l.Buckets[bandHash], jf)
	}
}

func (l *Indexer) BatchInsert(jfs []*code.JavaFile) {
	type result struct {
		jf      *code.JavaFile
		simhash uint64
	}

	numWorkers := runtime.NumCPU()
	resultChan := make(chan result, len(jfs))
	workChan := make(chan *code.JavaFile, len(jfs))
	remainingTasks := int32(len(jfs))

	// 启动工作者
	for i := 0; i < numWorkers; i++ {
		go func() {
			for jf := range workChan {
				resultChan <- result{jf, simHash(jf.CompactCode())} // expensive

				if atomic.AddInt32(&remainingTasks, -1) == 0 {
					close(resultChan)
				}
			}
		}()
	}

	// 发送工作内容
	for _, jf := range jfs {
		workChan <- jf
	}
	close(workChan)

	// 处理结果并插入到索引中
	l.mu.Lock()
	defer l.mu.Unlock()

	for r := range resultChan {
		jf, simhash := r.jf, r.simhash
		jf.Context = context.WithValue(context.Background(), simhashCacheKey, simhash)

		for i := 0; i < NumBands; i++ {
			bandHash := l.bandHash(simhash, i)
			l.Buckets[bandHash] = append(l.Buckets[bandHash], jf)
		}
	}
}

func (l *Indexer) GetCandidates(simhash uint64) []*code.JavaFile {
	uniqueCandidates := make(map[*code.JavaFile]struct{})

	l.mu.RLock()
	defer l.mu.RUnlock()

	for i := 0; i < NumBands; i++ {
		bandHash := l.bandHash(simhash, i)
		for _, jf := range l.Buckets[bandHash] {
			uniqueCandidates[jf] = struct{}{}
		}
	}

	result := make([]*code.JavaFile, 0, len(uniqueCandidates))
	for jf := range uniqueCandidates {
		result = append(result, jf)
	}
	return result
}

func (l *Indexer) bandHash(simhash uint64, i int) BandHash {
	return BandHash((simhash >> uint(i*BandBits)) & ((1 << BandBits) - 1))
}
