package similarity

import (
	"fmt"
	"io/ioutil"
	"log"
	"regexp"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"federate/pkg/code"
	"federate/pkg/java"
	"federate/pkg/manifest"
	"federate/pkg/primitive"
)

type Detector struct {
	m         *manifest.Manifest
	threshold float64

	algo  string
	algos map[string]func() ([]DuplicatePair, error)

	ignoreRegex *regexp.Regexp

	Indexer *Indexer

	TotalFiles int32
	RecallOps  int32
	Phase1     time.Duration
	Phase2     time.Duration
	Phase3     time.Duration
}

type DuplicatePair struct {
	File1      string  `json:"file1"`
	File2      string  `json:"file2"`
	Similarity float64 `json:"similarity"`
}

func NewDetector(m *manifest.Manifest, threshold float64, algo string) *Detector {
	d := &Detector{
		m:           m,
		threshold:   threshold,
		algo:        algo,
		ignoreRegex: regexp.MustCompile(`(Dto|DTO|Query|Mapper|Request|Response|Result|Vo|Dao|Po|Detail|Param)`),
		Indexer:     NewIndexer(),
	}
	d.algos = map[string]func() ([]DuplicatePair, error){
		"simhash": d.simhashDetect,
	}
	return d
}

func (d *Detector) Detect() ([]DuplicatePair, error) {
	dt, ok := d.algos[d.algo]
	if !ok {
		return nil, fmt.Errorf("Not Implemented: %s", d.algo)
	}

	return dt()
}

func (d *Detector) simhashDetect() ([]DuplicatePair, error) {
	t0 := time.Now()

	componentFiles := sync.Map{}
	var wg sync.WaitGroup

	// 并行获取文件，解析到 []JavaFile
	for _, c := range d.m.Components {
		wg.Add(1)
		go func(c manifest.ComponentInfo) {
			defer wg.Done()

			fileChan, _ := java.ListJavaMainSourceFilesAsync(c.RootDir())

			var javaFiles []*code.JavaFile
			for fileInfo := range fileChan {
				atomic.AddInt32(&d.TotalFiles, 1)

				className := strings.Trim(fileInfo.Info.Name(), ".java")
				if d.ignoreRegex.MatchString(className) {
					continue
				}

				content, err := ioutil.ReadFile(fileInfo.Path)
				if err != nil {
					log.Fatalf("Error reading file %s: %v", fileInfo.Path, err)
				}

				jf := code.NewJavaFile(fileInfo.Path, &c, content)
				if len(jf.CompactCode()) > 100 {
					javaFiles = append(javaFiles, jf)
				}
			}

			componentFiles.Store(c.Name, javaFiles)
		}(c)
	}

	wg.Wait()
	d.Phase1 = time.Since(t0)

	// 批量插入索引
	t0 = time.Now()
	componentFiles.Range(func(key, value any) bool {
		javaFiles := value.([]*code.JavaFile)
		d.Indexer.BatchInsert(javaFiles)
		return true
	})
	d.Phase2 = time.Since(t0)

	t0 = time.Now()
	processed := primitive.NewStringSet()
	var duplicates []DuplicatePair

	// sort, so that we compare in 1 direction: avoid duplicate ops
	components := make([]string, 0)
	componentFiles.Range(func(key, _ any) bool {
		components = append(components, key.(string))
		return true
	})
	sort.Strings(components)

	// 并行计算相似度
	var mu sync.Mutex
	for _, c1 := range components {
		wg.Add(1)
		go func(c1 string) {
			defer wg.Done()

			value, _ := componentFiles.Load(c1)
			jfs1 := value.([]*code.JavaFile)
			for _, jf1 := range jfs1 {
				simhash1 := jf1.Context.Value(simhashCacheKey).(uint64)
				for _, jf2 := range d.Indexer.GetCandidates(simhash1) {
					atomic.AddInt32(&d.RecallOps, 1)

					if jf1.Path() == jf2.Path() {
						continue
					}

					if c1 >= jf2.ComponentName() { // 只比较字典序更大的组件
						continue
					}

					processedKey := jf1.Path() + ":" + jf2.Path()
					if !processed.Contains(processedKey) {
						if sim := simHashJavaFileSimilarity(jf1, jf2); sim >= d.threshold {
							mu.Lock()
							duplicates = append(duplicates, DuplicatePair{
								File1:      jf1.Path(),
								File2:      jf2.Path(),
								Similarity: sim,
							})
							mu.Unlock()
						}
						processed.Add(processedKey)
					}
				}
			}
		}(c1)
	}

	wg.Wait()
	d.Phase3 = time.Since(t0)

	return duplicates, nil
}
