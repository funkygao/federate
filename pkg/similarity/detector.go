package similarity

import (
	"fmt"
	"io/ioutil"
	"log"
	"path/filepath"
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
	var totalFiles int32

	// 并行获取文件，解析到 []JavaFile
	for _, c := range d.m.Components {
		wg.Add(1)
		go func(c manifest.ComponentInfo) {
			defer wg.Done()

			files, err := java.ListJavaMainSourceFiles(c.RootDir())
			if err != nil {
				log.Printf("Error listing files for component %s: %v", c.Name, err)
				return
			}

			atomic.AddInt32(&totalFiles, int32(len(files)))

			var javaFiles []*code.JavaFile
			for _, path := range files {
				className := strings.Trim(filepath.Base(path), ".java")
				if d.ignoreRegex.MatchString(className) {
					continue
				}

				content, err := ioutil.ReadFile(path)
				if err != nil {
					log.Printf("Error reading file %s: %v", path, err)
					continue
				}

				jf := code.NewJavaFile(path, &c, content)
				if len(jf.CompactCode()) > 100 {
					javaFiles = append(javaFiles, jf)
				}
			}

			componentFiles.Store(c.Name, javaFiles)
		}(c)
	}

	wg.Wait()

	// 批量插入索引
	componentFiles.Range(func(key, value interface{}) bool {
		javaFiles := value.([]*code.JavaFile)
		d.Indexer.BatchInsert(javaFiles)
		return true
	})

	log.Printf("%d Java Files Parsed, Buckets: %d, cost %s", totalFiles, len(d.Indexer.Buckets), time.Since(t0))

	t0 = time.Now()
	processed := primitive.NewStringSet()
	var duplicates []DuplicatePair
	var ops int32

	// sort, so that we compare in 1 direction: avoid duplicate ops
	components := make([]string, 0)
	componentFiles.Range(func(key, _ interface{}) bool {
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
			jfs := value.([]*code.JavaFile)
			for _, jf1 := range jfs {
				simhash1 := jf1.Context.Value(simhashCacheKey).(uint64)
				for _, jf2 := range d.Indexer.GetCandidates(simhash1) {
					atomic.AddInt32(&ops, 1)

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

	log.Printf("Total ops: %d, cost %s", ops, time.Since(t0))

	return duplicates, nil
}
