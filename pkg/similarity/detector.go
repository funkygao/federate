package similarity

import (
	"io/ioutil"
	"log"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"federate/pkg/code"
	"federate/pkg/java"
	"federate/pkg/manifest"
	"federate/pkg/primitive"
)

type Detector struct {
	m         *manifest.Manifest
	threshold float64
	algo      string

	ignoreRegex *regexp.Regexp

	LSH *LSH
}

type DuplicatePair struct {
	File1      string  `json:"file1"`
	File2      string  `json:"file2"`
	Similarity float64 `json:"similarity"`
}

func NewDetector(m *manifest.Manifest, threshold float64, algo string) *Detector {
	return &Detector{
		m:           m,
		threshold:   threshold,
		algo:        algo,
		ignoreRegex: regexp.MustCompile(`(Dto|DTO|Query|Mapper|Request|Response|Result|Vo|Dao|Po|Detail|Param)`),
		LSH:         NewLSH(),
	}
}

func (d *Detector) Detect() ([]DuplicatePair, error) {
	t0 := time.Now()

	var totalFiles int
	componentFiles := make(map[string][]*code.JavaFile)
	for _, c := range d.m.Components {
		files, err := java.ListJavaMainSourceFiles(c.RootDir())
		if err != nil {
			return nil, err
		}

		totalFiles += len(files)

		if componentFiles[c.Name] == nil {
			componentFiles[c.Name] = make([]*code.JavaFile, 0, len(files))
		}

		for _, path := range files {
			className := strings.Trim(filepath.Base(path), ".java")
			if d.ignoreRegex.MatchString(className) {
				continue
			}

			content, err := ioutil.ReadFile(path)
			if err != nil {
				return nil, err
			}

			jf := code.NewJavaFile(path, &c, content)
			compactCode := jf.CompactCode()
			if len(compactCode) > 100 {
				componentFiles[c.Name] = append(componentFiles[c.Name], jf)

				d.LSH.Insert(jf, simHash(compactCode))
			}
		}
	}

	log.Printf("%d Java Files Parsed, LSH Buckets: %d, cost %s", totalFiles, d.LSH.Size(), time.Since(t0))

	t0 = time.Now()
	processed := primitive.NewStringSet()
	var duplicates []DuplicatePair
	var ops int

	// sort, so that we compare in 1 direction: avoid duplicate ops
	components := make([]string, 0, len(componentFiles))
	for c := range componentFiles {
		components = append(components, c)
	}
	sort.Strings(components)

	for _, c1 := range components {
		jfs := componentFiles[c1]
		for _, jf1 := range jfs {
			for _, jf2 := range d.LSH.GetCandidates(simHash(jf1.CompactCode())) { // near neighbors
				ops++

				if jf1.Path() == jf2.Path() { // itself
					continue
				}

				c2 := jf2.ComponentName()
				if c1 >= c2 { // 只比较字典序更大的组件
					continue
				}

				processedKey := jf1.Path() + ":" + jf2.Path()
				if !processed.Contains(processedKey) {
					if sim := simHashJavaFileSimilarity(jf1, jf2); sim >= d.threshold {
						duplicates = append(duplicates, DuplicatePair{
							File1:      jf1.Path(),
							File2:      jf2.Path(),
							Similarity: sim,
						})
					}
					processed.Add(processedKey)
				}
			}
		}

		log.Printf("%s Recalled %d Java Files, ops: %d, cost %s", c1, len(jfs), ops, time.Since(t0))
	}

	return duplicates, nil
}