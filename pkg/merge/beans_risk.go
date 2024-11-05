package merge

import (
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sync"
	"time"

	"federate/pkg/concurrent"
	"federate/pkg/java"
	"federate/pkg/manifest"
	"github.com/fatih/color"
)

type getBeanRisk struct {
	filePath string
	beanName string
}

func (b *XmlBeanManager) showRisk() {
	if b.plan.HasConflict() {
		log.Printf("Detecting getBean(name) risks caused by Bean Id Rewrite ...")
		if err := b.showGetBeanRisk(); err != nil {
			log.Fatalf("%v", err)
		}
	}
}

func (b *XmlBeanManager) showGetBeanRisk() error {
	getBeanExp := regexp.MustCompile(`\bgetBean\s*\(\s*"([^"]+)"\s*\)`)
	var risks []getBeanRisk
	var mu sync.Mutex
	counter := concurrent.NewCounter()

	t0 := time.Now()

	executor := concurrent.NewParallelExecutor(runtime.NumCPU())

	for _, component := range b.m.Components {
		executor.AddTask(&getBeanRiskTask{
			component:  component,
			getBeanExp: getBeanExp,
			risks:      &risks,
			mu:         &mu,
			counter:    counter,
		})
	}

	errors := executor.Execute()
	if len(errors) > 0 {
		return errors[0] // 返回第一个遇到的错误
	}

	if len(risks) > 0 {
		color.Yellow("getBean(name) %d risks, %d java files analyzed, cost %s", len(risks), counter.Value(), time.Since(t0))
		for _, risk := range risks {
			log.Printf("getBean(%s): %s", risk.beanName, risk.filePath)
		}
	}
	return nil
}

type getBeanRiskTask struct {
	component  manifest.ComponentInfo
	getBeanExp *regexp.Regexp
	risks      *[]getBeanRisk
	mu         *sync.Mutex
	counter    *concurrent.Counter
}

func (t *getBeanRiskTask) Execute() error {
	return filepath.Walk(t.component.RootDir(), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if java.IsJavaMainSource(info, path) {
			t.counter.Increment()
			names, err := findGetBeanNames(path, t.getBeanExp)
			if err != nil {
				return err
			}
			if len(names) > 0 {
				t.mu.Lock()
				for _, name := range names {
					*t.risks = append(*t.risks, getBeanRisk{filePath: path, beanName: name})
				}
				t.mu.Unlock()
			}
		}
		return nil
	})
}

func findGetBeanNames(filePath string, re *regexp.Regexp) ([]string, error) {
	var beanNames []string
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	matches := re.FindAllStringSubmatch(string(content), -1)
	for _, match := range matches {
		if len(match) > 1 {
			name := match[1] // 提取参数 name
			beanNames = append(beanNames, name)
		}
	}
	return beanNames, nil
}
