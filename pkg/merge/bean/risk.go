package bean

import (
	"io/ioutil"
	"log"
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
	var risks []getBeanRisk
	var mu sync.Mutex
	counter := concurrent.NewCounter()

	t0 := time.Now()

	executor := concurrent.NewParallelExecutor(runtime.NumCPU())

	for _, component := range b.m.Components {
		executor.AddTask(&getBeanRiskTask{
			component: component,
			risks:     &risks,
			mu:        &mu,
			counter:   counter,
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
	component manifest.ComponentInfo
	risks     *[]getBeanRisk
	mu        *sync.Mutex
	counter   *concurrent.Counter
}

func (t *getBeanRiskTask) Execute() error {
	files, err := java.ListJavaMainSourceFiles(t.component.RootDir())
	if err != nil {
		return err
	}
	for _, path := range files {
		t.counter.Increment()
		names, err := findGetBeanNames(path)
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
}

func findGetBeanNames(filePath string) ([]string, error) {
	var beanNames []string
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	matches := getBeanPattern.FindAllStringSubmatch(string(content), -1)
	for _, match := range matches {
		if len(match) > 1 {
			name := match[1] // 提取参数 name
			beanNames = append(beanNames, name)
		}
	}
	return beanNames, nil
}
