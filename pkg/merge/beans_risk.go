package merge

import (
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"time"

	"federate/pkg/java"
)

type getBeanRisk struct {
	filePath string
	beanName string
}

func (b *XmlBeanManager) showRisk() {
	if b.plan.HasConflict() {
		log.Printf("⚠️  Detecting getBean(name) risks caused by Bean Id Reconcilation ...")
		if err := b.showGetBeanRisk(); err != nil {
			log.Fatalf("%v", err)
		}
	}
}

func (b *XmlBeanManager) showGetBeanRisk() error {
	getBeanExp := regexp.MustCompile(`\bgetBean\s*\(\s*"([^"]+)"\s*\)`)
	risks := []getBeanRisk{}
	t0 := time.Now()
	n := 0
	for _, component := range b.m.Components {
		err := filepath.Walk(component.RootDir(), func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if java.IsJavaMainSource(info, path) {
				n++
				names, err := findGetBeanNames(path, getBeanExp)
				if err != nil {
					return err
				}
				for _, name := range names {
					risks = append(risks, getBeanRisk{filePath: path, beanName: name})
				}
			}
			return nil
		})
		if err != nil {
			return err
		}
	}
	log.Printf("⚠️  getBean(name) %d risks, %d java files analyzed, cost %s", len(risks), n, time.Since(t0))
	for _, risk := range risks {
		log.Printf("getBean(%s): %s", risk.beanName, risk.filePath)
	}
	return nil
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
