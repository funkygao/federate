package merge

import (
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"federate/pkg/manifest"
)

func TestShowGetBeanRisk(t *testing.T) {
	// Create a temporary directory using t.TempDir()
	tempDir := t.TempDir()

	// Create a temporary Java file with getBean calls
	javaFileContent := `
        public class Test {
            public void test() {
                ctx.getBean("risk");
                ctx.getBean("B.class");
                ctx.igetBean("C.class");
                ctx.getBean(Foo.class);
                ctx.getBeanByType("Foo.class");
                System.out.println();
                ctx.getBean("anotherBean");
            }
        }
    `
	componentName := "foo"
	os.Mkdir(filepath.Join(tempDir, componentName), 0755)
	javaFilePath := filepath.Join(tempDir, componentName, "Test.java")
	err := ioutil.WriteFile(javaFilePath, []byte(javaFileContent), 0644)
	if err != nil {
		t.Fatal(err)
	}

	// 创建一个模拟的 Manifest
	m := &manifest.Manifest{
		Components: []manifest.ComponentInfo{
			{
				Name:    componentName,
				BaseDir: tempDir,
			},
		},
	}

	// Create an XmlBeanManager instance
	xmlBeanManager := NewXmlBeanManager(m)

	// Capture log output
	var logOutput strings.Builder
	log.SetOutput(&logOutput)

	// Run the showGetBeanRisk function
	xmlBeanManager.showGetBeanRisk()

	// Check log output
	expectedLogs := []string{
		"getBean(risk)",
		"getBean(B.class)",
		"getBean(anotherBean)",
	}

	for _, expectedLog := range expectedLogs {
		if !strings.Contains(logOutput.String(), expectedLog) {
			t.Errorf("Expected log output to contain: %s", expectedLog)
		}
	}
}
