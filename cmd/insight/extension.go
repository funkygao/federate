package insight

import (
	"fmt"
	"io/ioutil"
	"log"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"federate/internal/hack"
	"federate/pkg/java"
	"federate/pkg/tabular"
	"federate/pkg/util"
	"github.com/spf13/cobra"
)

var (
	interfaceRegex = regexp.MustCompile(`(?s)(?:public\s+)?interface\s+(\w+)\s+extends\s+IDomainExtension`)
	commentRegex   = regexp.MustCompile(`@\w+(?:\([^)]*\))?`)
)

type Method struct {
	Name    string
	Javadoc string
	Dir     string
}

var extensionCmd = &cobra.Command{
	Use:   "extension [dir]",
	Short: "Analyze Java extension points in the specified directory",
	Run: func(cmd *cobra.Command, args []string) {
		root := "."
		if len(args) > 0 {
			root = args[0]
		}
		analyzeExtensions(root)
	},
}

func analyzeExtensions(root string) {
	subdirs, err := util.GetSubdirectories(root)
	if err != nil {
		log.Fatalf("Error getting subdirectories: %v", err)
	}

	var wg sync.WaitGroup
	extensionsChan := make(chan map[string][]Method, len(subdirs))

	for _, subdir := range subdirs {
		wg.Add(1)
		go func(dir string) {
			defer wg.Done()
			extensions := analyzeDirectory(dir)
			extensionsChan <- extensions
		}(subdir)
	}

	wg.Wait()
	close(extensionsChan)

	allExtensions := make(map[string][]Method)
	for extensions := range extensionsChan {
		for k, v := range extensions {
			allExtensions[k] = v
		}
	}

	printExtensionAnalysis(allExtensions)
}

func analyzeDirectory(dir string) map[string][]Method {
	log.Printf("Scan %s ...", dir)

	fileChan, errChan := java.ListJavaMainSourceFilesAsync(dir)
	extensions := make(map[string][]Method)

	for file := range fileChan {
		content, err := ioutil.ReadFile(file.Path)
		if err != nil {
			log.Fatalf("%v", err)
		}

		code := hack.B2s(content)
		if interfaceRegex.MatchString(code) {
			interfaceName := filepath.Base(file.Path)
			interfaceName = strings.TrimSuffix(interfaceName, ".java")
			methods := extractMethods(file.Path, code, dir)
			if methods != nil {
				extensions[interfaceName] = methods
			} else {
				log.Printf("%s has no methods", interfaceName)
			}
		}
	}

	if err := <-errChan; err != nil {
		log.Printf("Error in directory %s: %v", dir, err)
	}

	return extensions
}

func extractMethods(filePath, code, dir string) []Method {
	lines := strings.Split(code, "\n")
	var methods []Method
	var currentJavadoc []string
	inInterface := false
	inJavadoc := false

	for _, line := range lines {
		line = strings.TrimSpace(line)

		if strings.Contains(line, "interface") && strings.Contains(line, "extends IDomainExtension") {
			inInterface = true
			continue
		}

		if !inInterface {
			continue
		}

		if strings.HasPrefix(line, "/**") {
			inJavadoc = true
			currentJavadoc = []string{}
			continue
		}

		if inJavadoc {
			if strings.HasPrefix(line, "*/") {
				inJavadoc = false
			} else {
				trimmedLine := strings.TrimPrefix(line, "*")
				trimmedLine = strings.TrimSpace(trimmedLine)
				if trimmedLine != "" && !strings.HasPrefix(trimmedLine, "@") {
					currentJavadoc = append(currentJavadoc, trimmedLine)
				}
			}
			continue
		}

		if strings.HasPrefix(line, "}") {
			break
		}

		if strings.Contains(line, "(") && strings.Contains(line, ")") && strings.HasSuffix(line, ";") {
			methodName := extractMethodName(line)
			if methodName != "" {
				javadoc := ""
				if len(currentJavadoc) > 0 {
					javadoc = currentJavadoc[0]
				}
				methods = append(methods, Method{Name: methodName, Javadoc: javadoc, Dir: dir})
				currentJavadoc = []string{}
			}
		}
	}

	if len(methods) == 0 {
		log.Printf("No methods found in interface in file: %s", filePath)
	}

	return methods
}

func extractMethodName(line string) string {
	// 移除注解
	line = commentRegex.ReplaceAllString(line, "")

	// 查找方法名
	parts := strings.Split(line, "(")
	if len(parts) < 2 {
		return ""
	}

	methodPart := strings.TrimSpace(parts[0])
	words := strings.Fields(methodPart)
	if len(words) == 0 {
		return ""
	}

	return words[len(words)-1]
}

func printExtensionAnalysis(extensions map[string][]Method) {
	var data [][]string
	totalMethods := 0
	for interfaceName, methods := range extensions {
		for _, method := range methods {
			data = append(data, []string{method.Dir, interfaceName, util.Truncate(method.Name+" "+method.Javadoc, 50)})
			totalMethods++
		}
	}

	header := []string{"模块", "扩展点接口", "具体方法"}
	tabular.Display(header, data, true, 0)
	fmt.Printf("Total Interfaces: %d, Total Methods: %d\n", len(extensions), totalMethods)
}
