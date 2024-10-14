package microservice

import (
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"strings"

	"federate/internal/fs"
	"github.com/alecthomas/chroma/formatters"
	"github.com/alecthomas/chroma/lexers"
	"github.com/alecthomas/chroma/styles"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v2"
)

var (
	generateGuide bool
	outputFile    string
)

var manifestCmd = &cobra.Command{
	Use:   "manifest",
	Short: "Display or generate a guide for the manifest.yaml",
	Long:  `The manifest command displays the manifest.yaml with syntax highlighting or generates a detailed reference guide.`,
	Run: func(cmd *cobra.Command, args []string) {
		if generateGuide {
			generateManifestGuide()
		} else {
			showManifest()
		}
	},
}

func showManifest() {
	yaml, _ := fs.FS.ReadFile("templates/manifest.yaml")
	lexer := lexers.Get("yaml")
	iterator, err := lexer.Tokenise(nil, string(yaml))
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	style := styles.Get("pygments")
	formatter := formatters.Get("terminal")
	formatter.Format(os.Stdout, style, iterator)
}

func generateManifestGuide() {
	yaml, _ := fs.FS.ReadFile("templates/manifest.yaml")

	guide, err := doGenerateGuide(yaml)
	if err != nil {
		fmt.Printf("Error generating guide: %v\n", err)
		return
	}

	if outputFile != "" {
		err = ioutil.WriteFile(outputFile, []byte(guide), 0644)
		if err != nil {
			fmt.Printf("Error writing to output file: %v\n", err)
			return
		}
		fmt.Printf("Guide written to %s\n", outputFile)
	} else {
		fmt.Println(guide)
	}
}

func doGenerateGuide(data []byte) (string, error) {
	var manifest map[string]interface{}
	err := yaml.Unmarshal(data, &manifest)
	if err != nil {
		return "", fmt.Errorf("error parsing YAML: %v", err)
	}

	markdown := "# Manifest Reference Guide\n\n"
	markdown += parseYAML("", reflect.ValueOf(manifest), data)

	return markdown, nil
}

func parseYAML(prefix string, v reflect.Value, originalData []byte) string {
	var markdown string

	switch v.Kind() {
	case reflect.Map:
		for _, key := range v.MapKeys() {
			newPrefix := fmt.Sprintf("%s%s.", prefix, key.String())
			markdown += parseYAML(newPrefix, v.MapIndex(key), originalData)
		}
	case reflect.Slice:
		for i := 0; i < v.Len(); i++ {
			newPrefix := fmt.Sprintf("%s[%d].", prefix, i)
			markdown += parseYAML(newPrefix, v.Index(i), originalData)
		}
	default:
		keyName := strings.TrimSuffix(prefix, ".")
		markdown += fmt.Sprintf("## %s\n\n", keyName)
		markdown += fmt.Sprintf("Type: %v\n\n", v.Type())

		comment := getComment(originalData, keyName)
		if comment != "" {
			markdown += fmt.Sprintf("Description: %s\n\n", comment)
		} else {
			markdown += "Description: [Add description here]\n\n"
		}

		markdown += "Example:\n```yaml\n"
		markdown += fmt.Sprintf("%s: %v\n", keyName, v.Interface())
		markdown += "```\n\n"
	}

	return markdown
}

func getComment(data []byte, key string) string {
	lines := strings.Split(string(data), "\n")
	for i, line := range lines {
		if strings.Contains(line, key) {
			if i > 0 && strings.HasPrefix(strings.TrimSpace(lines[i-1]), "#") {
				return strings.TrimSpace(lines[i-1][1:])
			}
			break
		}
	}
	return ""
}

func init() {
	manifestCmd.Flags().BoolVarP(&generateGuide, "guide", "g", false, "Generate a reference guide for the manifest")
	manifestCmd.Flags().StringVarP(&outputFile, "output", "o", "", "Output file for the generated guide (default is stdout)")
}
