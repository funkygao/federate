package microservice

import (
	"fmt"
	"io/ioutil"
	"os"

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
	markdown += "This document provides a detailed description of the manifest.yaml file used in our project.\n\n"
	markdown += "## Structure\n\n"
	markdown += "The manifest.yaml file has the following top-level fields:\n\n"

	// Add top-level fields
	for key := range manifest {
		markdown += fmt.Sprintf("- %s\n", key)
	}
	markdown += "\n"

	markdown += "## Fields\n\n"

	markdown += generateMarkdownForValue("", manifest)

	return markdown, nil
}

func generateMarkdownForValue(prefix string, v interface{}) string {
	var markdown string

	switch val := v.(type) {
	case map[interface{}]interface{}:
		for k, v := range val {
			newPrefix := prefix
			if newPrefix != "" {
				newPrefix += "."
			}
			newPrefix += fmt.Sprintf("%v", k)
			markdown += generateMarkdownForValue(newPrefix, v)
		}
	case map[string]interface{}:
		for k, v := range val {
			newPrefix := prefix
			if newPrefix != "" {
				newPrefix += "."
			}
			newPrefix += k
			markdown += generateMarkdownForValue(newPrefix, v)
		}
	case []interface{}:
		for i, item := range val {
			newPrefix := fmt.Sprintf("%s[%d]", prefix, i)
			markdown += generateMarkdownForValue(newPrefix, item)
		}
	default:
		markdown += fmt.Sprintf("### %s\n\n**Type**: %s\n\n**Value**: `%v`\n\n", prefix, getType(v), v)
	}

	return markdown
}

func getType(v interface{}) string {
	switch v.(type) {
	case bool:
		return "Boolean"
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return "Integer"
	case float32, float64:
		return "Number"
	case string:
		return "String"
	case []interface{}:
		return "Array"
	case map[string]interface{}, map[interface{}]interface{}:
		return "Object"
	default:
		return fmt.Sprintf("Unknown (%T)", v)
	}
}

func init() {
	manifestCmd.Flags().BoolVarP(&generateGuide, "guide", "g", false, "Generate a reference guide for the manifest")
	manifestCmd.Flags().StringVarP(&outputFile, "output", "o", "", "Output file for the generated guide (default is stdout)")
}
