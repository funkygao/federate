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
	markdown += "This document provides a detailed description of the manifest.yaml file used in our project.\n\n"
	markdown += "## Structure\n\n"
	markdown += "The manifest.yaml file has the following top-level fields:\n\n"

	for key := range manifest {
		markdown += fmt.Sprintf("- [%s](#%s)\n", key, strings.ToLower(key))
	}
	markdown += "\n"

	markdown += generateMarkdownForField("", reflect.ValueOf(manifest), data, 0)

	return markdown, nil
}

func generateMarkdownForField(fieldPath string, v reflect.Value, originalData []byte, depth int) string {
	var markdown string
	indent := strings.Repeat("  ", depth)
	fieldName := getLastPathComponent(fieldPath)

	if fieldName == "" {
		// Skip empty field names (root level)
		for _, key := range v.MapKeys() {
			newPath := key.String()
			markdown += generateMarkdownForField(newPath, v.MapIndex(key), originalData, depth)
		}
		return markdown
	}

	markdown += fmt.Sprintf("%s## %s\n\n", indent, fieldName)
	description := getFieldDescription(originalData, fieldPath)
	if description != "" {
		markdown += fmt.Sprintf("%s%s\n\n", indent, description)
	}

	switch v.Kind() {
	case reflect.Map:
		markdown += fmt.Sprintf("%s**Type**: Object\n\n", indent)
		markdown += fmt.Sprintf("%s**Properties**:\n\n", indent)
		for _, key := range v.MapKeys() {
			newPath := fmt.Sprintf("%s.%s", fieldPath, key.String())
			markdown += generateMarkdownForField(newPath, v.MapIndex(key), originalData, depth+1)
		}
	case reflect.Slice:
		markdown += fmt.Sprintf("%s**Type**: Array\n\n", indent)
		markdown += fmt.Sprintf("%s**Items**:\n\n", indent)
		for i := 0; i < v.Len(); i++ {
			markdown += fmt.Sprintf("%s- %s\n", indent, formatValue(v.Index(i)))
		}
		markdown += "\n"
	default:
		markdown += fmt.Sprintf("%s**Type**: %v\n\n", indent, v.Type())
		markdown += fmt.Sprintf("%s**Value**: `%s`\n\n", indent, formatValue(v))
	}

	return markdown
}

func formatValue(v reflect.Value) string {
	switch v.Kind() {
	case reflect.Map, reflect.Struct:
		b, _ := yaml.Marshal(v.Interface())
		return "\n```yaml\n" + string(b) + "```"
	case reflect.Slice:
		if v.Len() == 0 {
			return "[]"
		}
		if v.Index(0).Kind() == reflect.Map || v.Index(0).Kind() == reflect.Struct {
			b, _ := yaml.Marshal(v.Interface())
			return "\n```yaml\n" + string(b) + "```"
		}
	}
	return fmt.Sprintf("%v", v.Interface())
}

func getFieldDescription(data []byte, fieldPath string) string {
	lines := strings.Split(string(data), "\n")
	fieldName := getLastPathComponent(fieldPath)
	description := ""
	foundField := false

	for i, line := range lines {
		if strings.Contains(line, fieldName+":") {
			foundField = true
			// Check for inline comments
			if idx := strings.Index(line, "#"); idx != -1 {
				description += strings.TrimSpace(line[idx+1:]) + "\n"
			}
			// Look for preceding comments
			for j := i - 1; j >= 0; j-- {
				prevLine := strings.TrimSpace(lines[j])
				if strings.HasPrefix(prevLine, "#") {
					description = prevLine[1:] + "\n" + description
				} else {
					break
				}
			}
			break
		}
	}

	if !foundField {
		// If field not found, look for comments above the parent field
		parentPath := getParentPath(fieldPath)
		if parentPath != "" {
			return getFieldDescription(data, parentPath)
		}
	}

	return strings.TrimSpace(description)
}

func getLastPathComponent(path string) string {
	components := strings.Split(path, ".")
	return components[len(components)-1]
}

func getParentPath(path string) string {
	components := strings.Split(path, ".")
	if len(components) > 1 {
		return strings.Join(components[:len(components)-1], ".")
	}
	return ""
}

func init() {
	manifestCmd.Flags().BoolVarP(&generateGuide, "guide", "g", false, "Generate a reference guide for the manifest")
	manifestCmd.Flags().StringVarP(&outputFile, "output", "o", "", "Output file for the generated guide (default is stdout)")
}
