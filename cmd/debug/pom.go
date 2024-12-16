package debug

import (
	"fmt"
	"os"

	"federate/pkg/java"
	"github.com/beevik/etree"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var pomCmd = &cobra.Command{
	Use:   "pom [dir]",
	Short: "Search for Maven artifacts with the same coordinates in pom.xml files",
	Long:  `Search for Maven artifacts with the same coordinates in pom.xml files within a specified directory recursively.`,
	Args:  cobra.MaximumNArgs(1),
	Run:   runPomCmd,
}

func runPomCmd(cmd *cobra.Command, args []string) {
	searchDir := "."
	if len(args) > 0 {
		searchDir = args[0]
	}

	pomChan, _ := java.ListFilesAsync(searchDir, func(info os.FileInfo, path string) bool {
		return info.Name() == "pom.xml"
	})

	artifacts := make(map[string][]string)
	for pom := range pomChan {
		searchPom(pom.Path, artifacts)
	}
	printResults(artifacts)
}

func searchPom(pomPath string, artifacts map[string][]string) {
	doc := etree.NewDocument()
	if err := doc.ReadFromFile(pomPath); err != nil {
		color.Red("Error reading pom.xml at %s: %v", pomPath, err)
		return
	}

	root := doc.SelectElement("project")
	if root == nil {
		return
	}

	groupId := getElementText(root, "groupId")
	if groupId == "" {
		parentElem := root.SelectElement("parent")
		if parentElem != nil {
			groupId = getElementText(parentElem, "groupId")
		}
	}

	artifactId := getElementText(root, "artifactId")

	if groupId != "" && artifactId != "" {
		key := fmt.Sprintf("%s:%s", groupId, artifactId)
		artifacts[key] = append(artifacts[key], pomPath)
	}
}

func getElementText(element *etree.Element, name string) string {
	if elem := element.SelectElement(name); elem != nil {
		return elem.Text()
	}
	return ""
}

func printResults(artifacts map[string][]string) {
	conflicts := 0
	for coordinates, paths := range artifacts {
		if len(paths) > 1 {
			conflicts++
			color.Yellow("[%s] found in:", coordinates)
			for _, path := range paths {
				fmt.Printf("  - %s\n", path)
			}
			fmt.Println()
		}
	}
	fmt.Printf("%d conflicts found\n", conflicts)
}
