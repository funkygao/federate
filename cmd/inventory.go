package cmd

import (
	"fmt"
	"io/ioutil"
	"os"
	"sort"
	"strings"

	"federate/pkg/tablerender"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v2"
)

var (
	inventoryFile string
	format        string
	env           string
)

var inventoryCmd = &cobra.Command{
	Use:   "inventory",
	Short: "Parse and display the inventory configuration from YAML file",
	Long: `The inventory command parses the specified YAML inventory file and displays its contents.
It can output information in different formats suitable for direct use in Makefile or for human reading.

Available formats:
- human: Human-readable output
- make: Makefile-friendly variable definitions
- env-list: List of available environments
- repo-list: List of all repositories
- repo-info: Detailed repository information for a specific environment`,
	Run: func(cmd *cobra.Command, args []string) {
		parseInventory()
	},
}

type Inventory struct {
	Environments map[string]Environment `yaml:"environments"`
	Repos        map[string]RepoConfig  `yaml:"repos"`
}

type Environment struct {
	Repos map[string]EnvRepoConfig `yaml:"repos"`
}

type EnvRepoConfig struct {
	Branch       string `yaml:"branch"`
	MavenProfile string `yaml:"maven_profile"`
}

type RepoConfig struct {
	Address           string `yaml:"address"`
	MavenBuildModules string `yaml:"maven_build_modules"`
}

func parseInventory() {
	data, err := ioutil.ReadFile(inventoryFile)
	if err != nil {
		fmt.Printf("Error reading inventory file: %v\n", err)
		os.Exit(1)
	}

	var inventory Inventory
	err = yaml.Unmarshal(data, &inventory)
	if err != nil {
		fmt.Printf("Error parsing YAML: %v\n", err)
		os.Exit(1)
	}

	switch format {
	case "human":
		printHumanReadable(inventory)
	case "make":
		printMakeFormat(inventory)
	case "env-list":
		printEnvList(inventory)
	case "repo-list":
		printRepoList(inventory)
	case "repo-info":
		printRepoInfoTable(inventory, env)
	default:
		fmt.Printf("Unknown format: %s\n", format)
		os.Exit(1)
	}
}

func printHumanReadable(inventory Inventory) {
	fmt.Printf("Environments:\n")
	for env, config := range inventory.Environments {
		fmt.Printf("  %s:\n", env)
		for repo, repoConfig := range config.Repos {
			fmt.Printf("    %s:\n", repo)
			fmt.Printf("      Branch: %s\n", repoConfig.Branch)
			fmt.Printf("      Maven Profile: %s\n", repoConfig.MavenProfile)
		}
	}

	fmt.Printf("\nRepositories:\n")
	for repo, config := range inventory.Repos {
		fmt.Printf("  %s:\n", repo)
		fmt.Printf("    Address: %s\n", config.Address)
		fmt.Printf("    Maven Build Modules: %s\n", config.MavenBuildModules)
	}
}

func printMakeFormat(inventory Inventory) {
	for repo, config := range inventory.Repos {
		fmt.Printf("%s_ADDRESS=%s\n", strings.ToUpper(repo), config.Address)
		fmt.Printf("%s_MAVEN_BUILD_MODULES=%s\n", strings.ToUpper(repo), config.MavenBuildModules)
	}
	for env, config := range inventory.Environments {
		for repo, repoConfig := range config.Repos {
			fmt.Printf("%s_%s_BRANCH=%s\n", strings.ToUpper(repo), strings.ToUpper(env), repoConfig.Branch)
			fmt.Printf("%s_%s_MAVEN_PROFILE=%s\n", strings.ToUpper(repo), strings.ToUpper(env), repoConfig.MavenProfile)
		}
	}
}

func printEnvList(inventory Inventory) {
	envs := make([]string, 0, len(inventory.Environments))
	for env := range inventory.Environments {
		envs = append(envs, env)
	}
	fmt.Println(strings.Join(envs, " "))
}

func printRepoList(inventory Inventory) {
	for repo := range inventory.Repos {
		fmt.Println(repo)
	}
}

func printRepoInfoTable(inventory Inventory, env string) {
	if envConfig, ok := inventory.Environments[env]; ok {
		header := []string{"Repository", "Git Address", "Branch", "Maven Profile", "Maven Build Modules"}
		var rows [][]string

		for repo, repoConfig := range envConfig.Repos {
			row := []string{
				repo,
				inventory.Repos[repo].Address,
				repoConfig.Branch,
				repoConfig.MavenProfile,
				inventory.Repos[repo].MavenBuildModules,
			}
			rows = append(rows, row)
		}

		// 按仓库名称排序
		sort.Slice(rows, func(i, j int) bool {
			return rows[i][0] < rows[j][0]
		})

		fmt.Printf("Environment: %s\n\n", env)
		tablerender.DisplayTable(header, rows, true, 0)
	} else {
		fmt.Printf("Environment %s not found\n", env)
		os.Exit(1)
	}
}

func init() {
	inventoryCmd.Flags().StringVarP(&inventoryFile, "inventory", "i", "", "Path to the inventory.yaml")
	inventoryCmd.MarkFlagRequired("inventory")
	inventoryCmd.Flags().StringVarP(&format, "format", "f", "human", "Output format: human, make, env-list, repo-list, or repo-info")
	inventoryCmd.Flags().StringVarP(&env, "env", "e", "", "Environment for repo-info format")
}
