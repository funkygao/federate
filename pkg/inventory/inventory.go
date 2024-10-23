package inventory

import (
	"fmt"
	"io/ioutil"
	"sort"

	"gopkg.in/yaml.v2"
)

type Inventory struct {
	Environments map[string]Environment `yaml:"environments"`
	Repos        map[string]RepoConfig  `yaml:"repos"`
}

type Environment map[string]EnvRepoConfig

type EnvRepoConfig struct {
	Branch       string `yaml:"branch"`
	MavenProfile string `yaml:"maven_profile"`
}

type RepoConfig struct {
	Address           string `yaml:"address"`
	MavenBuildModules string `yaml:"maven_install_module"`
}

func ReadInventory(filename string) (*Inventory, error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("error reading inventory file: %v", err)
	}

	var inventory Inventory
	err = yaml.Unmarshal(data, &inventory)
	if err != nil {
		return nil, fmt.Errorf("error parsing YAML: %v", err)
	}

	return &inventory, nil
}

func (inv *Inventory) GetEnvironments() []string {
	envs := make([]string, 0, len(inv.Environments))
	for env := range inv.Environments {
		envs = append(envs, env)
	}
	sort.Strings(envs)
	return envs
}

func (inv *Inventory) GetRepos() []string {
	repos := make([]string, 0, len(inv.Repos))
	for repo := range inv.Repos {
		repos = append(repos, repo)
	}
	sort.Strings(repos)
	return repos
}

func (inv *Inventory) GetRepoInfo(env string) ([][]string, error) {
	if envConfig, ok := inv.Environments[env]; ok {
		var rows [][]string
		for repo, repoConfig := range inv.Repos {
			envRepoConfig := envConfig[repo]
			row := []string{
				repo,
				repoConfig.Address,
				envRepoConfig.Branch,
				envRepoConfig.MavenProfile,
				repoConfig.MavenBuildModules,
			}
			rows = append(rows, row)
		}
		sort.Slice(rows, func(i, j int) bool {
			return rows[i][0] < rows[j][0]
		})
		return rows, nil
	}
	return nil, fmt.Errorf("environment %s not found", env)
}
