package manifest

import (
	"fmt"
	"sort"
)

type Inventory struct {
	Environments map[string]Environment
	Repos        map[string]RepoConfig
}

type Environment map[string]EnvRepoConfig // key is repo

type EnvRepoConfig struct {
	Branch       string
	MavenProfile string
}

type RepoConfig struct {
	Address           string
	MavenBuildModules string
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

func (m *Manifest) ToInventory() *Inventory {
	inventory := &Inventory{
		Environments: make(map[string]Environment),
		Repos:        make(map[string]RepoConfig),
	}

	// 填充 Environments
	for _, component := range m.Components {
		for _, env := range component.Envs {
			if _, ok := inventory.Environments[env.Name]; !ok {
				inventory.Environments[env.Name] = make(Environment)
			}
			inventory.Environments[env.Name][component.Name] = EnvRepoConfig{
				Branch:       env.Branch,
				MavenProfile: env.MavenProfile,
			}
		}
	}

	// 填充 Repos
	for _, component := range m.Components {
		repoConfig := RepoConfig{
			Address:           component.Repo,
			MavenBuildModules: component.MavenBuildModules(),
		}
		inventory.Repos[component.Name] = repoConfig
	}

	return inventory
}
