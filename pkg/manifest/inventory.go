package manifest

import (
	"fmt"
	"sort"
	"strings"
)

type Inventory struct {
	Environments map[string]Environment
	Repos        map[string]RepoConfig
}

type Environment map[string]EnvRepoConfig

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
	for _, deployment := range m.Deployments {
		env := make(Environment)
		for _, component := range m.Components {
			// 查找匹配的环境配置
			var envSpec *EnvironmentSpec
			for _, e := range component.Envs {
				if e.Name == deployment.Env {
					envSpec = &e
					break
				}
			}

			if envSpec == nil {
				// 如果没有找到匹配的环境配置，使用默认值
				envRepoConfig := EnvRepoConfig{
					Branch:       "main", // 或者其他默认分支
					MavenProfile: component.SpringProfile,
				}
				env[component.Name] = envRepoConfig
			} else {
				envRepoConfig := EnvRepoConfig{
					Branch:       envSpec.Branch,
					MavenProfile: envSpec.MavenProfile,
				}
				env[component.Name] = envRepoConfig
			}
		}
		inventory.Environments[deployment.Env] = env
	}

	// 填充 Repos
	for _, component := range m.Components {
		repoConfig := RepoConfig{
			Address:           component.Repo,
			MavenBuildModules: strings.Join(component.RawDependencies, ","),
		}
		inventory.Repos[component.Name] = repoConfig
	}

	return inventory
}
