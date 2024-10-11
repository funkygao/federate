package manifest

import (
	"strings"
)

type DependencyInfo struct {
	GroupId    string
	ArtifactId string
	Version    string
	Scope      string
}

func parseDependencies(rawDeps []string) []DependencyInfo {
	var parsedDependencies []DependencyInfo
	for _, dep := range rawDeps {
		parsedDependencies = append(parsedDependencies, parseDependency(dep))
	}
	return parsedDependencies
}

func parseDependency(dep string) DependencyInfo {
	parts := strings.Split(dep, ":")
	if len(parts) == 3 {
		return DependencyInfo{
			GroupId:    parts[0],
			ArtifactId: parts[1],
			Version:    parts[2],
			Scope:      "compile",
		}
	}
	if len(parts) == 4 {
		return DependencyInfo{
			GroupId:    parts[0],
			ArtifactId: parts[1],
			Version:    parts[2],
			Scope:      parts[3],
		}
	}
	return DependencyInfo{}
}
