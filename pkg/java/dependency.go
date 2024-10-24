package java

import (
	"strings"
)

type DependencyInfo struct {
	GroupId    string
	ArtifactId string
	Version    string
	Scope      string
}

func ParseDependencies(rawDeps []string) []DependencyInfo {
	var parsedDependencies []DependencyInfo
	for _, dep := range rawDeps {
		parsedDependencies = append(parsedDependencies, ParseDependency(dep))
	}
	return parsedDependencies
}

func ParseDependency(dep string) (info DependencyInfo) {
	parts := strings.Split(dep, ":")
	switch len(parts) {
	case 2:
		info.GroupId, info.ArtifactId = parts[0], parts[1]
	case 3:
		info.GroupId, info.ArtifactId, info.Version = parts[0], parts[1], parts[2]
		info.Scope = "compile"
	case 4:
		info.GroupId, info.ArtifactId, info.Version, info.Scope = parts[0], parts[1], parts[2], parts[3]
	}

	return
}
