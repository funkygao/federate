package java

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseDependency(t *testing.T) {
	expectedDependencies := []DependencyInfo{
		{},
		{
			ArtifactId: "fusion-project",
			GroupId:    "com.google.wms",
		},
		{
			ArtifactId: "fusion-project",
			GroupId:    "com.google.wms",
			Version:    "1.0.0-SNAPSHOT",
		},
		{
			ArtifactId: "fusion-project",
			GroupId:    "com.google.wms",
			Version:    "1.0.0-SNAPSHOT",
			Scope:      "provided",
		},
	}
	fixtures := []string{
		"",
		"com.google.wms:fusion-project",
		"com.google.wms:fusion-project:1.0.0-SNAPSHOT",
		"com.google.wms:fusion-project:1.0.0-SNAPSHOT:provided",
	}
	for i, f := range fixtures {
		info := ParseDependency(f)
		assert.Equal(t, info.ArtifactId, expectedDependencies[i].ArtifactId)
	}
}
