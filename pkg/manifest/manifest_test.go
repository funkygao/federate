package manifest

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLoadManifest(t *testing.T) {
	filePath = "unit-test.yaml"
	manifest := LoadManifest()

	assert.Equal(t, "1.2", manifest.Version)
	assert.Equal(t, "com.jdwl.wms.runtime", manifest.Main.FederatedRuntimePackage())
	assert.Equal(t, "wms-stock-api-provider", manifest.Main.Dependency.Includes[0].ArtifactId)
	assert.Equal(t, "wms-inventory-web", manifest.Main.Dependency.Excludes[0].ArtifactId)
	assert.Equal(t, 1, len(manifest.Main.Reconcile.Resources.PropertySettlement))

	if manifest.Main.Name != "demo-starter" {
		t.Errorf("Expected main name to be 'demo-starter', got '%s'", manifest.Main.Name)
	}
	if manifest.Main.Version != "1.0.0-SNAPSHOT" {
		t.Errorf("Expected default version to be '1.0.0-SNAPSHOT', got '%s'", manifest.Main.Version)
	}

	expectedComponents := []ComponentInfo{
		{
			Name:          "wms-stock",
			SpringProfile: "on-premise",
			Dependencies: []DependencyInfo{
				{GroupId: "com.jdwl.wms", ArtifactId: "wms-stock-api-provider", Version: "1.0.0"},
				{GroupId: "com.jdwl.wms", ArtifactId: "wms-stock-work", Version: "1.0.0"},
			},
			Resources: ComponentResourceSpec{
				BaseDirs: []string{
					"wms-stock-api-provider/src/main/resources",
					"wms-stock-api/src/main/resources",
				},
			},
		},
		{
			Name:          "wms-inventory",
			SpringProfile: "on-premise-test",
			Dependencies: []DependencyInfo{
				{GroupId: "com.jdwl.wms", ArtifactId: "wms-inventory-web", Version: "1.0.1-SNAPSHOT"},
			},
			Resources: ComponentResourceSpec{
				BaseDirs: []string{
					"/wms-inventory-web/src/main/resources",
				},
			},
		},
	}

	if len(manifest.Components) != len(expectedComponents) {
		t.Fatalf("Expected %d components, got %d", len(expectedComponents), len(manifest.Components))
	}

	for i, expectedComponent := range expectedComponents {
		component := manifest.Components[i]
		if component.Name != expectedComponent.Name {
			t.Errorf("Expected component name to be '%s', got '%s'", expectedComponent.Name, component.Name)
		}
		if component.SpringProfile != expectedComponent.SpringProfile {
			t.Errorf("Expected component '%s' to have spring profile '%s', got '%s'", expectedComponent.Name, expectedComponent.SpringProfile, component.SpringProfile)
		}
		if len(component.Dependencies) != len(expectedComponent.Dependencies) {
			t.Errorf("Expected component '%s' to have %d dependencies, got %d", expectedComponent.Name, len(expectedComponent.Dependencies), len(component.Dependencies))
			continue
		}
		for j, expectedDep := range expectedComponent.Dependencies {
			if component.Dependencies[j].GroupId != expectedDep.GroupId ||
				component.Dependencies[j].ArtifactId != expectedDep.ArtifactId ||
				component.Dependencies[j].Version != expectedDep.Version {
				t.Errorf("Expected component '%s' dependency '%d' to have groupId '%s', artifactId '%s', version '%s', got groupId '%s', artifactId '%s', version '%s'",
					expectedComponent.Name, j, expectedDep.GroupId, expectedDep.ArtifactId, expectedDep.Version,
					component.Dependencies[j].GroupId, component.Dependencies[j].ArtifactId, component.Dependencies[j].Version)
			}
		}
		if len(component.Resources.BaseDirs) != len(expectedComponent.Resources.BaseDirs) {
			t.Errorf("Expected component '%s' to have %d resource base dirs, got %d", expectedComponent.Name, len(expectedComponent.Resources.BaseDirs), len(component.Resources.BaseDirs))
			continue
		}
		for j, expectedDir := range expectedComponent.Resources.BaseDirs {
			if component.Resources.BaseDirs[j] != expectedDir {
				t.Errorf("Expected component '%s' resource base dir '%d' to be '%s', got '%s'", expectedComponent.Name, j, expectedDir, component.Resources.BaseDirs[j])
			}
		}
	}
}

func TestParseMainClass(t *testing.T) {
	manifest := &Manifest{
		Main: MainSystem{
			MainClass: MainClassSpec{
				Name: "com.example.MainClass",
			},
		},
	}

	packageName, className := manifest.ParseMainClass()
	if packageName != "com.example" {
		t.Errorf("Expected package name to be 'com.example', got '%s'", packageName)
	}
	if className != "MainClass" {
		t.Errorf("Expected class name to be 'MainClass', got '%s'", className)
	}
}

func TestComponentDependencies(t *testing.T) {
	manifest := &Manifest{
		Components: []ComponentInfo{
			{
				Name: "component1",
				Dependencies: []DependencyInfo{
					{GroupId: "com.example", ArtifactId: "example1", Version: "1.0.0"},
				},
			},
			{
				Name: "component2",
				Dependencies: []DependencyInfo{
					{GroupId: "com.example", ArtifactId: "example2", Version: "2.0.0"},
				},
			},
		},
	}

	dependencies := manifest.ComponentDependencies()
	expectedDependencies := []DependencyInfo{
		{GroupId: "com.example", ArtifactId: "example1", Version: "1.0.0"},
		{GroupId: "com.example", ArtifactId: "example2", Version: "2.0.0"},
	}

	if len(dependencies) != len(expectedDependencies) {
		t.Fatalf("Expected %d dependencies, got %d", len(expectedDependencies), len(dependencies))
	}

	expectedDepMap := make(map[string]DependencyInfo)
	for _, dep := range expectedDependencies {
		key := dep.GroupId + ":" + dep.ArtifactId + ":" + dep.Version
		expectedDepMap[key] = dep
	}

	for _, dep := range dependencies {
		key := dep.GroupId + ":" + dep.ArtifactId + ":" + dep.Version
		if _, exists := expectedDepMap[key]; !exists {
			t.Errorf("Unexpected dependency found: groupId '%s', artifactId '%s', version '%s'",
				dep.GroupId, dep.ArtifactId, dep.Version)
		}
	}
}

func TestHasFeature(t *testing.T) {
	manifest := &Manifest{
		Main: MainSystem{
			Features: []string{"redis", "redis-lock"},
		},
	}

	if !manifest.HasFeature("redis") {
		t.Errorf("Expected HasFeature('redis') to be true")
	}

	if !manifest.HasFeature("redis-lock") {
		t.Errorf("Expected HasFeature('redis-lock') to be true")
	}

	if manifest.HasFeature("non-existent-feature") {
		t.Errorf("Expected HasFeature('non-existent-feature') to be false")
	}
}

func TestGroupId(t *testing.T) {
	main := MainSystem{
		MainClass: MainClassSpec{
			Name: "com.jdl.wms.ob.Foo",
		},
	}
	assert.Equal(t, "com.jdl.wms", main.GroupId())
}

func TestTargetResourceDir(t *testing.T) {
	manifest := &Manifest{
		Main: MainSystem{
			Name: "foo",
		},
	}
	assert.Equal(t, "generated/foo/src/main/resources", manifest.TargetResourceDir())
}
