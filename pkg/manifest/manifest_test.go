package manifest

import (
	"os"
	"testing"

	"federate/pkg/java"
	"github.com/stretchr/testify/assert"
)

func TestLoad(t *testing.T) {
	t.Cleanup(func() {
		os.Unsetenv("GROUP_ID")
	})

	filePath = "unit-test.yaml"
	manifest := Load()

	assert.Equal(t, "1.2", manifest.Version)
	assert.Equal(t, "com.jdwl.wms.runtime", manifest.Main.FederatedRuntimePackage())
	assert.Equal(t, "wms-stock-api-provider", manifest.Main.Dependency.Includes[0].ArtifactId)
	assert.Equal(t, "com.goog", manifest.Main.GroupId)

	// 设置环境变量
	os.Setenv("GROUP_ID", "com.goog.foo")
	manifest = Load()
	assert.Equal(t, "com.goog.foo", manifest.Main.GroupId)

	assert.Equal(t, "wms-inventory-web", manifest.Main.Dependency.Excludes[0].ArtifactId)
	assert.Equal(t, 1, len(manifest.Main.Reconcile.Resources.Property.Overrides))
	assert.Equal(t, false, manifest.Main.Reconcile.Resources.Property.IsDryRun())
	assert.Equal(t, int16(8082), manifest.RpmByEnv("on-premise").TomcatPort)

	c := manifest.ComponentByName("wms-stock")
	assert.Equal(t, "bar", c.Transform.Beans["foo"])
	assert.Equal(t, "baz", c.Transform.Beans["egg"])
	assert.Equal(t, "myFooService", c.Transform.Services["com.foo.FooService"])
	assert.Equal(t, true, c.NeedsTransform())
	assert.Equal(t, false, manifest.ComponentByName("wms-inventory").NeedsTransform())
	assert.Equal(t, 1, len(c.Transform.Autowired.Excludes))
	assert.Equal(t, true, c.Transform.Autowired.ExcludeBeanType("CountTaskMasterExternalService"))

	refMap := c.Transform.ServiceBeanRefMap()
	assert.Equal(t, "myFooService", refMap["fooService"])

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
			Modules: []java.DependencyInfo{
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
			Modules: []java.DependencyInfo{
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
		if len(component.Modules) != len(expectedComponent.Modules) {
			t.Errorf("Expected component '%s' to have %d dependencies, got %d", expectedComponent.Name, len(expectedComponent.Modules), len(component.Modules))
			continue
		}
		for j, expectedDep := range expectedComponent.Modules {
			if component.Modules[j].GroupId != expectedDep.GroupId ||
				component.Modules[j].ArtifactId != expectedDep.ArtifactId ||
				component.Modules[j].Version != expectedDep.Version {
				t.Errorf("Expected component '%s' dependency '%d' to have groupId '%s', artifactId '%s', version '%s', got groupId '%s', artifactId '%s', version '%s'",
					expectedComponent.Name, j, expectedDep.GroupId, expectedDep.ArtifactId, expectedDep.Version,
					component.Modules[j].GroupId, component.Modules[j].ArtifactId, component.Modules[j].Version)
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

func TestComponentModules(t *testing.T) {
	manifest := &Manifest{
		Components: []ComponentInfo{
			{
				Name: "component1",
				Modules: []java.DependencyInfo{
					{GroupId: "com.example", ArtifactId: "example1", Version: "1.0.0"},
				},
			},
			{
				Name: "component2",
				Modules: []java.DependencyInfo{
					{GroupId: "com.example", ArtifactId: "example2", Version: "2.0.0"},
				},
			},
		},
	}

	dependencies := manifest.ComponentModules()
	expectedDependencies := []java.DependencyInfo{
		{GroupId: "com.example", ArtifactId: "example1", Version: "1.0.0"},
		{GroupId: "com.example", ArtifactId: "example2", Version: "2.0.0"},
	}

	if len(dependencies) != len(expectedDependencies) {
		t.Fatalf("Expected %d dependencies, got %d", len(expectedDependencies), len(dependencies))
	}

	expectedDepMap := make(map[string]java.DependencyInfo)
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

func TestTargetResourceDir(t *testing.T) {
	manifest := &Manifest{
		Main: MainSystem{
			Name: "foo",
		},
	}
	assert.Equal(t, "foo/src/main/resources", manifest.TargetResourceDir())
}

func TestParseEnvVars(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		envVars  map[string]string
		expected string
	}{
		{
			name:     "No env vars",
			input:    "Hello, World!",
			envVars:  nil,
			expected: "Hello, World!",
		},
		{
			name:     "Single env var with default",
			input:    "Hello, ${NAME:World}!",
			envVars:  nil,
			expected: "Hello, World!",
		},
		{
			name:  "Single env var without default",
			input: "Hello, ${NAME}!",
			envVars: map[string]string{
				"NAME": "John",
			},
			expected: "Hello, John!",
		},
		{
			name:  "Multiple env vars",
			input: "${GREETING:Hello}, ${NAME:World}!",
			envVars: map[string]string{
				"GREETING": "Hi",
			},
			expected: "Hi, World!",
		},
		{
			name:  "Env var within text",
			input: "The value is ${VALUE:default} today",
			envVars: map[string]string{
				"VALUE": "special",
			},
			expected: "The value is special today",
		},
		{
			name:     "Env var with empty default",
			input:    "Empty default: ${EMPTY:}",
			envVars:  nil,
			expected: "Empty default: ",
		},
		{
			name:  "Env var with colon in value",
			input: "Time: ${TIME:12:00}",
			envVars: map[string]string{
				"TIME": "15:30",
			},
			expected: "Time: 15:30",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set environment variables
			for k, v := range tt.envVars {
				os.Setenv(k, v)
				defer os.Unsetenv(k)
			}

			result := parseEnvVars(tt.input)
			if result != tt.expected {
				t.Errorf("parseEnvVars(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestTrimComponentPath(t *testing.T) {
	c := ComponentInfo{Name: "wms-stock"}
	assert.Equal(t, "foo/bar.java", c.TrimComponentPath("wms-stock/foo/bar.java"))
}
