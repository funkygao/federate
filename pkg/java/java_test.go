package java

import (
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestClassSimpleName(t *testing.T) {
	assert.Equal(t, "Foo", ClassSimpleName("com.jdl.wms.Foo"))
}

func TestClassPackageName(t *testing.T) {
	assert.Equal(t, "com.jdl.wms", ClassPackageName("com.jdl.wms.Foo"))
}

func TestIsResourceFile(t *testing.T) {
	tests := []struct {
		path     string
		expected bool
	}{
		{"config.json", true},
		{"beans.xml", true},
		{"messages.properties", true},
		{"application.yml", true},
		{"application.yaml", false},
		{"template.html", true},
		{"script.js", false},
		{"style.css", false},
	}

	for _, test := range tests {
		result := IsResourceFile(newMockFileInfo(), test.path)
		if result != test.expected {
			t.Errorf("isJavaResourceFile(%s) = %v; want %v", test.path, result, test.expected)
		}
	}
}

func TestPkg2Path(t *testing.T) {
	assert.Equal(t, "com/goog/wms/addon", Pkg2Path("com.goog.wms.addon"))
}

func TestIsSpringYamlFile(t *testing.T) {
	tests := []struct {
		path     string
		expected bool
	}{
		{"application.yaml", false},
		{"application.yml", true},
		{"application-dev.yaml", false},
		{"application-prod.yml", true},
		{"application-gray-ka.yml", true},
		{"application-on-premise-test.YML", true},
		{"application-config.xml", false},
		{"config.yaml", false},
		{"application.properties", false},
	}

	for _, test := range tests {
		result := IsSpringYamlFile(newMockFileInfo(), test.path)
		if result != test.expected {
			t.Errorf("isSpringYamlFile(%s) = %v; want %v", test.path, result, test.expected)
		}
	}
}

type mockFileInfo struct{}

func (m mockFileInfo) Name() string       { return "mockFile" }
func (m mockFileInfo) Size() int64        { return 0 }
func (m mockFileInfo) Mode() os.FileMode  { return 0 }
func (m mockFileInfo) ModTime() time.Time { return time.Time{} }
func (m mockFileInfo) IsDir() bool        { return false }
func (m mockFileInfo) Sys() interface{}   { return nil }

func newMockFileInfo() os.FileInfo {
	return mockFileInfo{}
}
