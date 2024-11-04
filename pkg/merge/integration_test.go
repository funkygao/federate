package merge

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"federate/pkg/manifest"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAnalyzeAllPropertySources(t *testing.T) {
	tempDir := t.TempDir()
	m := prepareTestManifest(t, tempDir)

	pm := NewPropertyManager(m)
	require.NoError(t, pm.AnalyzeAllPropertySources())
	allPropertiesJSON, _ := json.MarshalIndent(pm.allProperties, "", "  ")
	t.Logf("All properties:\n%s", string(allPropertiesJSON))

	conflicts := pm.IdentifyAllConflicts()
	conflictsJSON, _ := json.MarshalIndent(conflicts, "", "  ")
	t.Logf("Conflicts:\n%s", string(conflictsJSON))

	// 验证冲突检测
	assert.Contains(t, conflicts, "datasource.mysql.url")
	assert.Contains(t, conflicts, "mysql.url")
	assert.Contains(t, conflicts, "mysql.maximumPoolSize")
	assert.Contains(t, conflicts, "wms.datasource.maximumPoolSize")
	assert.NotContains(t, conflicts, "a.key")
	assert.NotContains(t, conflicts, "b.key")
	assert.NotContains(t, conflicts, "mysql.driver")

	// 验证引用解析
	assert.Equal(t, "jdbc:mysql://1.1.1.1", pm.allProperties["a"]["datasource.mysql.url"])
	assert.Equal(t, "jdbc:mysql://1.1.1.8", pm.allProperties["b"]["datasource.mysql.url"])

	assert.Equal(t, "jdbc:mysql://1.1.1.1", conflicts["datasource.mysql.url"]["a"])
	assert.Equal(t, "jdbc:mysql://1.1.1.8", conflicts["datasource.mysql.url"]["b"])

	// 不冲突
	assert.Equal(t, "foo", pm.allProperties["a"]["a.key"])
	assert.Equal(t, "0", pm.allProperties["b"]["b.key"])
	assert.Equal(t, "com.mysql.jdbc.Driver", pm.allProperties["a"]["wms.datasource.driverClassName"])
}

// properties引用了yml，yml也引用了properties
// 有冲突，也有不冲突
func prepareTestManifest(t *testing.T, tempDir string) *manifest.Manifest {
	a_properties := `
a.key=foo
mysql.driver=com.mysql.jdbc.Driver
mysql.url=${datasource.mysql.url}
mysql.maximumPoolSize=10
`
	aRoot := filepath.Join(tempDir, "a")
	os.MkdirAll(aRoot, 0755)
	err := os.WriteFile(filepath.Join(aRoot, "application.properties"), []byte(a_properties), 0644)
	require.NoError(t, err)

	a_yaml := `
datasource:
  mysql:
    url: jdbc:mysql://1.1.1.1
wms:
  datasource:
    driverClassName: ${mysql.driver}
    maximumPoolSize: ${mysql.maximumPoolSize}
`
	err = os.WriteFile(filepath.Join(aRoot, "application.yml"), []byte(a_yaml), 0644)
	require.NoError(t, err)

	b_properties := `
b.key=0
mysql.driver=com.mysql.jdbc.Driver
mysql.url=${datasource.mysql.url}
mysql.maximumPoolSize=20
`
	bRoot := filepath.Join(tempDir, "b")
	os.MkdirAll(bRoot, 0755)
	err = os.WriteFile(filepath.Join(bRoot, "application.properties"), []byte(b_properties), 0644)
	require.NoError(t, err)

	b_yaml := `
datasource:
  mysql:
    url: jdbc:mysql://1.1.1.8
wms:
  datasource:
    driverClassName: ${mysql.driver}
    maximumPoolSize: ${mysql.maximumPoolSize}
`
	err = os.WriteFile(filepath.Join(bRoot, "application.yml"), []byte(b_yaml), 0644)
	require.NoError(t, err)

	return &manifest.Manifest{
		Components: []manifest.ComponentInfo{
			{
				Name:    "a",
				BaseDir: tempDir,
				Resources: manifest.ComponentResourceSpec{
					BaseDirs: []string{""},
					PropertySources: []string{
						"application.properties",
					},
				},
			},
			{
				Name:    "b",
				BaseDir: tempDir,
				Resources: manifest.ComponentResourceSpec{
					BaseDirs: []string{""},
					PropertySources: []string{
						"application.properties",
					},
				},
			},
		},
	}
}
