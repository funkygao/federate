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
	resolvedPropertiesJSON, _ := json.MarshalIndent(pm.resolvedProperties, "", "  ")
	t.Logf("All properties:\n%s", string(resolvedPropertiesJSON))
	unresolvedPropertiesJSON, _ := json.MarshalIndent(pm.unresolvedProperties, "", "  ")
	t.Logf("Unresovled properties:\n%s", string(unresolvedPropertiesJSON))

	conflicts := pm.IdentifyAllConflicts()
	conflictsJSON, _ := json.MarshalIndent(conflicts, "", "  ")
	t.Logf("Conflicts:\n%s", string(conflictsJSON))

	// 验证无法解析的keys
	assert.Equal(t, "${non.exist}", pm.unresolvedProperties["b"]["wms.datasource.unresolved"].Value)
	assert.Equal(t, 1, len(pm.unresolvedProperties["b"]))

	// 验证冲突检测
	assert.Contains(t, conflicts, "datasource.mysql.url")
	assert.Contains(t, conflicts, "mysql.url")
	assert.Contains(t, conflicts, "mysql.maximumPoolSize")
	assert.Contains(t, conflicts, "wms.datasource.maximumPoolSize")
	assert.NotContains(t, conflicts, "a.key")
	assert.NotContains(t, conflicts, "b.key")
	assert.NotContains(t, conflicts, "mysql.driver")

	// 验证引用解析
	assert.Equal(t, "1234", pm.resolvedProperties["a"]["schedule.token"].Value) // 自己引用自己，只是properties引用yaml
	assert.Equal(t, "jdbc:mysql://1.1.1.1", pm.resolvedProperties["a"]["datasource.mysql.url"].Value)
	assert.Equal(t, "jdbc:mysql://1.1.1.1", pm.resolvedProperties["a"]["datasource.mysql.url"].Value)
	assert.Equal(t, "jdbc:mysql://1.1.1.8", pm.resolvedProperties["b"]["datasource.mysql.url"].Value)
	assert.Equal(t, "jdbc:mysql://1.1.1.9", pm.resolvedProperties["a"]["wms.reverse.datasource.ds0-master.jdbcUrl"].Value)

	assert.Equal(t, "jdbc:mysql://1.1.1.1", conflicts["datasource.mysql.url"]["a"])
	assert.Equal(t, "jdbc:mysql://1.1.1.8", conflicts["datasource.mysql.url"]["b"])

	// 不冲突
	assert.Equal(t, "foo", pm.resolvedProperties["a"]["a.key"].Value)
	assert.Equal(t, "0", pm.resolvedProperties["b"]["b.key"].Value)
	assert.Equal(t, "com.mysql.jdbc.Driver", pm.resolvedProperties["a"]["wms.datasource.driverClassName"].Value)

	// 调和冲突
	_, err := pm.ReconcileConflicts(true) // 使用 dryRun 模式
	require.NoError(t, err)

	// 检查解决后的属性
	resolvedProps := pm.resolvedProperties

	// 检查 datasource.mysql.url 是否被正确前缀化
	assert.Contains(t, resolvedProps["a"], "a.datasource.mysql.url")
	assert.Contains(t, resolvedProps["b"], "b.datasource.mysql.url")
	assert.Equal(t, "jdbc:mysql://1.1.1.1", resolvedProps["a"]["a.datasource.mysql.url"].Value)
	assert.Equal(t, "jdbc:mysql://1.1.1.8", resolvedProps["b"]["b.datasource.mysql.url"].Value)

	// 检查 mysql.maximumPoolSize 是否被正确前缀化
	assert.Contains(t, resolvedProps["a"], "a.mysql.maximumPoolSize")
	assert.Contains(t, resolvedProps["b"], "b.mysql.maximumPoolSize")
	assert.Equal(t, "10", resolvedProps["a"]["a.mysql.maximumPoolSize"].Value)
	assert.Equal(t, "20", resolvedProps["b"]["b.mysql.maximumPoolSize"].Value)

	// 检查非冲突的键是否保持不变
	assert.Equal(t, "foo", resolvedProps["a"]["a.key"].Value)
	assert.Equal(t, "0", resolvedProps["b"]["b.key"].Value)

	// 检查原始的冲突键是否仍然存在（因为它们可能被第三方包使用）
	assert.Contains(t, resolvedProps["a"], "datasource.mysql.url")
	assert.Contains(t, resolvedProps["b"], "datasource.mysql.url")
}

// properties引用了yml，yml也引用了properties
// 有冲突，也有不冲突
func prepareTestManifest(t *testing.T, tempDir string) *manifest.Manifest {
	a_properties := `
a.key=foo
mysql.driver=com.mysql.jdbc.Driver
mysql.url=${datasource.mysql.url}
mysql.maximumPoolSize=10
reverse.master.ds0.mysql.url=${datasource.reverse.master.mysql.url}
schedule.token=${schedule.token}
`
	aRoot := filepath.Join(tempDir, "a")
	os.MkdirAll(aRoot, 0755)
	err := os.WriteFile(filepath.Join(aRoot, "application.properties"), []byte(a_properties), 0644)
	require.NoError(t, err)

	a_yaml := `
datasource:
  mysql:
    url: jdbc:mysql://1.1.1.1
  reverse:
    master:
      mysql:
        url: jdbc:mysql://1.1.1.9

schedule:
  token: 1234

wms:
  reverse:
    datasource:
      ds0-master:
        jdbcUrl: ${reverse.master.ds0.mysql.url}
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
    unresolved: ${non.exist}
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
