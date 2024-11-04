package manifest

import (
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"reflect"
	"regexp"

	"federate/pkg/java"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v2"
)

var (
	filePath    string
	envVarRegex = regexp.MustCompile(`\$\{([^:}]+)(:([^}]*))?\}`)
)

func RequiredManifestFileFlag(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&filePath, "input", "i", "manifest.yaml", "Path of the manifest file")
}

func File() string {
	return filePath
}

func Load() *Manifest {
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("Error loading manifest: %v", err)
	}
	defer file.Close()

	data, err := ioutil.ReadAll(file)
	if err != nil {
		log.Fatalf("Error loading manifest: %v", err)
	}

	// set default value
	manifest := Manifest{
		Version: "1.0",
		Kind:    "Fusion",
		Dir:     filepath.Dir(filePath),
		Main: MainSystem{
			Version: defaultMainVersion,
			Reconcile: ReconcileSpec{
				Taint: TaintSpec{
					LogConfigXml:     defaultLogConfigXml,
					MybatisConfigXml: defaultMybatisConfigXml,
				},
			},
		},
	}
	err = yaml.Unmarshal(data, &manifest)
	if err != nil {
		log.Fatalf("Error loading manifest: %v", err)
	}

	manifest.Main.Dependency.Includes = java.ParseDependencies(manifest.Main.Dependency.RawInclude)
	manifest.Main.Dependency.Excludes = java.ParseDependencies(manifest.Main.Dependency.RawExclude)
	manifest.Starter.Dependencies = java.ParseDependencies(manifest.Starter.RawDependencies)
	manifest.Main.Parent = java.ParseDependency(manifest.Main.RawParent)

	manifest.Main.Reconcile.M = &manifest.Main

	// 设置每个组件的引用 MainSystem 并解析 dependencies 字段
	for i := range manifest.Components {
		manifest.Components[i].M = &manifest.Main
		manifest.Components[i].Modules = java.ParseDependencies(manifest.Components[i].RawModules)
	}

	return &manifest
}

// 为 Manifest 结构体添加 UnmarshalYAML 方法
func (m *Manifest) UnmarshalYAML(unmarshal func(interface{}) error) error {
	// 创建一个临时结构体来解析 YAML
	type TempManifest Manifest
	temp := (*TempManifest)(m)

	// 解析 YAML 到临时结构体
	if err := unmarshal(temp); err != nil {
		return err
	}

	// 处理所有字符串字段
	m.processStrings(reflect.ValueOf(temp).Elem())

	return nil
}

// 递归处理所有字符串字段
func (m *Manifest) processStrings(v reflect.Value) {
	switch v.Kind() {
	case reflect.String:
		if v.CanSet() {
			v.SetString(parseEnvVars(v.String()))
		}
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			m.processStrings(v.Field(i))
		}
	case reflect.Slice:
		for i := 0; i < v.Len(); i++ {
			m.processStrings(v.Index(i))
		}
	case reflect.Map:
		for _, key := range v.MapKeys() {
			mapValue := v.MapIndex(key)
			if mapValue.Kind() == reflect.String && mapValue.CanSet() {
				newValue := parseEnvVars(mapValue.String())
				v.SetMapIndex(key, reflect.ValueOf(newValue))
			} else {
				m.processStrings(mapValue)
			}
		}
	case reflect.Ptr:
		if !v.IsNil() {
			m.processStrings(v.Elem())
		}
	}
}

func parseEnvVars(value string) string {
	return envVarRegex.ReplaceAllStringFunc(value, func(match string) string {
		parts := envVarRegex.FindStringSubmatch(match)
		envVar := parts[1]
		defaultValue := parts[3] // 这可能是空字符串

		if envValue, exists := os.LookupEnv(envVar); exists {
			return envValue
		}
		// 如果环境变量不存在，返回默认值（可能是空字符串）
		return defaultValue
	})
}
