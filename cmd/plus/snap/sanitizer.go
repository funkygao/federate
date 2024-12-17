package snap

import (
	"log"
	"path/filepath"
	"strings"

	"federate/pkg/manifest"
)

func sanitizeCodeRepo(m *manifest.Manifest) {
	for _, component := range m.Components {
		log.Printf("Component: %s", component.RootDir())

		rootPom := filepath.Join(component.RootDir(), "pom.xml")
		processPom(rootPom, 1)

		for _, module := range component.MavenModules() {
			processModule(module, 1)
		}

		// Additional processing for the component
		removeUnnecessaryFiles(component.RootDir(), 1)
		removeSensitiveInformation(component.RootDir(), 1)
		updateConfigurations(component.RootDir(), 1)

		hinted = true
	}
}

func processPom(pomPath string, indent int) {
	if hinted {
		return
	}
	logIndent(indent, "Processing pom.xml:")
	logIndent(indent+1, "- Remove unnecessary dependencies")
	logIndent(indent+1, "- Update version numbers if needed")
	logIndent(indent+1, "- Remove references to internal repositories")
	logIndent(indent+1, "- Add reference to local Maven repository")
}

func processModule(moduleRoot string, indent int) {
	logIndent(indent, "module: %s", filepath.Base(moduleRoot))
}

func removeUnnecessaryFiles(dir string, indent int) {
	if hinted {
		return
	}
	logIndent(indent, "Removing unnecessary files:")
	logIndent(indent+1, "- Remove test code if not needed")
	logIndent(indent+1, "- Remove unnecessary resources")
	logIndent(indent+1, "- Update configurations for the specific profile")
	logIndent(indent+1, "- Remove internal documentation")
	logIndent(indent+1, "- Remove development scripts")
	logIndent(indent+1, "- Remove any temporary or cache files")
}

func removeSensitiveInformation(dir string, indent int) {
	if hinted {
		return
	}
	logIndent(indent, "Removing sensitive information:")
	logIndent(indent+1, "- Remove API keys, passwords, and other credentials")
	logIndent(indent+1, "- Remove internal comments that might contain sensitive info")
	logIndent(indent+1, "- Remove or obfuscate internal IP addresses or URLs")
}

func updateConfigurations(dir string, indent int) {
	if hinted {
		return
	}
	logIndent(indent, "Updating configurations:")
	logIndent(indent+1, "- Update configuration files for the specific profile")
	logIndent(indent+1, "- Remove or mask any internal-only configuration options")
}

func finalChecks(indent int) {
	logIndent(indent+1, "- Ensure all sensitive information has been removed")
	logIndent(indent+1, "- Check that the code is appropriate for customer use")
	logIndent(indent+1, "- Verify that all necessary documentation is included")
	logIndent(indent+1, "- Confirm that all open-source dependencies are compatible with customer use")
	logIndent(indent+1, "- Review git history to ensure no sensitive information remains")
	logIndent(indent+1, "- Verify compliance with customer contract and legal requirements")
	logIndent(indent+1, "- 待交付代码是否包含京东服务器密码配置及应用账号、密码、秘钥等敏感信息")
	logIndent(indent+1, "- 待交付代码是否已剔除集团内部研发的公共SDK及其他非必需交付的代码片段")
}

func logIndent(indent int, format string, v ...any) {
	indentStr := strings.Repeat("  ", indent)
	log.Printf(indentStr+format, v...)
}
