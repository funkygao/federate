package snap

import (
	"log"
	"os"
	"path/filepath"

	"federate/pkg/manifest"
)

var localRepoPath = "generated/artifactory"

func createLocalMavenRepo() {
	logIndent(0, "Creating local Maven repository at: %s", localRepoPath)
	err := os.MkdirAll(localRepoPath, 0755)
	if err != nil {
		log.Fatalf("Failed to create local Maven repository: %v", err)
	}
	logIndent(1, "Local Maven repository created successfully")
}

func updatePomFilesForLocalRepo(m *manifest.Manifest) {
	logIndent(0, "Updating pom.xml files to use local Maven repository")
	for _, component := range m.Components {
		pomPath := filepath.Join(component.RootDir(), "pom.xml")
		logIndent(1, "Processing: %s", pomPath)
		logIndent(2, "Adding local repository reference:")
		logIndent(3, "<repositories>")
		logIndent(4, "<repository>")
		logIndent(5, "<id>local-maven-repo</id>")
		logIndent(5, "<url>file://${project.basedir}/%s</url>", filepath.Base(localRepoPath))
		logIndent(4, "</repository>")
		logIndent(3, "</repositories>")

		logIndent(2, "Removing internal repository references")
		logIndent(2, "Updating dependency versions if necessary")

		// TODO: Implement actual XML parsing and modification
		logIndent(2, "Command to update pom.xml:")
		logIndent(3, "sed -i 's|<repositories>.*</repositories>|<repositories><repository><id>local-maven-repo</id><url>file://${project.basedir}/%s</url></repository></repositories>|g' %s", filepath.Base(localRepoPath), pomPath)
	}
}

func copyDependenciesToLocalRepo() {
	logIndent(0, "Copying required dependencies to local Maven repository")
	logIndent(1, "Using Maven dependency:copy-dependencies plugin")
	logIndent(2, "mvn dependency:copy-dependencies -DoutputDirectory=%s", localRepoPath)

	logIndent(1, "Copying internal JARs")
	// Assuming internal JARs are in a directory called 'internal-libs'
	internalLibsDir := "./internal-libs"
	logIndent(2, "find %s -name '*.jar' -exec cp {} %s \\;", internalLibsDir, localRepoPath)

	logIndent(1, "Generating Maven metadata files")
	logIndent(2, "for jar in %s/*.jar; do", localRepoPath)
	logIndent(3, "  groupId=$(echo $jar | sed -E 's/.*\\/(.*)\\/.*\\/.*\\.jar/\\1/')")
	logIndent(3, "  artifactId=$(echo $jar | sed -E 's/.*\\/.*//')")
	//logIndent(3, "  version=$(echo $jar | sed -E 's/.*-(.*)\.jar/\\1/')")
	logIndent(3, "  mkdir -p %s/$groupId/$artifactId/$version", localRepoPath)
	logIndent(3, "  mv $jar %s/$groupId/$artifactId/$version/", localRepoPath)
	logIndent(3, "  cat << EOF > %s/$groupId/$artifactId/$version/maven-metadata-local.xml", localRepoPath)
	logIndent(3, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
	logIndent(3, "<metadata>")
	logIndent(4, "  <groupId>$groupId</groupId>")
	logIndent(4, "  <artifactId>$artifactId</artifactId>")
	logIndent(4, "  <version>$version</version>")
	logIndent(3, "</metadata>")
	logIndent(3, "EOF")
	logIndent(2, "done")
}
