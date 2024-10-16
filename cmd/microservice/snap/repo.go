package snap

import (
	"log"
	"os"
	"os/exec"
)

func createLocalMavenRepo() {
	err := os.MkdirAll(localRepoPath, 0755)
	if err != nil {
		log.Fatalf("Failed to create local Maven repository: %v", err)
	}
	log.Printf("Local Maven repository created at: %s", localRepoPath)
}

func copyDependenciesToLocalRepo() {
	cmd := exec.Command("mvn", "dependency:copy-dependencies", "-DoutputDirectory="+localRepoPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		log.Fatalf("Failed to copy dependencies: %v", err)
	}
	log.Println("Dependencies copied to local Maven repository")

	// Copy internal JARs
	internalLibsDir := "./internal-libs"
	cmd = exec.Command("find", internalLibsDir, "-name", "*.jar", "-exec", "cp", "{}", localRepoPath, ";")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err = cmd.Run()
	if err != nil {
		log.Fatalf("Failed to copy internal JARs: %v", err)
	}
	log.Println("Internal JARs copied to local Maven repository")
}
