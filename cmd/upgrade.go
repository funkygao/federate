package cmd

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"

	"github.com/spf13/cobra"
)

const (
	codingReleaseBaseUrl = "https://coding.jd.com/webapi/wms-ng/wms-microfusion/files"
	codingReleaseID      = 18465
)

var upgradeCmd = &cobra.Command{
	Use:   "upgrade",
	Short: "Automatically upgrade the federate tool",
	Long: `The upgrade command downloads the latest version of the federate tool
from the official repository and replaces the current binary with the new version.

Example usage:
  federate version upgrade`,
	Run: func(cmd *cobra.Command, args []string) {
		upgradeBinary()
	},
}

func upgradeBinary() {
	arch := runtime.GOARCH
	if arch != "amd64" && arch != "arm64" {
		log.Fatalf("Unsupported architecture: %s", arch)
	}
	url := fmt.Sprintf("%s/%d/federate.%s.%s", codingReleaseBaseUrl, codingReleaseID, runtime.GOOS, arch)
	log.Printf("Downloading: %s", url)

	// Create a temporary file
	tmpFile, err := os.CreateTemp("", "federate-*")
	if err != nil {
		log.Fatalf("Failed to create temporary file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	// Download the latest binary
	cmd := exec.Command("curl", "-s", "-f", "-L", "-o", tmpFile.Name(), url)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		log.Fatalf("Failed to download the latest version: %v", err)
	}

	// Check if the downloaded file is a binary or an HTML page
	magicNumber := make([]byte, 4)
	_, err = tmpFile.Read(magicNumber)
	if err != nil {
		log.Fatalf("Failed to read the temporary file: %v", err)
	}
	if !bytes.Equal(magicNumber, []byte{0xcf, 0xfa, 0xed, 0xfe}) { // Check for Mach-O binary magic number
		log.Fatalf("Downloaded file is not a valid binary")
	}

	// Get the absolute path of the current executable
	execPath, err := os.Executable()
	if err != nil {
		log.Fatalf("Failed to get the executable path: %v", err)
	}
	execPath, err = filepath.EvalSymlinks(execPath)
	if err != nil {
		log.Fatalf("Failed to resolve symlinks: %v", err)
	}

	// Replace the current binary
	if err := os.Rename(tmpFile.Name(), execPath); err != nil {
		log.Fatalf("Failed to replace the binary: %v", err)
	}

	fmt.Println("Upgrade successful")
	showVersion()
}
