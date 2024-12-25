package util

import (
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"

	"federate/internal/fs"
	"github.com/spf13/cobra"
)

const (
	CFR       = "cfr-0.152.jar"
	CFR_EMBED = "templates/jar/" + CFR
)

var decompileCmd = &cobra.Command{
	Use:   "decompile [jar_file]",
	Short: "Decompile a specified JAR file",
	Long:  `The 'decompile' command decompiles a specified JAR file using CFR decompiler.`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		runDecompile(args[0])
	},
}

func runDecompile(jarFile string) {
	// Check if the JAR file exists
	if _, err := os.Stat(jarFile); os.IsNotExist(err) {
		log.Fatalf("Error: JAR file '%s' does not exist.", jarFile)
	}

	// Create output directory
	outputDir := filepath.Join(filepath.Dir(jarFile), "decompiled")
	err := os.MkdirAll(outputDir, 0755)
	if err != nil {
		log.Fatalf("Error creating output directory: %v", err)
	}

	// Extract CFR jar to a temporary file
	tempDir, err := os.MkdirTemp("", "cfr")
	if err != nil {
		log.Fatalf("Error creating temporary directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	cfrJarPath := filepath.Join(tempDir, CFR)
	cfrJarFile, err := os.Create(cfrJarPath)
	if err != nil {
		log.Fatalf("Error creating temporary CFR jar file: %v", err)
	}
	defer cfrJarFile.Close()

	cfrJarContent, err := fs.FS.Open(CFR_EMBED)
	if err != nil {
		log.Fatalf("Error opening embedded CFR jar: %v", err)
	}
	defer cfrJarContent.Close()

	_, err = io.Copy(cfrJarFile, cfrJarContent)
	if err != nil {
		log.Fatalf("Error copying CFR jar content: %v", err)
	}

	// Run CFR decompiler
	cmd := exec.Command("java", "-jar", cfrJarPath, jarFile, "--outputdir", outputDir)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	err = cmd.Run()
	if err != nil {
		log.Fatalf("Error decompiling JAR: %v", err)
	}

	log.Printf("Decompilation complete. Output directory: %s", outputDir)
}
