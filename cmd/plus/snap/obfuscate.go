//go:build proguard

package snap

import (
	"embed"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"

	"federate/pkg/manifest"
	"federate/pkg/merge/ledger"
	"federate/pkg/step"
	"federate/pkg/util"
)

//go:embed embedded/proguard.jar
var proguardJar embed.FS

func obfuscateJars(m *manifest.Manifest, bar step.Bar) {
	if !enableObfuscation {
		log.Println("Obfuscation not enabled")
		return
	}

	oldBarDesc := bar.State().Description
	defer bar.Describe(oldBarDesc)

	proguardJarPath, err := extractProGuardJar()
	if err != nil {
		log.Fatalf("%v", err)
	}
	defer os.RemoveAll(filepath.Dir(proguardJarPath))

	var jars []string
	err = filepath.Walk(absLocalRepoPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && m.Main.Plus.ObfuscateJar(info) {
			jars = append(jars, path)
		}
		return nil
	})

	if err != nil {
		log.Fatalf("Failed to walk through local repository: %v", err)
	}

	bar.ChangeMax(len(jars))

	cwd, _ := os.Getwd()

	for _, jarPath := range jars {
		bar.Describe(fmt.Sprintf("%s %s", oldBarDesc, filepath.Base(jarPath)))
		err := obfuscateJar(jarPath, proguardJarPath)
		bar.Add(1)
		if err != nil {
			relPath, _ := filepath.Rel(cwd, jarPath)
			log.Printf("\nFailed to obfuscate JAR %s: %v", relPath, err)

			ledger.Get().FailToObfuscateJar(relPath)
		}
	}
}

func extractProGuardJar() (string, error) {
	tempDir, err := os.MkdirTemp("", "proguard")
	if err != nil {
		return "", fmt.Errorf("failed to create temp dir: %v", err)
	}

	jarPath := filepath.Join(tempDir, "proguard.jar")
	jarFile, err := proguardJar.Open("embedded/proguard.jar")
	if err != nil {
		os.RemoveAll(tempDir)
		return "", fmt.Errorf("failed to open embedded ProGuard JAR: %v", err)
	}
	defer jarFile.Close()

	outFile, err := os.Create(jarPath)
	if err != nil {
		os.RemoveAll(tempDir)
		return "", fmt.Errorf("failed to create ProGuard JAR file: %v", err)
	}
	defer outFile.Close()

	_, err = io.Copy(outFile, jarFile)
	if err != nil {
		os.RemoveAll(tempDir)
		return "", fmt.Errorf("failed to write ProGuard JAR: %v", err)
	}

	return jarPath, nil
}

func obfuscateJar(jarPath, proguardJarPath string) error {
	// 创建临时目录
	tempDir, err := os.MkdirTemp("", "obfuscated")
	if err != nil {
		return fmt.Errorf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	obfuscatedJarPath := filepath.Join(tempDir, "obfuscated.jar")

	proguardConfig := fmt.Sprintf(`
-injars %s
-outjars %s
-libraryjars <java.home>/lib/rt.jar
-libraryjars <java.home>/lib/jce.jar
-libraryjars <java.home>/lib/jsse.jar
-libraryjars %s/slf4j-api-1.7.30.jar
-dontusemixedcaseclassnames
-keepattributes Signature
-dontwarn **
-keep class ** { *; }
-keep interface ** { *; }
-keep public class * {
    public protected *;
}
-keepclassmembernames class * {
    java.lang.Class class$(java.lang.String);
    java.lang.Class class$(java.lang.String, boolean);
}
-keepclasseswithmembernames class * {
    native <methods>;
}
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}
-keepclassmembers class * implements java.io.Serializable {
    static final long serialVersionUID;
    private static final java.io.ObjectStreamField[] serialPersistentFields;
    private void writeObject(java.io.ObjectOutputStream);
    private void readObject(java.io.ObjectInputStream);
    java.lang.Object writeReplace();
    java.lang.Object readResolve();
}
-verbose
`, jarPath, obfuscatedJarPath, absLocalRepoPath)

	// 将配置写入临时文件
	configPath := filepath.Join(tempDir, "proguard.config")
	err = os.WriteFile(configPath, []byte(proguardConfig), 0644)
	if err != nil {
		return fmt.Errorf("Failed to write ProGuard config: %v", err)
	}

	// 运行 ProGuard
	cmd := exec.Command("java", "-jar", proguardJarPath, "@"+configPath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("ProGuard failed: %v\nOutput: %s", err, util.Truncate(string(output), 8<<10))
	}

	// 用混淆后的 JAR 替换原始 JAR
	err = os.Rename(obfuscatedJarPath, jarPath)
	if err != nil {
		return fmt.Errorf("Failed to replace original JAR with obfuscated JAR: %v", err)
	}

	return nil
}
