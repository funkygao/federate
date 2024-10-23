package version

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

const (
	cacheDir  = ".federate"
	cacheFile = "release_cache.json"
)

var forceUpgrade bool

type GithubRelease struct {
	TagName string `json:"tag_name"`
	Assets  []struct {
		Name               string `json:"name"`
		BrowserDownloadURL string `json:"browser_download_url"`
	} `json:"assets"`
}

var upgradeCmd = &cobra.Command{
	Use:   "upgrade",
	Short: "Automatically upgrade the federate tool",
	Long: `The upgrade command downloads the latest version of the federate tool
from the official GitHub repository and replaces the current binary with the new version.

Example usage:
  federate version upgrade`,
	Run: func(cmd *cobra.Command, args []string) {
		upgradeBinary()
	},
}

func upgradeBinary() {
	t0 := time.Now()

	if err := ensureCacheDir(); err != nil {
		log.Printf("Warning: Failed to create cache directory: %v", err)
	}

	// 获取最新的 release 信息
	release, err := getLatestRelease()
	if err != nil {
		log.Fatalf("Failed to get latest release: %v", err)
	}

	if !forceUpgrade {
		cachedRelease, err := loadCachedRelease()
		if err == nil && release.TagName <= cachedRelease.TagName {
			log.Println("Already lastest stable version.")
			return
		}
	}

	// 确定当前系统架构
	arch := runtime.GOARCH
	if arch != "amd64" && arch != "arm64" {
		log.Fatalf("Unsupported architecture: %s", arch)
	}

	// 查找匹配的资源文件
	var downloadURL string
	for _, asset := range release.Assets {
		if strings.Contains(asset.Name, "darwin") && strings.Contains(asset.Name, arch) {
			downloadURL = asset.BrowserDownloadURL
			break
		}
	}

	if downloadURL == "" {
		log.Fatalf("No matching binary found for %s-%s", runtime.GOOS, arch)
	}

	log.Printf("Downloading: %s", downloadURL)

	// 创建临时文件
	tmpFile, err := os.CreateTemp("", "federate-*")
	if err != nil {
		log.Fatalf("Failed to create temporary file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	// 下载最新的二进制文件
	if err := downloadFile(tmpFile.Name(), downloadURL); err != nil {
		log.Fatalf("Failed to download the latest version: %v", err)
	}

	homebrewPrefix := os.Getenv("HOMEBREW_PREFIX")
	if homebrewPrefix == "" {
		homebrewPrefix = "/usr/local"
	}
	binPath := filepath.Join(homebrewPrefix, "bin", "federate")
	log.Printf("Installing to %s", binPath)

	// 替换当前的二进制文件
	if err := os.Rename(tmpFile.Name(), binPath); err != nil {
		log.Fatalf("Failed to replace the binary: %v", err)
	}

	if err := os.Chmod(binPath, 0755); err != nil {
		log.Printf("Warning: Failed to set executable permissions: %v", err)
		log.Println("You may need to manually set the executable permission using 'chmod a+x'")
	}

	// 缓存结果
	cacheRelease(release)

	fmt.Printf("🍺 Upgrade successful, cost: %s\n", time.Since(t0))
}

func getLatestRelease() (*GithubRelease, error) {
	client := &http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}

	resp, err := client.Get("https://github.com/funkygao/federate/releases/latest")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusFound {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	location := resp.Header.Get("Location")
	if location == "" {
		return nil, fmt.Errorf("location header not found")
	}

	parts := strings.Split(location, "/")
	if len(parts) == 0 {
		return nil, fmt.Errorf("invalid location header")
	}

	tagName := parts[len(parts)-1]

	release := &GithubRelease{
		TagName: tagName,
	}

	// 获取资源列表
	assetsURL := fmt.Sprintf("https://github.com/funkygao/federate/releases/expanded_assets/%s", tagName)
	assetsResp, err := http.Get(assetsURL)
	if err != nil {
		return nil, err
	}
	defer assetsResp.Body.Close()

	body, err := io.ReadAll(assetsResp.Body)
	if err != nil {
		return nil, err
	}

	// 解析 HTML 以获取资源列表
	// 这里需要一个简单的 HTML 解析来提取下载链接
	// 为了简化，我们可以使用字符串操作来提取链接
	lines := strings.Split(string(body), "\n")
	for _, line := range lines {
		if strings.Contains(line, "href") && strings.Contains(line, "download") {
			parts := strings.Split(line, "href=\"")
			if len(parts) > 1 {
				downloadURL := strings.Split(parts[1], "\"")[0]
				name := filepath.Base(downloadURL)
				release.Assets = append(release.Assets, struct {
					Name               string `json:"name"`
					BrowserDownloadURL string `json:"browser_download_url"`
				}{
					Name:               name,
					BrowserDownloadURL: "https://github.com" + downloadURL,
				})
			}
		}
	}

	return release, nil
}

func cacheRelease(release *GithubRelease) error {
	data, err := json.Marshal(release)
	if err != nil {
		return err
	}
	return os.WriteFile(getCachePath(), data, 0644)
}

func loadCachedRelease() (*GithubRelease, error) {
	data, err := os.ReadFile(getCachePath())
	if err != nil {
		return nil, err
	}
	var release GithubRelease
	if err := json.Unmarshal(data, &release); err != nil {
		return nil, err
	}
	return &release, nil
}

func getCachePath() string {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		log.Printf("Failed to get home directory: %v", err)
		return cacheFile
	}
	return filepath.Join(homeDir, cacheDir, cacheFile)
}

func ensureCacheDir() error {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get home directory: %v", err)
	}
	cacheDir := filepath.Join(homeDir, cacheDir)
	return os.MkdirAll(cacheDir, 0755)
}

func downloadFile(filepath string, url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

func init() {
	upgradeCmd.Flags().BoolVarP(&forceUpgrade, "force", "f", false, "Force upgrade")
}
