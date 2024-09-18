package prompt

import (
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/pkoukk/tiktoken-go"
)

func CountTokensInK(text string) float64 {
	// 获取当前用户的 Home 目录
	homeDir, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	// 设置环境变量 TIKTOKEN_CACHE_DIR
	cacheDir := filepath.Join(homeDir, ".tiktoken")
	if _, err := os.Stat(cacheDir); os.IsNotExist(err) {
		fmt.Println("首次计算token数量需要下载tiktoken字典，3.4MB，因此可能会比较慢。")
	}
	err = os.Setenv("TIKTOKEN_CACHE_DIR", cacheDir)
	if err != nil {
		panic(err)
	}

	// 获取编码器
	enc, err := tiktoken.EncodingForModel("gpt-4o")
	if err != nil {
		panic(err)
	}

	// 编码文本并计算 token 数量
	tokens := enc.Encode(text, nil, nil)
	tokenCount := float64(len(tokens)) / 1000

	// 返回以千为单位的 token 数量，精确到小数点后2位
	return math.Round(tokenCount*100) / 100
}
