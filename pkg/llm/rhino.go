package llm

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
)

// 大模型请求结构体
type gptRequest struct {
	Model       string    `json:"model"`
	Messages    []message `json:"messages"`
	Temperature float64   `json:"temperature"`
	TopP        float64   `json:"top_p"`
	N           int       `json:"n"`
	Stream      bool      `json:"stream"`
	MaxTokens   int       `json:"max_tokens"`
	// 其他参数...
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// 调用京东言犀大模型API
func CallRhino(prompt string) (*AIResponse, error) {
	config := DefaultConfig()
	request := gptRequest{
		Model: config.Model,
		Messages: []message{
			{Role: "user", Content: prompt},
		},
		Temperature: 0.1,
		TopP:        0.8,
		N:           1,
		Stream:      false,
		MaxTokens:   1000000,
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", config.GPTURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}

	req.Header.Set("api-key", config.GPTAPIKey)
	req.Header.Set("Accept-Charset", "UTF-8")
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var aiResponse AIResponse
	err = json.Unmarshal(body, &aiResponse)
	if err != nil {
		return nil, err
	}

	return &aiResponse, nil
}
