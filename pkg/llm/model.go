package llm

const (
	Model750B = "Chatrhino-750B" // 60s
	Model470B = "Chatrhino-470B-preview-0103"
	Model81B  = "Chatrhino-81B-Pro" // 6s
	Model14B  = "Chatrhino-14B"     // 2s
)

type Config struct {
	ProjectID string
	APIToken  string
	GPTURL    string
	GPTAPIKey string
	Model     string
}

func DefaultConfig() Config {
	return Config{
		ProjectID: "475447",
		APIToken:  "MDDKFW1dm2SLL2pDjhWf",
		GPTURL:    "http://api.chatrhino.jd.com/api/v1/chat/completions",
		GPTAPIKey: "5rFc1Wj7F2k67tvbgbFjKJEYycDMeWDD",
		Model:     Model14B,
	}
}
