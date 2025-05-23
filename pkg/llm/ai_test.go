package llm

import (
	"testing"
)

func TestCallAI(t *testing.T) {
	response, err := CallRhino("给我做一首关于下雨的唐诗")
	if err != nil {
		t.Fatalf("CallAI failed: %v", err)
	}

	if len(response.Choices) > 0 {
		poem := response.Choices[0].Message.Content
		t.Logf("Generated poem:\n%s", poem)
	} else {
		t.Error("No choices returned from AI")
	}
}
