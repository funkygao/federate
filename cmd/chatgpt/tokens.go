package chatgpt

import (
	"fmt"
	"io/ioutil"
	"log"

	"federate/pkg/prompt"
	"github.com/spf13/cobra"
)

var tokensCmd = &cobra.Command{
	Use:   "tokens",
	Short: "Calculate a prompt text file tokens count",
	Long: `The tokens command calculates a prompt text file tokens count.

Example usage:
  federate chatgpt tokens <txt file>`,
	Run: func(cmd *cobra.Command, args []string) {
		countTokens(args[0])
	},
}

func countTokens(fn string) {
	content, err := ioutil.ReadFile(fn)
	if err != nil {
		log.Fatalf("Failed to read file: %v", err)
	}

	tokenCount := prompt.CountTokensInK(string(content))
	fmt.Printf("The file %s contains %.2fK tokens\n", fn, tokenCount)
}
