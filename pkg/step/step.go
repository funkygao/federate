package step

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/fatih/color"
)

var AutoConfirm bool

type Step struct {
	Name string
	Fn   func()
}

func Run(steps []Step) {
	totalSteps := len(steps)
	for i, step := range steps {
		promptToProceed(i+1, totalSteps, step.Name)
		step.Fn()
	}
}

func promptToProceed(seq, total int, stepName string) {
	c := color.New(color.FgMagenta)
	c.Printf("Step [%d/%d] %s", seq, total, stepName)
	if AutoConfirm {
		fmt.Println()
		return
	}
	reader := bufio.NewReader(os.Stdin)
	reader.ReadString('\n')
}

func ConfirmAction(prompt string) bool {
	fmt.Printf("%s (y/N): ", prompt)
	var response string
	_, err := fmt.Scanln(&response)
	if err != nil && response != "" {
		log.Fatalf("Error reading response: %v", err)
	}

	return strings.ToLower(response) == "y" || strings.ToLower(response) == "yes"
}
