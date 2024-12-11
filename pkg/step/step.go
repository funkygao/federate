package step

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/schollz/progressbar/v3"
)

var AutoConfirm bool

type Step struct {
	Name           string
	Fn             func()
	FnWithProgress func(*progressbar.ProgressBar)
}

func Run(steps []Step) {
	totalSteps := len(steps)
	for i, step := range steps {
		promptToProceed(i+1, totalSteps, step.Name)

		if step.FnWithProgress != nil {
			bar := progressbar.NewOptions(100,
				progressbar.OptionEnableColorCodes(true),
				progressbar.OptionSetWidth(15),
				progressbar.OptionSetDescription(fmt.Sprintf("[magenta]Step [%d/%d] %s[reset]", i+1, totalSteps, step.Name)),
				progressbar.OptionSetTheme(progressbar.Theme{
					Saucer:        "[green]=[reset]",
					SaucerHead:    "[green]>[reset]",
					SaucerPadding: " ",
					BarStart:      "[",
					BarEnd:        "]",
				}))

			step.FnWithProgress(bar)

			bar.Finish()
		} else {
			step.Fn()
		}

	}
}

func promptToProceed(seq, total int, stepName string) {
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
