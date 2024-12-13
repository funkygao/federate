package github

import (
	"context"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	"federate/pkg/tabular"
	"github.com/google/go-github/v38/github"
	"github.com/spf13/cobra"
	"golang.org/x/oauth2"
)

var topReposCmd = &cobra.Command{
	Use:   "repos",
	Short: "Discover top-starred GitHub repositories",
	Long: `Explore and discover the most popular repositories on GitHub based on star count.

This command allows you to view the top repositories overall or filtered by a specific programming language.

To use this command, you need to set up a GitHub Personal Access Token:

1. Create a token at https://github.com/settings/tokens
   - You only need 'public_repo' scope for this command
2. Set the token as an environment variable:
   export GITHUB_TOKEN=your_token_here

Alternatively, you can pass the token directly using the --token flag.`,
	Run: func(cmd *cobra.Command, args []string) {
		token := getToken()
		if err := displayTopRepos(language, topN, token); err != nil {
			fmt.Printf("Error: %v\n", err)
		}
	},
}

func getToken() string {
	if tokenFlag != "" {
		return tokenFlag
	}
	return os.Getenv("GITHUB_TOKEN")
}

func displayTopRepos(lang string, n int, token string) error {
	ctx := context.Background()
	ts := oauth2.StaticTokenSource(
		&oauth2.Token{AccessToken: token},
	)
	tc := oauth2.NewClient(ctx, ts)

	client := github.NewClient(tc)

	opt := &github.SearchOptions{
		Sort:  "stars",
		Order: "desc",
		ListOptions: github.ListOptions{
			PerPage: n,
		},
	}

	query := "stars:>1"
	if lang != "" {
		query += " language:" + lang
	}

	result, _, err := client.Search.Repositories(ctx, query, opt)
	if err != nil {
		return fmt.Errorf("error searching repositories: %v", err)
	}

	if len(result.Repositories) == 0 {
		return fmt.Errorf("no repositories found")
	}

	printRepos(result.Repositories)
	return nil
}

func printRepos(repos []*github.Repository) {
	sort.Slice(repos, func(i, j int) bool {
		return *repos[i].StargazersCount > *repos[j].StargazersCount
	})

	header := []string{"Rank", "Repository", "Stars", "URL"}
	var rows [][]string

	for i, repo := range repos {
		row := []string{
			strconv.Itoa(i + 1),
			*repo.FullName,
			addThousandsSeparator(*repo.StargazersCount),
			*repo.HTMLURL,
		}
		rows = append(rows, row)
	}

	fmt.Println("Top GitHub Repositories:")
	tabular.Display(header, rows, false)
}

func addThousandsSeparator(n int) string {
	str := strconv.FormatInt(int64(n), 10)
	if len(str) < 4 {
		return str
	}
	var result []string
	for i := len(str); i > 0; i -= 3 {
		start := i - 3
		if start < 0 {
			start = 0
		}
		result = append([]string{str[start:i]}, result...)
	}
	return strings.Join(result, ",")
}

func init() {
	topReposCmd.Flags().IntVarP(&topN, "top", "t", 50, "Number of top repositories to display")
	topReposCmd.Flags().StringVarP(&language, "language", "l", "", "Programming language to filter by")
	topReposCmd.Flags().StringVarP(&tokenFlag, "token", "k", "", "GitHub Personal Access Token (overrides GITHUB_TOKEN env variable)")
}
