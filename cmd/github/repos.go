package github

import (
	"context"
	"fmt"
	"os"
	"sort"

	"github.com/google/go-github/v38/github"
	"github.com/spf13/cobra"
	"golang.org/x/oauth2"
)

var topReposCmd = &cobra.Command{
	Use:   "explore",
	Short: "Explore top GitHub repositories by stars",
	Run: func(cmd *cobra.Command, args []string) {
		token := getToken()
		if err := displayTopRepos(language, topN, token); err != nil {
			fmt.Printf("Error: %v\n", err)
		}
	},
}

func init() {
	topReposCmd.Flags().IntVarP(&topN, "top", "t", 20, "Number of top repositories to display")
	topReposCmd.Flags().StringVarP(&language, "language", "l", "", "Programming language to filter by")
	topReposCmd.Flags().StringVarP(&tokenFlag, "token", "k", "", "GitHub Personal Access Token (overrides GITHUB_TOKEN env variable)")
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

	for i, repo := range repos {
		fmt.Printf("%d. %s (%d stars)\n   %s\n\n", i+1, *repo.FullName, *repo.StargazersCount, *repo.HTMLURL)
	}
}
