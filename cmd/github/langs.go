package github

import (
	"context"
	"fmt"
	"net/url"
	"sort"

	"github.com/google/go-github/v38/github"
	"github.com/spf13/cobra"
	"golang.org/x/oauth2"
)

var languagesCmd = &cobra.Command{
	Use:   "languages",
	Short: "Analyze popular programming languages on GitHub",
	Run: func(cmd *cobra.Command, args []string) {
		token := getToken()
		if err := displayPopularLanguages(topN, token); err != nil {
			fmt.Printf("Error: %v\n", err)
		}
	},
}

func displayPopularLanguages(topN int, token string) error {
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
			PerPage: 100, // Fetch 100 repositories to get a good sample
		},
	}

	query := "stars:>1000" // Focus on popular repositories
	result, _, err := client.Search.Repositories(ctx, query, opt)
	if err != nil {
		return fmt.Errorf("error searching repositories: %v", err)
	}

	if len(result.Repositories) == 0 {
		return fmt.Errorf("no repositories found")
	}

	languages := make(map[string]int)
	for _, repo := range result.Repositories {
		if repo.Language != nil {
			languages[*repo.Language]++
		}
	}

	printLanguages(languages, topN)
	return nil
}

func printLanguages(languages map[string]int, n int) {
	type langCount struct {
		Name  string
		Count int
	}

	var langCounts []langCount
	for lang, count := range languages {
		langCounts = append(langCounts, langCount{Name: lang, Count: count})
	}

	sort.Slice(langCounts, func(i, j int) bool {
		return langCounts[i].Count > langCounts[j].Count
	})

	fmt.Printf("Top %d popular programming languages on GitHub:\n", n)
	fmt.Println("(Language names can be used directly with 'github explore -l' command)")
	for i, lc := range langCounts {
		if i >= n {
			break
		}
		// Use url.QueryEscape to ensure the language name is properly formatted for URL use
		escapedName := url.QueryEscape(lc.Name)
		fmt.Printf("%d. %s\n   Use with: github explore -l %s\n", i+1, lc.Name, escapedName)
	}
}

func init() {
	languagesCmd.Flags().IntVarP(&topN, "top", "t", 20, "Number of top languages to display")
	languagesCmd.Flags().StringVarP(&tokenFlag, "token", "k", "", "GitHub Personal Access Token (overrides GITHUB_TOKEN env variable)")
}
