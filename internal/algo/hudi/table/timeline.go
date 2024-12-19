package table

import (
	"sort"
	"sync"
	"time"
)

type Timeline struct {
	mu      sync.RWMutex
	commits []Commit
}

func NewTimeline() *Timeline {
	return &Timeline{
		commits: make([]Commit, 0),
	}
}

func (t *Timeline) AddCommit(commit Commit) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.commits = append(t.commits, commit)
	sort.Slice(t.commits, func(i, j int) bool {
		return t.commits[i].Timestamp.Before(t.commits[j].Timestamp)
	})
}

func (t *Timeline) GetCommitsAfter(timestamp time.Time) []Commit {
	t.mu.RLock()
	defer t.mu.RUnlock()
	var result []Commit
	for _, commit := range t.commits {
		if commit.Timestamp.After(timestamp) {
			result = append(result, commit)
		}
	}
	return result
}
