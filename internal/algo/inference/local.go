package main

import (
	"sort"
)

// LocalScheduler manages request scheduling within a prefill instance
type LocalScheduler struct {
	rawQueue       []Request
	scheduledQueue []Request
	policy         SchedulingPolicy
}

type SchedulingPolicy string

const (
	FCFS SchedulingPolicy = "FCFS"
	SJF  SchedulingPolicy = "SJF"
	LJF  SchedulingPolicy = "LJF"
)

func NewLocalScheduler(policy SchedulingPolicy) *LocalScheduler {
	return &LocalScheduler{
		policy: policy,
	}
}

func (s *LocalScheduler) AddRequest(request Request) {
	s.rawQueue = append(s.rawQueue, request)
}

func (s *LocalScheduler) ScheduleRequests(requests []Request) {
	switch s.policy {
	case FCFS:
		s.scheduleFCFS()
	case SJF:
		s.scheduleSJF()
	case LJF:
		s.scheduleLJF()
	}
}

func (s *LocalScheduler) scheduleFCFS() {
	s.scheduledQueue = append(s.scheduledQueue, s.rawQueue...)
	s.rawQueue = nil
}

func (s *LocalScheduler) scheduleSJF() {
	s.scheduledQueue = append(s.scheduledQueue, s.rawQueue...)
	sort.Slice(s.scheduledQueue, func(i, j int) bool {
		return s.scheduledQueue[i].PromptTokens < s.scheduledQueue[j].PromptTokens
	})
	s.rawQueue = nil
}

func (s *LocalScheduler) scheduleLJF() {
	s.scheduledQueue = append(s.scheduledQueue, s.rawQueue...)
	sort.Slice(s.scheduledQueue, func(i, j int) bool {
		return s.scheduledQueue[i].PromptTokens > s.scheduledQueue[j].PromptTokens
	})
	s.rawQueue = nil
}

func (s *LocalScheduler) GetScheduledRequests() []Request {
	return s.scheduledQueue
}
