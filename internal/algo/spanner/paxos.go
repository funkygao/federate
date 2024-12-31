package main

type Paxos interface {
	Propose(value interface{}) (bool, error)
	Accept(value interface{}) (bool, error)
	Learn(value interface{}) error
}

type paxos struct {
	acceptors []Acceptor
	learners  []Learner
}

func NewPaxos(acceptors []Acceptor, learners []Learner) Paxos {
	return &paxos{acceptors: acceptors, learners: learners}
}

func (p *paxos) Propose(value interface{}) (bool, error) {
	return true, nil
}

func (p *paxos) Accept(value interface{}) (bool, error) {
	return true, nil
}

func (p *paxos) Learn(value interface{}) error {
	return nil
}

type Acceptor interface {
	Promise(proposalID int) (bool, int, interface{})
	Accept(proposalID int, value interface{}) bool
}

type Learner interface {
	Learn(value interface{})
}
