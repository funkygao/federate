package main

type Region interface {
	ID() string
	AddReplica(Replica)
	Replicas() []Replica
}

type region struct {
	id       string
	replicas []Replica
}

func NewRegion(id string) Region {
	return &region{id: id, replicas: []Replica{}}
}

func (r *region) ID() string {
	return r.id
}

func (r *region) AddReplica(replica Replica) {
	r.replicas = append(r.replicas, replica)
}

func (r *region) Replicas() []Replica {
	return r.replicas
}
