package ast

import "federate/pkg/primitive"

type Filter interface {
	Apply(info *Info) *Info
}

type FilterChain struct {
	filters []Filter
}

func NewFilterChain(filters ...Filter) *FilterChain {
	return &FilterChain{filters: filters}
}

func (fc *FilterChain) Apply(info *Info) *Info {
	for _, filter := range fc.filters {
		info = filter.Apply(info)
	}
	return info
}

type IgnoreInterfacesFilter struct {
	IgnoredInterfaces *primitive.StringSet
}

func (f *IgnoreInterfacesFilter) Apply(info *Info) *Info {
	filteredInterfaces := make(map[string][]string)
	for class, interfaces := range info.Interfaces {
		var filtered []string
		for _, iface := range interfaces {
			if !f.IgnoredInterfaces.Contains(iface) {
				filtered = append(filtered, iface)
			}
		}
		if len(filtered) > 0 {
			filteredInterfaces[class] = filtered
		}
	}
	info.Interfaces = filteredInterfaces
	return info
}

type IgnoreAnnotationsFilter struct {
	IgnoredAnnotations *primitive.StringSet
}

func (f *IgnoreAnnotationsFilter) Apply(info *Info) *Info {
	var filteredAnnotations []string
	for _, annotation := range info.Annotations {
		if !f.IgnoredAnnotations.Contains(annotation) {
			filteredAnnotations = append(filteredAnnotations, annotation)
		}
	}
	info.Annotations = filteredAnnotations
	return info
}
