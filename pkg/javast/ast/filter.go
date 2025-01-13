package ast

import (
	"strings"

	"federate/pkg/primitive"
)

func (i *Info) ApplyFilters(filters ...Filter) *Info {
	chain := NewFilterChain(filters...)
	return chain.Apply(i)
}

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
}

func (f *IgnoreInterfacesFilter) Apply(info *Info) *Info {
	filteredInterfaces := make(map[string][]string)
	for class, interfaces := range info.Interfaces {
		var filtered []string
		for _, iface := range interfaces {
			if !ignoredInterfaces.Contains(iface) {
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
}

func (f *IgnoreAnnotationsFilter) Apply(info *Info) *Info {
	var filteredAnnotations []string
	for _, annotation := range info.Annotations {
		if !ignoredAnnotations.Contains(annotation) {
			filteredAnnotations = append(filteredAnnotations, annotation)
		}
	}
	info.Annotations = filteredAnnotations
	return info
}

type IgnoreCompositionsFilter struct {
}

func (f *IgnoreAnnotationsFilter) Apply(info *Info) *Info {
	var compositions []CompositionInfo
	for _, comp = range info.Compositions {
		if !ignoredCompositionTypes.Contains(comp.ComposedClass) && !strings.Contains(comp.ComposedClass, "<") {
			compositions = append(compositions, comp)
		}
	}
	info.Compositions = compositions
	return info
}
