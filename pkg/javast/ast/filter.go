package ast

import (
	"strings"
)

func (i *Info) ApplyFilters(filters ...Filter) *Info {
	chain := NewFilterChain(filters...)
	return chain.Apply(i)
}

type Filter interface {
	Apply(info *Info) *Info
}

func DefaultFilters() []Filter {
	return []Filter{&InterfacesFilter{}, &AnnotationsFilter{}, &CompositionsFilter{}}
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

type InterfacesFilter struct {
}

func (f *InterfacesFilter) Apply(info *Info) *Info {
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

type AnnotationsFilter struct {
}

func (f *AnnotationsFilter) Apply(info *Info) *Info {
	var filteredAnnotations []string
	for _, annotation := range info.Annotations {
		if !ignoredAnnotations.Contains(annotation) {
			filteredAnnotations = append(filteredAnnotations, annotation)
		}
	}
	info.Annotations = filteredAnnotations
	return info
}

type CompositionsFilter struct {
}

func (f *CompositionsFilter) Apply(info *Info) *Info {
	var compositions []CompositionInfo
	for _, comp := range info.Compositions {
		if !ignoredCompositionTypes.Contains(comp.ComposedClass) && !strings.Contains(comp.ComposedClass, "<") {
			compositions = append(compositions, comp)
		}
	}
	info.Compositions = compositions
	return info
}
