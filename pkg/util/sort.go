package util

import (
	"reflect"
	"sort"
)

func MapSortedStringKeys(m any) []string {
	v := reflect.ValueOf(m)
	if v.Kind() != reflect.Map {
		return nil
	}

	keys := make([]string, 0, v.Len())
	for _, k := range v.MapKeys() {
		if k.Kind() == reflect.String {
			keys = append(keys, k.String())
		}
	}
	sort.Strings(keys)
	return keys
}
