package table

import (
	"encoding/json"
	"fmt"
	"reflect"
	"time"
)

type FieldType string

const (
	StringType   FieldType = "string"
	IntType      FieldType = "int"
	FloatType    FieldType = "float"
	BoolType     FieldType = "bool"
	DateTimeType FieldType = "datetime"
)

type SchemaField struct {
	Name string
	Type FieldType
}

type Schema struct {
	Fields []SchemaField
}

func (s *Schema) Validate(record Record) error {
	for _, field := range s.Fields {
		value, ok := record.Fields[field.Name]
		if !ok {
			return fmt.Errorf("missing field: %s", field.Name)
		}

		switch field.Type {
		case StringType:
			if _, ok := value.(string); !ok {
				return fmt.Errorf("invalid type for field %s: expected string, got %v", field.Name, reflect.TypeOf(value))
			}
		case IntType:
			switch v := value.(type) {
			case int:
				// Value is already an int, okay
			case float64:
				// Check if float64 value is an integer
				if v == float64(int(v)) {
					// Value can be safely converted to int
				} else {
					return fmt.Errorf("invalid type for field %s: expected int, got float64 with non-integer value", field.Name)
				}
			default:
				return fmt.Errorf("invalid type for field %s: expected int, got %v", field.Name, reflect.TypeOf(value))
			}
		case FloatType:
			if _, ok := value.(float64); !ok {
				return fmt.Errorf("invalid type for field %s: expected float, got %v", field.Name, reflect.TypeOf(value))
			}
		case BoolType:
			if _, ok := value.(bool); !ok {
				return fmt.Errorf("invalid type for field %s: expected bool, got %v", field.Name, reflect.TypeOf(value))
			}
		case DateTimeType:
			// Handle date-time values
			switch v := value.(type) {
			case time.Time:
				// ok
			case string:
				// Try to parse the string as a time
				if _, err := time.Parse(time.RFC3339, v); err != nil {
					return fmt.Errorf("invalid date-time format for field %s: %v", field.Name, err)
				}
			default:
				return fmt.Errorf("invalid type for field %s: expected datetime, got %v", field.Name, reflect.TypeOf(value))
			}
		default:
			return fmt.Errorf("unknown field type: %s", field.Type)
		}
	}
	return nil
}

func (s *Schema) ToJSON() ([]byte, error) {
	return json.Marshal(s)
}

func SchemaFromJSON(data []byte) (*Schema, error) {
	var schema Schema
	err := json.Unmarshal(data, &schema)
	if err != nil {
		return nil, err
	}
	return &schema, nil
}
