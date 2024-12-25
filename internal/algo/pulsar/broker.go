package main

type Broker interface {
	Publish(topic string, msg Message) error
	Subscribe(topic, subscriptionName string, subType SubscriptionType) (Subscription, error)
}
