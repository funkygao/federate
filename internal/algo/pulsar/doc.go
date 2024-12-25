/*
Package main implements a simplified version of Apache Pulsar, demonstrating its core concepts and architecture.

# Pulsar 的核心机制：存算分离

Apache Pulsar 的一个关键设计理念是存算分离（Separation of Storage and Compute）。这种架构将消息的存储和处理分离，
提供了更高的灵活性、可扩展性和性能。

核心组件和存算分离实现：

1. Broker（计算层）：

  - 负责消息的接收、分发和管理。

  - 处理客户端连接、订阅管理和消息路由。

  - 维护轻量级状态和多层缓存机制，包括内存缓存和页面缓存。

  - 不直接存储消息数据，而是与存储层（BookKeeper）交互。

    Broker 的状态和缓存：
    a) 轻量级状态：维护活跃连接和订阅信息。
    b) 元数据缓存：缓存 topic 和 subscription 元数据，减少 ZooKeeper 访问。
    c) 消息缓存：实现多层次缓存（内存缓存、页面缓存）提高读取性能。
    d) 预读取机制：预测性读取后续消息，优化顺序读取。
    e) 写入缓冲：批量处理写入请求，提高写入效率。

2. BookKeeper（存储层）：
  - 由多个 Bookie 节点组成，负责消息的持久化存储。
  - 实现了高性能、低延迟的分布式日志存储系统。
  - 使用 Ledger 作为基本存储单元，提供高效的数据写入和检索。

3. ZooKeeper：
  - 用于元数据管理和协调。
  - 存储 topic、subscription 等元数据信息。
  - 管理 Broker 和 Bookie 的成员关系和负载均衡。

存储结构层次：
Topic -> Partition -> TimeSegment -> Ledger -> Entry

- Topic: 消息的逻辑通道。
- Partition: Topic 的子集，允许并行处理。
- TimeSegment: Partition 的时间片段，便于数据管理和清理。通常包含多个 Ledger。
- Ledger: BookKeeper 中的基本存储单元，代表一段连续的消息日志。类似于 Kafka 中的 Segment。
- Entry: Ledger 中的单个消息或一组消息。

存算分离的优势：

1. 灵活的扩展：存储层和计算层可以独立扩展。
2. 高性能：Broker 专注于消息处理，BookKeeper 提供高效存储。
3. 数据持久性和可靠性：多副本存储确保数据安全。
4. 高效的存储管理：支持分层存储和灵活的数据保留策略。
5. 多租户支持：不同 Topic 可使用不同存储策略，易于资源隔离。

消息流程：

1. 生产者发送消息到 Broker。
2. Broker 将消息写入 BookKeeper（创建新的 Ledger 或追加到现有 Ledger）。
3. BookKeeper 确认写入成功。
4. Broker 确认消息已持久化。
5. 消费者从 Broker 请求消息。
6. Broker 从缓存或 BookKeeper 读取消息。
7. Broker 将消息发送给消费者。
8. 消费者确认消息处理完成。

延迟消息和消息重试：

- 延迟消息存储在特殊的延迟队列中。
- 到期时，消息被移动到目标 Topic 进行处理。
- 消息重试利用类似机制，将失败的消息重新安排在未来的时间点处理。

总结：
Pulsar 的存算分离架构通过将消息存储（BookKeeper）与消息处理（Broker）分离，实现了高度的灵活性和可扩展性。
Broker 的轻状态设计和多层缓存机制在保持可扩展性的同时提供了优秀的性能。BookKeeper 的分布式存储保证了数据的
持久性和可靠性。这种架构使 Pulsar 特别适合大规模、高吞吐量的消息处理场景，同时支持复杂的消息处理逻辑。
*/
package main
