# federate

Microservices architecture, despite its advantages in scalability, flexibility, and deployment speed, can introduce significant challenges:
- Resource Redundancy: Proliferation of similar resources across services, leading to inefficient utilization and increased costs.
- Code Duplication: Repetition of logic across multiple services, violating the DRY principle and complicating maintenance.
- Over-granularization: Excessive service decomposition, potentially resulting in "nanoservices" that increase complexity and network overhead.

`federate` functions as a specialized 'immunosuppressive compiler', employing heuristic rule-based approach to seamlessly consolidate multiple microservices into a single JVM deployment. Users simply author declarative manifests, while the system autonomously manages the 'transplantation process' - reconciling both direct and indirect 'tissue rejections' through Runtime Code Generation, Code Instrumentation, Static Analysis and Rewriting, a two-phase generation process, and pre-runtime JAR compatibility analysis. This approach effectively optimizes inter-service communication and ensures harmonious integration of diverse service components.

This approach decouples logical boundaries from physical deployment, offering flexibility while maintaining microservice architecture benefits. The result is a system combining microservice agility with consolidated architecture efficiency.

## Quickstart: Install and Run **federate**

```bash
brew install go
make install
federate
```

## 核心设计

### 概念模型

```
                        ┌──────────────────────────────────┐
                        │        federated system          │
                        └──────────────────────────────────┘
    ┌────────────────────────────────────┼────────────────────────────────────┐
    │            ┌───────────────────────┴───────────────────────┐            │
    │            │               fusion-starter                  │            │
    │            └───────────────────────┬───────────────────────┘            │
    │                     ┌──────────────┴──────────────┐                     │
    │     ┌───────────────────────┐            ┌───────────────────────┐      │
    │     │    User Extension     │            │    Provide Taint      │      │
    │     │      Development      │            │        Files          │      │
    │     └───────────────────────┘            └───────────────────────┘      │
    └─────────────────────────────────────────────────────────────────────────┘
                        ┌─────────────────┴──────────────────┐
                        │              manifest              │
                        │ (Declarative Development Paradigm) │
                        │                                    │
                        │    - Define desired state          │
                        │    - Debug and test                │
                        │    - Version control               │
                        └─────────────────┬──────────────────┘
                        ┌─────────────────┼─────────────────┐
    ┌───────────────────────┐┌───────────────────────┐┌───────────────────────┐
    │     Component 1       ││     Component 2       ││     Component N       │
    │   (Git Submodule)     ││   (Git Submodule)     ││   (Git Submodule)     │
    │                       ││                       ││                       │
    │ Code Instrumentation  ││ Code Instrumentation  ││ Code Instrumentation  │
    │ (Uncommitted Changes) ││ (Uncommitted Changes) ││ (Uncommitted Changes) │
    └───────────────────────┘└───────────────────────┘└───────────────────────┘
```

### fusion-starter

>直接生成目标系统不好吗，为什么需要 `fusion-starter` 这个中间工程？

- 它允许用户进行扩展开发
   - `pom.xml` 只能通过 `manifest.yaml` 声明，不得手工修改
- 它允许用户提供 `taint files`
- 方便目标系统排除一下冲突包
   - 例如，slf4j-log4j12 与 logback-classic冲突，如果直接生成目标，则必须找到谁引入的才能排除冲突

### Fusion Projects

Fusion Projects acts as logical monoliths, offload the decisions of how to distribute and run applications to `federate` phase.
This approach decouples logical boundaries (how code is written) from physical boundaries (how code is deployed).

Additionally, Fusion Projects leverage the `federate` command to automatically generate scaffolding. 
This powerful feature streamlines the development process by creating a standardized project structure and boilerplate code. 

```bash
federate microservice scaffold-monolith -h
```

### manifest

Top level fields:
- federated
- components
- fusion-starter
- deployment

## TODO

- manifest.components.springProfile 需要为不同环境配置
