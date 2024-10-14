# federate

`federate` functions as a specialized 'immunosuppressive compiler', employing heuristic rule-based approach to seamlessly consolidate multiple microservices into a single JVM deployment. Users simply author declarative manifests, while the system autonomously manages the 'transplantation process' - reconciling both direct and indirect 'tissue rejections' through advanced code rewriting, a two-phase generation process, and pre-runtime JAR compatibility analysis. This approach effectively optimizes inter-service communication and ensures harmonious integration of diverse service components.

## 关键设计

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
                                          │
                        ┌─────────────────┴─────────────────┐
                        │              manifest             │
                        │ (Declarative Development Paradigm)│
                        │                                   │
                        │    - Define desired state         │
                        │    - Debug and test               │
                        │    - Version control              │
                        └─────────────────┬─────────────────┘
                        ┌─────────────────┼─────────────────┐
              ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
              │   component1    │ │   component2    │ │   componentN    │
              │ (git submodule) │ │ (git submodule) │ │ (git submodule) │
              └─────────────────┘ └─────────────────┘ └─────────────────┘
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
federate microservice scaffold-monolith
```

### manifest
