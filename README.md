# federate

Microservices architecture, despite its advantages in scalability, flexibility, and deployment speed, can introduce significant challenges:
- Resource Redundancy: Proliferation of similar resources across services, leading to inefficient utilization and increased costs.
- Code Duplication: Repetition of logic across multiple services, violating the DRY principle and complicating maintenance.
- Over-granularization: Excessive service decomposition, potentially resulting in "nanoservices" that increase complexity and network overhead.

`federate` functions as a specialized 'immunosuppressive compiler', employing heuristic rule-based approach to seamlessly consolidate multiple microservices into a single JVM deployment. Users simply author declarative manifests, while the system autonomously manages the 'transplantation process' - reconciling both direct and indirect 'tissue rejections' through Runtime Code Generation, Code Instrumentation, Static Analysis and Rewriting, a two-phase generation process, and pre-runtime JAR compatibility analysis. This approach effectively optimizes inter-service communication and ensures harmonious integration of diverse service components.

This approach decouples logical boundaries from physical deployment, offering flexibility while maintaining microservice architecture benefits. The result is a system combining microservice agility with consolidated architecture efficiency.

## Quickstart

```bash
make install
```

or

```bash
brew install funkygao/stable/federate
federate version upgrade
```

## Under the hood

### Conceptual Model

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

>Why not directly generate the target system? Why is the fusion-starter intermediate project necessary?

- It allows users to perform extension development.
- It allows users to provide taint files.
- It facilitates the target system in excluding conflicting packages.

### Fusion Projects

Fusion Projects acts as logical monoliths, offload the decisions of how to distribute and run applications to `federate` phase.
This approach decouples logical boundaries (how code is written) from physical boundaries (how code is deployed).

Additionally, Fusion Projects leverage the `federate` command to automatically generate scaffolding. 
This powerful feature streamlines the development process by creating a standardized project structure and boilerplate code. 

```bash
federate microservice scaffold -h
```

### Instrumentation

| Target | Instrumentation |
|--------|-----------------|
| Java Source Code | - Replace `@Resource` with `@Autowired`, [`@Qualifier`]<br>- Transform `@RequestMapping`, `@Service`, `@Component`, `@ImportResource`, `@Value`, `@ConfigurationProperties`, `@Transational`<br>- Detect `System.getProperty()`, `getBean(beanId)` |
| Resource Files | - Merge RPC Consumer Configurations<br>- Detect RPC Provider alias conflicts<br>- Allow user-specified imports<br>- Resolve Bean Conflicts |
| Property Files | - Segregate conflicting keys |
| pom.xml | - Disable `spring-boot-maven-plugin` to enable post-`mvn install` usage |
| Code Generation | - Makefile<br>- fusion-starter runtime<br>- Java AST Manipulate Project<br>- Fusion Maven Project |

