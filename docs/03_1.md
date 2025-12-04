Here are the **Mermaid code blocks** for each flowchart:

## **1. Detailed Workflow Diagram**

```mermaid
flowchart TD
    A[Start Research Task] --> B[Checkout main branch]
    B --> C[Pull latest changes]
    C --> D[Create short-lived branch<br>git checkout -b experiment-1]
    
    D --> E[Work on task<br>Make code changes]
    E --> F{Ready to commit?}
    F -->|Yes| G[Commit with clear message<br>git commit -m 'Add dropout parameter']
    G --> H[Push branch<br>git push origin experiment-1]
    F -->|No| E
    
    H --> I[Create Pull Request<br>Request review]
    I --> J{PR Approved?}
    J -->|Yes| K[Merge into main branch]
    J -->|No| L[Address feedback<br>Update branch]
    L --> H
    
    K --> M[Delete branch<br>git branch -d experiment-1]
    M --> N{Reached milestone?}
    N -->|Yes| O[Create tag<br>git tag -a v1.0-paper]
    O --> P[Push tags<br>git push --tags]
    N -->|No| A
    
    P --> Q[Documentation<br>Update README/CHANGELOG]
    Q --> A
    
    style A fill:#e1f5fe
    style D fill:#f3e5f5
    style G fill:#e8f5e8
    style K fill:#fff3e0
    style O fill:#ffebee
```

## **2. Simplified Overview Diagram**

```mermaid
flowchart LR
    A[main<br>Always stable] --> B[Create short-lived branch]
    B --> C[Work & Commit]
    C --> D[Push & PR]
    D --> E[Review & Merge]
    E --> A
    E --> F[Tag milestones]
    
    style A fill:#e1f5fe,stroke:#333,stroke-width:2px
    style F fill:#fff3e0
```

## **3. Alternative: Timeline View**


```mermaid
gantt
    title Research Git Workflow Timeline
    dateFormat HH:mm
    axisFormat %H:%M
    
    section Day 1
    Checkout main & sync :a1, 09:00, 15m
    Create branch :a2, after a1, 5m
    Implement changes :a3, after a2, 2h
    Commit changes :a4, after a3, 10m
    Push & create PR :a5, after a4, 5m
    
    section Day 1-2
    Review & feedback :b1, after a5, 4h
    Address feedback :b2, after b1, 1h
    
    section Day 2
    Merge to main :c1, after b2, 5m
    Delete branch :c2, after c1, 2m
    
    section Milestone
    Tag version :d1, after c2, 3m
```

---

## **4. Collaboration-Focused Diagram**

```mermaid
flowchart TB
    subgraph "Researcher A"
        A1[Create branch A] --> A2[Work on experiment]
        A2 --> A3[Commit & push]
        A3 --> A4[Create PR]
    end
    
    subgraph "Researcher B"
        B1[Create branch B] --> B2[Work on analysis]
        B2 --> B3[Commit & push]
        B3 --> B4[Create PR]
    end
    
    A4 --> C[Review each other's PRs]
    B4 --> C
    
    C --> D{Merge approved?}
    D -->|Yes| E[Merge to main]
    D -->|No| F[Request changes]
    F --> A2
    F --> B2
    
    E --> G[main: Integrated work]
    G --> H[Tag: v1.0-experiment]
    
    style G fill:#e1f5fe
    style H fill:#fff3e0
```

---

## **Usage Instructions:**

1. **Copy** any Mermaid code block
2. **Paste** into any Markdown file that supports Mermaid (GitHub, GitLab, Obsidian, VS Code with Mermaid extension)
3. **Render** automatically or use a Mermaid viewer
4. **Customize** colors, text, or structure as needed

Each diagram serves a different purpose:
- **Detailed**: Step-by-step tutorial
- **Simplified**: Quick reference
- **Timeline**: Understanding time expectations
- **Collaboration**: Team workflow visualization

Would you like me to create any other variations or explain specific parts of the Mermaid syntax?