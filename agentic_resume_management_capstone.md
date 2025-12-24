# Agentic Resume Management & Skill Search System

## Capstone Project Documentation (Minimal Implementation)

---

## 1. Project Overview

The **Agentic Resume Management & Skill Search System** is a capstone project that demonstrates the use of **AI agents** to automate resume ingestion, skill extraction, and intelligent candidate search. The system uses Large Language Models (LLMs) to reason over unstructured resume data and respond to skill-based queries.

The goal of this project is to showcase **agentic workflows, reasoning, and collaboration** rather than building a full-scale Applicant Tracking System (ATS).

---

## 2. Problem Statement

Organizations receive a large number of resumes in unstructured formats (PDF, DOC, text). Manually extracting skills and searching for suitable candidates is time-consuming and inefficient.

This project solves the problem by:
- Automatically extracting structured data from resumes
- Normalizing and storing skills
- Enabling intelligent, skill-based resume search using AI agents

---

## 3. Why Agentic Architecture?

Instead of a monolithic application, this system is designed using **AI agents**, where each agent:
- Has a single responsibility
- Makes decisions based on context
- Collaborates with other agents to complete tasks

This reflects real-world **agentic AI systems** used in enterprise AI platforms.

---

## 4. High-Level Architecture

```
User
  |
  v
Search Agent  <------------------- Resume Upload
  |                                  |
  v                                  v
Skill Indexing Agent        Resume Ingestion Agent
  |                                  |
  v                                  v
Structured Resume Store (JSON / SQLite)
```

---

## 5. Core Agents

### 5.1 Resume Ingestion Agent

**Responsibility:**
- Accept resumes in PDF, DOC, or text format
- Extract structured information using LLM prompts

**Extracted Fields:**
- Candidate Name
- Skills
- Years of Experience (optional)

**Sample Output:**
```json
{
  "name": "John Doe",
  "skills": ["Angular", "TypeScript", "REST APIs"],
  "experience_years": 5
}
```

---

### 5.2 Skill Indexing Agent

**Responsibility:**
- Normalize extracted skills (e.g., Angular → Angular 2+)
- Store structured resume data

**Storage Options (Minimal):**
- JSON file
- SQLite database
- In-memory data structure

This agent ensures consistency in skill representation.

---

### 5.3 Search Agent

**Responsibility:**
- Accept natural language queries from users
- Identify required skills
- Perform reasoning-based matching
- Rank resumes based on relevance

**Example Query:**
> "Find candidates with Angular and microservices"

---

## 6. Agentic Workflow

1. User uploads a resume
2. Resume Ingestion Agent extracts structured data
3. Skill Indexing Agent normalizes and stores skills
4. User submits a search query
5. Search Agent reasons over stored data
6. Ranked results are returned to the user

This workflow highlights **decision-making and collaboration** among agents.

---

## 7. Technology Stack

### Backend
- Python + FastAPI (recommended)
- OR NestJS (alternative)

### AI / LLM
- OpenAI / Azure OpenAI / Gemini
- Prompt-based extraction (no model training)

### Agent Framework (Optional)
- LangChain Agents
- OR custom function-based agents

### Frontend (Optional)
- Angular (resume upload & search UI)
- Postman demo acceptable for capstone

---

## 8. Key Capstone Highlights

- Demonstrates agent-based architecture
- Uses LLM reasoning instead of rule-based parsing
- Clear separation of concerns across agents
- Easily extensible to job–candidate matching

---

## 9. Scope Control (What Is Intentionally Excluded)

- Authentication & authorization
- Full ATS features
- Advanced embeddings & vector databases
- Multi-tenant support

The focus is on **conceptual clarity and agentic reasoning**.

---

## 10. Future Enhancements (Optional)

- Resume-to-job matching agent
- Vector database for semantic search
- Autonomous resume ranking improvements
- Skill gap analysis

---

## 11. Resume Description (For Job Applications)

**Capstone Project: Agentic Resume Management System**  
Designed a multi-agent AI system for resume ingestion, skill extraction, and intelligent candidate search using LLM-powered agents and FastAPI.

---

## 12. Conclusion

This project demonstrates how **agentic AI systems** can solve real-world problems using minimal yet powerful implementations. The architecture emphasizes reasoning, collaboration, and extensibility, making it an ideal capstone project.

---

**End of Document**