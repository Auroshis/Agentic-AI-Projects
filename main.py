# main.py
import asyncio
from agent_flow import build_graph

async def main():
    graph = build_graph()
    init_state = {
        "job_description": """We’re looking for a Senior Python Engineer to join our Recommendations team. Your mission? Help millions of pet lovers find the perfect products—without building yet another monolith that cries in staging. You'll work with product owners, architects, and developers who’ve sworn off spaghetti code and embrace actual architecture (yes, that exists).


What you’ll be doing (aka your excuse to brag):

Build modern, scalable backend features for one of Europe’s largest pet e-com platforms.
Code, test, ship, repeat (with CI/CD, Docker, AWS, and a sprinkle of Terraform magic).
Work with machine learning models, microservices, and data-heavy systems that actually get used.
Join tech discussions that matter (no meetings that could’ve been emails).
Collaborate like a pro across teams in a friendly, Agile environment.


What you need to bring:

5+ years of Python experience and clean code habits your mom would be proud of
Flask / FastAPI – you know your APIs and how to keep them lean
Distributed systems experience + bonus points for ML model handling
SQL, NoSQL, React (yes, frontend too—you’ll survive)
Strong understanding of design patterns (and knowing when not to use them)
DevOps-curious or better


Bonus points if you have:

Microservices experience
Worked in e-commerce
TDD tendencies (or at least no fear of tests)""",
        "resume_text": """
### 🎓 **Education**

* **B.Tech, IIIT Bhubaneswar** — 2017–2021

  * CGPA: **8.48**

---

### 💼 **Experience**

#### **Senior Automation Engineer — WWT (Pfizer)** *(Jan 2025 – Present, Bangalore)*

* Automated **firewall rule creation** with *Policy-as-Code* maintained in GitHub, cutting operational tickets and costs by **50%**.
* Designed **AI Agent-driven workflows** for validation, approval routing, and SLA notifications — reduced turnaround by **4 days**.
* Built **automated firewall management system**, replacing a manual 5-stage workflow for cybersecurity.
* Led **ServiceNow migration** for 3 major use cases (10K+ tickets/month) with **zero downtime**.

#### **SDE 2 — M2P Fintech (Connect)** *(Feb 2024 – Jan 2025, New Delhi)*

* Designed **high-throughput workflow systems** (2000 TPS) using *CQRS, caching, and modular services* — improved 95th percentile latency from **650 ms → 380 ms** at 30M monthly transactions.
* Built **secure secrets manager** with *AES-256* and *PKCS#7 signing* for PCI-DSS compliance.
* Developed **Kafka-based streaming** system with ack/commit handling to eliminate order loss.
* Created **multilingual, multi-tenant GenAI chatbot** using *LangChain, Llama2,* and *Azure OpenAI*.
* Handled production systems, ingress configs, and optimized **DevOps workflows**.

#### **Software Engineer II/I — Dell Technologies** *(Aug 2021 – Jan 2024, Bangalore)*

* Improved backend reliability (43% → 73%) by fixing silent failures and data refresh issues in telemetry.
* Reduced CPU usage (100% → 20%) and storage costs (↓ 40%) via database indexing, ETL improvements, and *S3 lifecycle policies*.
* Implemented **JWT-based authentication** and **RBAC** using *Kong API Gateway*.
* Recognized with **Dell Inspire Award X10**.

---

### ⚙️ **Technical Skills**

**Core Skills:**

* Backend Development, System Design, Data Structures & Algorithms

**Languages:**

* Python, Node.js, SQL

**Frameworks:**

* FastAPI, Django, NestJS, LangChain

**Databases & Messaging:**

* PostgreSQL, MongoDB, Redis, Kafka, RabbitMQ, Kong

**DevOps & Infrastructure:**

* Docker, Kubernetes, ArgoCD, Ansible, GitLab/GitHub CI/CD

**Cloud & AI Services:**

* AWS (EC2, S3, Lambda), GCP (Cloud Run, Registry), Azure OpenAI

---

### 🧠 **Projects**

**OneBot**

* Tech Stack: *Llama2, LangChain, ChromaDB, Azure OpenAI, Streamlit*
* Built a **multi-tenant, multilingual GenAI assistant** for knowledge retrieval across multiple data sources.
* Achieved major **query response time reduction** and **time savings** for users.

---

### 🏆 **Recognitions**

* **Dell Inspire Award X10**
* **AWS Certified Cloud Practitioner**


""",
        "github_user": "Auroshis"
    }
    result_state = await graph.ainvoke(init_state)
    print("Missing topics:", result_state["missing_topics"])
    print("Learning plan:", result_state["learning_plan"])
    print("Tuned resume:", result_state["tuned_resume"])

if __name__ == "__main__":
    asyncio.run(main())
