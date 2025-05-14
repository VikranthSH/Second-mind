SecondMind - AI-Driven Idea Generator
Project Introduction
SecondMind is an advanced AI-driven autonomous agent framework designed to enhance research, knowledge synthesis, and decision-making. This project integrates multiple intelligent agents, each with a specialized function, working collaboratively to analyze, refine, and evaluate generated hypotheses.
The system leverages Large Language Models (LLMs) like Gemini, along with Natural Language Processing (NLP), Machine Learning (ML), and web scraping techniques to collect and process information from diverse sources such as research papers, patents, and news articles.
With a modular and scalable architecture, SecondMind enables real-time insights by:
•	Generating hypotheses based on retrieved knowledge.
•	Reflecting on and ranking outputs for coherence and credibility.
•	Evolving ideas through iterative improvements.
•	Providing historical context using previous interactions.
•	Automating web research using APIs and scrapers.
The system can be extended to various domains, including academic research, market analysis, and innovation tracking. Its agentic framework allows it to adapt and learn over time, ensuring high-quality outputs with minimal human intervention.
 Features
•	AI-generated text based on prompts
•	Reflection and evaluation to improve outputs
•	FastAPI-powered REST API
•	Modular agent-based architecture
•	Logging and storage support

Project Goals
The primary goal of SecondMind is to create an autonomous, intelligent research and decision-support system that can:
✅ Automate Research: Collect and process data from various sources, including academic papers, patents, and news articles.
✅ Generate Insights: Utilize LLMs to synthesize and summarize information into meaningful insights.
✅ Validate and Refine Information: Use agent-based ranking, reflection, and evolution mechanisms to improve content accuracy and coherence.
✅ Enhance Decision-Making: Assist users in making informed decisions by presenting well-structured, context-aware recommendations.
✅ Ensure Scalability & Adaptability: Implement a modular design that allows integration with new tools, APIs, and custom workflows.
 Project Structure
second_mind/
├── main.py                     # Entry point
├── config.py                   # Configuration and settings
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── core/
│   ├── _init_.py
│   ├── supervisor.py           # Supervisor agent implementation
│   ├── storage.py              # Memory storage system
│   └── context_manager.py      # Context protocol server
├── agents/
│   ├── _init_.py
│   ├── base_agent.py           # Abstract base agent class
│   ├── generation_agent.py     # Creates initial hypotheses
│   ├── reflection_agent.py     # Checks coherence
│   ├── ranking_agent.py        # Scores outputs
│   ├── evolution_agent.py      # Refines ideas
│   ├── proximity_agent.py      # Links to past interactions
│   └── meta_review_agent.py    # Evaluates process
├── web/
│   ├── _init_.py
│   ├── google_search_wrapper.py# Google Search API integration
│   ├── scraper.py              # Web scraping utilities
│   ├── academic_scraper.py     # Academic repository scraper
│   ├── patent_scraper.py       # Patent database scraper
│   └── news_scraper.py         # Tech news scraper
└── utils/
    ├── _init_.py
    ├── logger.py               # Logging utilities



# **🧠 Second Mind**  
**An AI-driven system that mimics human learning through specialized agents.**  

## **🛠 Installation & Setup**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/SkandaGanesha1/Second_Mind_Project.git
```

### **2️⃣ Install Dependencies**  
Ensure you have Python 3.9+ installed, then run:  
```bash
pip install -r requirements.txt
```

### **3️⃣ Set Up API Keys**  
Create a `.env` file in the project root and add:  
```plaintext
GEMINI_API_KEY=your-gemini-api-key
```

---

## **🚀 Usage**  

### **🎯 Running the Core System**  
To generate AI-powered ideas, execute:  
```bash
python main.py
```

### **🌍 Running the API**  
Start the API server using FastAPI:  
```bash
uvicorn web.api:app --reload
```
Test the API using:  
```bash
curl -X POST "http://127.0.0.1:8000/generate" -d "prompt=AI use case"
```

---

## **📡 API Endpoints**  

| Method | Endpoint  | Description                  |
|--------|----------|------------------------------|
| **POST** | `/generate` | Generates AI-based responses |

---

## **🛠 Technologies Used**  
- **Python 3.9+**  
- **Google Gemini AI** (for text generation)  
- **FastAPI** (for API development)  
- **Logging & Storage** (JSON-based)  

---

## **🚀 Future Enhancements**  
✅ **Add more specialized agents** (Ranking, Evolution, etc.)  
✅ **Enhance storage** with SQLite or Redis  
✅ **Deploy API** using Docker & Cloud  
✅ **Integrate a web-based UI**  

---

## **👥 Contributors**  
👨‍💻 **Vikranth S H** - [GitHub Profile]([https://github.com/VikranthSH))  
📧 **Contact:** vikranthsh83@gmail.com  

![Screenshot 2025-03-20 150335](https://github.com/user-attachments/assets/10ed83b2-e5b5-4dc2-9d84-d7cb69a4014f)


