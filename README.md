# cpp-vector-search-rag

A vector search engine written in C++ with three interchangeable index types (HNSW, KD-Tree, brute force), a REST API, a browser UI, and a local RAG pipeline (Ollama embeddings + LLM answers over your own documents).

The goal is to make the internals of vector databases like Pinecone, Weaviate, and Chroma visible: you can run all three algorithms on the same data, compare latency, and see the semantic space plotted in 2D.

---

## Features

| Area | What's included |
|---|---|
| Index types | HNSW (approximate), KD-Tree (exact), brute force (exact baseline) |
| Distance metrics | Cosine, Euclidean, Manhattan |
| Demo dataset | 20 hand-made 16D vectors in 4 categories (CS, Math, Food, Sports) |
| Visualization | 2D PCA scatter plot of the vector space |
| Document ingestion | Text is chunked (250 words, overlapping), embedded with `nomic-embed-text` (768D), indexed in HNSW |
| RAG | Question → embedding → top-k chunks from HNSW → `llama3.2` answers from that context |
| API | REST endpoints for insert, delete, search, benchmark, index stats |

## How the RAG pipeline works

```
Document text ──► chunk ──► Ollama embed (768D) ──► HNSW index
                                                        │
Question ──► Ollama embed ──► HNSW top-k search ◄───────┘
                                    │
                         retrieved chunks as context
                                    │
                              Ollama LLM ──► answer
```

## Algorithms

| Algorithm | Search complexity | Exact? | Notes |
|---|---|---|---|
| Brute force | O(N·d) | Yes | Baseline for correctness and speed comparison |
| KD-Tree | ~O(log N) in low dimensions | Yes | Degrades toward brute force as dimensions grow |
| HNSW | ~O(log N) | No (approximate) | Multilayer small-world graph; used for the 768D document index |

**HNSW in brief.** Each node is assigned a random max layer. Upper layers are sparse and hold long-range links; layer 0 holds every node. Insert and search both descend greedily from the top layer, then run a beam search (`efConstruction` on insert, `ef` on query) at the lower layers. Each new node is linked bidirectionally to its `M` nearest neighbours.

**Why KD-Tree struggles at high dimensions.** Its pruning relies on axis-aligned distance bounds. At hundreds of dimensions almost no subtree can be pruned, so it behaves like brute force. Graph-based indexes like HNSW don't depend on that bound.

## Project structure

```
.
├── main.cpp      # BruteForce, KDTree, HNSW, VectorDB, DocumentDB, OllamaClient, REST routes
├── httplib.h     # cpp-httplib (single-header HTTP server, third-party)
├── index.html    # Frontend: PCA plot, document upload, chat UI, benchmark
└── README.md
```

## Requirements

- A C++17 compiler (g++ via MSYS2 on Windows, or g++/clang++ on Linux/macOS)
- [Ollama](https://ollama.com) with two models pulled:
  ```bash
  ollama pull nomic-embed-text   # embeddings, ~274 MB
  ollama pull llama3.2           # generation, ~2 GB
  ```
- Around 8 GB RAM recommended for running the models

## Build and run

**Windows (MSYS2 UCRT64 toolchain on PATH):**
```powershell
g++ -std=c++17 -O2 main.cpp -o db -lws2_32
./db
```

**Linux / macOS:**
```bash
g++ -std=c++17 -O2 -pthread main.cpp -o db
./db
```

Make sure Ollama is running (`ollama serve`, or the tray app on Windows), then open <http://localhost:8080>.

Expected startup output:
```
=== VectorDB Engine ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
```

## Using the UI

1. **Search tab:** query the demo vectors, pick an algorithm and metric, or run *Compare All* to see latency for all three side by side.
2. **Documents tab:** paste text and click *Embed & Insert*. It's chunked, embedded, and added to the HNSW document index.
3. **Ask tab:** ask a question about your documents. The top-3 chunks are retrieved and passed to the LLM; the chunks used are shown so you can verify the answer.

## REST API

Base URL: `http://localhost:8080`

**Demo vectors (16D)**

| Method | Endpoint | Description |
|---|---|---|
| GET | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | k-NN search |
| POST | `/insert` | Insert a vector |
| DELETE | `/delete/:id` | Delete by ID |
| GET | `/items` | List all vectors |
| GET | `/benchmark?v=...&k=5&metric=cosine` | Run all 3 algorithms and compare |
| GET | `/hnsw-info` | Layer structure and stats |
| GET | `/stats` | Database statistics |

**Documents and RAG (768D)**

| Method | Endpoint | Body | Description |
|---|---|---|---|
| POST | `/doc/insert` | `{"title":"...","text":"..."}` | Chunk, embed, store |
| GET | `/doc/list` | | List stored chunks |
| DELETE | `/doc/delete/:id` | | Delete a chunk |
| POST | `/doc/ask` | `{"question":"...","k":3}` | Retrieve and generate |
| GET | `/status` | | Ollama status and model names |

**Examples**
```bash
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"

curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` | Start Ollama with `ollama serve` |
| First embedding is very slow | Ollama is loading the model; wait a minute or two |
| `g++: command not found` | Add the compiler's `bin` directory to PATH |
| `undefined reference to WSA...` (Windows) | Add `-lws2_32` to the compile command |
| Port 8080 in use | Stop the other process or change the port in `main.cpp` |
| LLM answers are slow | Use a smaller model: `ollama pull llama3.2:1b` and change `genModel` in `main.cpp` |

## Acknowledgements

- [cpp-httplib](https://github.com/yhirose/cpp-httplib) for the HTTP server
- Ollama for local embedding and generation
- HNSW: Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs* (2016)
- <!-- If this project builds on someone else's repository, credit it here. -->

## License

MIT