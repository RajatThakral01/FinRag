# 📈 FinRAG Terminal Frontend

A modern React + TypeScript + Vite web client built for the **Financial Intelligence RAG System**.

---

## 🎨 Visual Identity & Design Tokens

Designed as an institutional research terminal with a strict **monochrome-plus-single-accent** palette:

- **Page Background (`--bg-app`):** `#111214` (Dark Graphite)
- **Surface Background (`--bg-surface`):** `#1A1C20` (Dark Card Surface)
- **Default Border (`--border-default`):** `#2A2B2F`
- **Strong Border (`--border-strong`):** `#33353B`
- **Primary Text (`--text-primary`):** `#EDEDEE` (High Contrast White)
- **Metadata Text (`--text-secondary`):** `#8B8D94` (Cool Slate)
- **Sole Accent (`--accent-sole`):** `#8CA0C7` (Muted Blue-Gray)

### Typography
- **Serif Headers:** `Newsreader`
- **Body Prose:** `Inter`
- **Financial Figures & Tags:** `JetBrains Mono` with `font-variant-numeric: tabular-nums`

---

## 🧩 Key Components

### 1. `SessionSidebar`
- Sidebar listing past research sessions loaded via `GET /sessions`.
- Creates new research sessions (`POST /sessions`).
- Highlights active session with accent indicator.

### 2. `ChatContainer` & `MessageBubble`
- Displays turn history for active sessions.
- Renders `ResolvedQuestionBadge` when multi-turn questions are rewritten by context resolver.
- Renders `WarningBanner` for low-confidence retrievals.
- Renders `HonestFailureCard` for unverified answers.

### 3. `SourcesPanel`
- Accordion component listing citation cards from `chunk_sources`.
- Shows table names, `[TABLE]` / `[PROSE]` badges, ticker, and monospace `chunk_id`.
- Click-to-expand **Audited Chunk Parameters** drawer.
- *"Show full chunk"* action button triggering slide-in preview.

### 4. `ChunkPreviewPanel`
- Non-modal slide-in preview panel (~38% viewport width desktop, full overlay mobile).
- Fetches full document text and metadata via `GET /chunks/{chunk_id}`.
- Renders chunk body text in readable `Inter` font.
- Supports keyboard close (`Escape` key).

---

## 🛠️ Development & Build Scripts

```bash
# Install dependencies
npm install

# Start Vite dev server on http://localhost:3000
npm run dev

# TypeScript check & production build
npm run build

# Preview production build
npm run preview
```
