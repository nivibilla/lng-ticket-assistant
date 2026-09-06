# lng-ticket-assistant

A Gemini-backed Deep Agents assistant for answering L&G IT support questions
from a local LanceDB knowledge base.

The assistant uses hybrid retrieval: Jina retrieval embeddings provide semantic
search, LanceDB full-text search preserves exact terms such as error codes, and
LanceDB fuses both result sets. Responses are grounded in retrieved passages
and include source-page citations. The assistant can recommend ticket
submission but does not create or update tickets.

## Requirements

- Python 3.13 or newer
- A Google Generative AI API key
- Internet access the first time the Jina embedding model is downloaded
- The native Tesseract executable when processing scanned PDFs that do not
  contain enough extractable text

Install the locked development environment:

```sh
uv sync
```

Alternatively, install the package from the project root:

```sh
python -m pip install .
```

## Configuration

Create a local environment file and add your Gemini API key:

```sh
cp .env.example .env
```

```dotenv
GOOGLE_API_KEY=your-api-key
```

The CLI loads `.env` from the current working directory. The real `.env` is
ignored by Git. Existing shell or CI environment variables take precedence
over values in the file.

Optional `.env` values:

```dotenv
# Defaults to gemini-3.7-flash
GOOGLE_MODEL_ID=gemini-3.7-flash

# Defaults to ./kb_db when running from the repository
KB_DB_PATH=/absolute/path/to/kb_db

# Defaults to jinaai/jina-embeddings-v5-text-small
EMBEDDING_MODEL_ID=jinaai/jina-embeddings-v5-text-small
```

## Build the knowledge base

Run the two-stage pipeline after changing the source PDF. First extract and
chunk the PDF into `chunks_df.parquet`:

```sh
uv run lng-ticket-preprocess --pdf "KB Articles.pdf"
```

The preprocessing command uses native PDF text where available and falls back
to Tesseract OCR for pages with fewer than 50 extracted characters. Install
Tesseract separately if OCR is required; `pytesseract` does not provide the
native executable.

Then embed the Parquet chunks with Jina's asymmetric retrieval document prompt
and publish the complete table and full-text index to LanceDB:

```sh
uv run lng-ticket-rebuild-embeddings
```

Use custom pipeline paths when needed:

```sh
uv run lng-ticket-preprocess \
  --pdf /absolute/path/to/articles.pdf \
  --output /absolute/path/to/chunks.parquet

uv run lng-ticket-rebuild-embeddings \
  --chunks-path /absolute/path/to/chunks.parquet \
  --database-path /absolute/path/to/kb_db
```

Run `uv run lng-ticket-preprocess --help` to tune chunk size, overlap, OCR
threshold, or render scale. Parquet output is replaced atomically, and
preprocessing does not modify the live database. The embedding command
validates the six chunk columns and computes every replacement vector before
publishing, preserving source and page metadata. Its first run downloads code
and model weights from the configured Hugging Face model because Jina requires
`trust_remote_code=True`.

## Run the Teams-style browser demo

The standalone web app recreates a minimal Teams desktop chat and connects to
the real assistant. It includes streamed replies, thinking/search pulses, and
clickable citations with the exact source passage and a PDF viewer. No Microsoft
account, Teams installation, or Microsoft service is required.

Install Node.js 22.13 or newer alongside the Python requirements above. From
the repository root, install and build the browser app once:

```sh
uv sync
npm --prefix frontend ci
npm --prefix frontend run build
```

Start the demo:

```sh
uv run lng-ticket-web
```

Open [the local demo](http://127.0.0.1:8000). The server loads `.env` just like
the CLI; `GOOGLE_API_KEY` and the local knowledge base are still required for
answers. Before accepting connections, the server initializes the agent, loads
Jina, runs a real hybrid knowledge-base search, and completes a short streamed
Gemini greeting. Wait for **Assistant ready** and Uvicorn's listening URL before
opening the demo. The terminal reports each startup stage and its elapsed time.

The first startup can take longer while model weights download; local loading
has no startup deadline. The Gemini smoke turn has a 60-second deadline and uses
a small live API request on every restart. Missing configuration, loading or
retrieval errors, empty search results, and a failed or timed-out Gemini check
stop startup with an unsuccessful exit. Check the failed stage in the terminal:
verify `GOOGLE_API_KEY`/`GOOGLE_MODEL_ID` for Google configuration, the local
database for retrieval, and internet access for model downloads or Gemini.

Chats reuse the warmed model and retriever in the same server process, so the
first support question no longer pays the model-loading cost. Normal retrieval
and Gemini response time still apply. Smoke-test messages never appear in chat
history. Frontend assets and the PDF viewer are bundled locally; only the existing
AI/model-download services require internet access. CLI startup is unchanged.

The source PDF defaults to `KB Articles.pdf` in this checkout. When using a
different knowledge base, put its PDFs in a source directory with filenames that
match the stored `source` metadata:

```sh
uv run lng-ticket-web --port 8000 \
  --database-path /absolute/path/to/kb_db \
  --source-dir /absolute/path/to/source-pdfs
```

Alternatively, set `KB_SOURCE_DIR` in `.env`. If a PDF is unavailable, citations
still show the retrieved passage. The current PDF is scanned, so the viewer
navigates to the cited page rather than highlighting text. Only known source PDFs
referenced in the current browser session are served; arbitrary filesystem paths
are never exposed.

There is one conversation per browser session. Completed messages survive page
refreshes while the server runs; server restarts clear history. Stop interrupts
the current reply, failed turns can be retried, and Reset starts a fresh chat.
Refreshing during a reply interrupts that turn and restores a retryable message.
Other Teams navigation and composer icons are visual context for the demo.
The server binds to `127.0.0.1` by default and has no login or durable storage;
it is intended for local demonstrations, not a shared production deployment.

For frontend development, keep `uv run lng-ticket-web` running and start Vite in
a second terminal:

```sh
npm --prefix frontend run dev
```

Open the Vite URL printed in the terminal. It proxies `/api` to port 8000 and
reloads frontend changes automatically. Restart the Python server after backend
changes. For the single-server demo, rebuild the frontend after UI changes.

## Run the command-line assistant

```sh
uv run lng-ticket-assistant
# Or:
uv run python -m lng_ticket_assistant
```

Use `quit`, `exit`, Ctrl-C, or Ctrl-D to leave the conversation. Specify a
different database with:

```sh
uv run lng-ticket-assistant --database-path /absolute/path/to/kb_db
```

## Tests

The tests use fake embedding and chat models, so they do not call Google or
download Jina:

```sh
uv run python -m unittest discover -s tests -v
npm --prefix frontend test
npm --prefix frontend run build
```

Application code lives in `src/lng_ticket_assistant/`. The notebooks under
`dev/` remain the exploratory PDF preprocessing and embedding workflow.

The Python tests also exercise startup readiness, reuse of the warmed models,
real hybrid retrieval with fake embeddings, and failed or timed-out smoke checks.
The Python web tests exercise streamed responses, internal-history filtering,
grounded citations, isolated sessions, retry, reset/cancellation, timeouts, and PDF
access. Frontend tests cover stream framing, partial citations, Markdown safety,
the chat lifecycle, and the PDF panel. They use fake models and PDF rendering;
they do not call Google or download Jina.

The browser uses `GET /api/conversation`, `POST /api/chat` (SSE over fetch),
`DELETE /api/conversation`, and `GET /api/sources/{source_id}`. Chat events are
`accepted`, `status`, `message_start`, `delta`, `complete`, and `error`; completed
messages include validated citation records. The web-specific prompt emits
`[[cite:chunk-N]]` markers. The CLI retains its readable filename/page citations.
