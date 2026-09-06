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

## Run the assistant

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
```

Application code lives in `src/lng_ticket_assistant/`. The notebooks under
`dev/` remain the exploratory PDF preprocessing and embedding workflow.
