import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft20Regular,
  ArrowRight20Regular,
  ArrowUpRight20Regular,
  Dismiss20Regular,
  DocumentPdf24Regular,
  Subtract20Regular,
  Add20Regular,
  DocumentSearch24Regular,
} from "@fluentui/react-icons";
import type { PDFDocumentProxy } from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import type { Citation } from "../types";
import { pageLabel } from "../lib/citations";

export default function CitationPanel({
  citation,
  onClose,
}: {
  citation: Citation;
  onClose: () => void;
}) {
  const close = useRef<HTMLButtonElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const viewport = useRef<HTMLDivElement>(null);
  const [document, setDocument] = useState<PDFDocumentProxy | null>(null);
  const [page, setPage] = useState(citation.page_start ?? 1);
  const [zoom, setZoom] = useState(1);
  const [width, setWidth] = useState(400);
  const [error, setError] = useState("");
  const [rendering, setRendering] = useState(true);

  useEffect(() => {
    close.current?.focus();
  }, []);
  useEffect(() => {
    setPage(citation.page_start ?? 1);
    setZoom(1);
  }, [citation.evidence_id, citation.page_start]);
  useEffect(() => {
    const element = viewport.current;
    if (!element) return;
    const observer = new ResizeObserver((entries) =>
      setWidth(entries[0].contentRect.width),
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let disposed = false;
    let destroy: (() => void) | undefined;
    setDocument(null);
    setError("");
    setRendering(true);
    if (!citation.pdf_url) return;
    void import("pdfjs-dist")
      .then(async (pdfjs) => {
        if (disposed) return;
        pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;
        const loading = pdfjs.getDocument({
          url: citation.pdf_url!,
          withCredentials: true,
        });
        destroy = () => {
          void loading.destroy();
        };
        const pdf = await loading.promise;
        if (!disposed) setDocument(pdf);
      })
      .catch(() => {
        if (!disposed)
          setError(
            "The original PDF couldn’t be loaded. You can still read the source passage above.",
          );
      });
    return () => {
      disposed = true;
      destroy?.();
    };
  }, [citation.pdf_url]);

  const currentPage = document
    ? Math.max(1, Math.min(page, document.numPages))
    : page;
  useEffect(() => {
    if (!document || !canvas.current) return;
    let disposed = false;
    let cancel: (() => void) | undefined;
    setRendering(true);
    void document
      .getPage(currentPage)
      .then(async (pdfPage) => {
        if (disposed || !canvas.current) return;
        const base = pdfPage.getViewport({ scale: 1 });
        const scale = (Math.max(150, width - 32) / base.width) * zoom;
        const view = pdfPage.getViewport({ scale });
        const dpr = window.devicePixelRatio || 1;
        const element = canvas.current;
        element.width = Math.floor(view.width * dpr);
        element.height = Math.floor(view.height * dpr);
        element.style.width = `${view.width}px`;
        element.style.height = `${view.height}px`;
        const task = pdfPage.render({
          canvas: element,
          viewport: view,
          transform: [dpr, 0, 0, dpr, 0, 0],
        });
        cancel = () => task.cancel();
        await task.promise;
        if (!disposed) setRendering(false);
      })
      .catch((error) => {
        if (!disposed && error.name !== "RenderingCancelledException") {
          setError(
            "This page couldn’t be displayed. Try opening the original PDF.",
          );
        }
      });
    return () => {
      disposed = true;
      cancel?.();
    };
  }, [document, currentPage, width, zoom]);

  return (
    <aside
      className="citation-panel"
      aria-label="Source details"
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.stopPropagation();
          onClose();
        }
        // The small-screen panel is modal; keep keyboard focus inside it.
        if (
          event.key === "Tab" &&
          window.matchMedia("(max-width: 1100px)").matches
        ) {
          const items = Array.from(
            event.currentTarget.querySelectorAll<HTMLElement>(
              "button:not(:disabled), a[href]",
            ),
          );
          const first = items[0],
            last = items[items.length - 1];
          if (event.shiftKey && window.document.activeElement === first) {
            event.preventDefault();
            last?.focus();
          }
          if (!event.shiftKey && window.document.activeElement === last) {
            event.preventDefault();
            first?.focus();
          }
        }
      }}
    >
      <div className="panel-heading">
        <h2>Source details</h2>
        <button
          ref={close}
          className="icon-button"
          onClick={onClose}
          aria-label="Close source details"
        >
          <Dismiss20Regular />
        </button>
      </div>
      <div className="source-summary">
        <div className="source-file">
          <span className="pdf-file-icon">
            <DocumentPdf24Regular />
          </span>
          <div>
            <h3>{citation.source}</h3>
            <span>
              {pageLabel(citation)} <span className="dot-divider">·</span>{" "}
              Knowledge base
            </span>
          </div>
          <span className="source-number">{citation.number}</span>
        </div>
        <div className="passage-label">
          <DocumentSearch24Regular />
          <h4>Referenced passage</h4>
        </div>
        <blockquote>{citation.content}</blockquote>
      </div>
      <div className="pdf-heading">
        <h4>Original document</h4>
        {citation.pdf_url && (
          <a
            href={`${citation.pdf_url}#page=${citation.page_start ?? 1}`}
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Open original PDF in new tab"
          >
            <ArrowUpRight20Regular />
          </a>
        )}
      </div>
      {citation.pdf_url && (
        <div className="pdf-toolbar">
          <button
            className="icon-button"
            disabled={!document || currentPage <= 1}
            aria-label="Previous PDF page"
            onClick={() => setPage(currentPage - 1)}
          >
            <ArrowLeft20Regular />
          </button>
          <span className="page-count">
            {document
              ? `Page ${currentPage} of ${document.numPages}`
              : "Loading PDF…"}
          </span>
          <button
            className="icon-button"
            disabled={!document || currentPage >= document.numPages}
            aria-label="Next PDF page"
            onClick={() => setPage(currentPage + 1)}
          >
            <ArrowRight20Regular />
          </button>
          <span className="toolbar-separator" />
          <button
            className="icon-button"
            aria-label="Zoom out"
            disabled={zoom <= 0.75 || !document}
            onClick={() => setZoom((value) => Math.max(0.75, value - 0.25))}
          >
            <Subtract20Regular />
          </button>
          <button
            className="zoom-label"
            aria-label="Fit PDF to width"
            onClick={() => setZoom(1)}
          >
            {Math.round(zoom * 100)}%
          </button>
          <button
            className="icon-button"
            aria-label="Zoom in"
            disabled={zoom >= 2 || !document}
            onClick={() => setZoom((value) => Math.min(2, value + 0.25))}
          >
            <Add20Regular />
          </button>
        </div>
      )}
      <div
        className="pdf-viewport"
        ref={viewport}
        aria-busy={!!citation.pdf_url && !error && rendering}
      >
        {!citation.pdf_url || error ? (
          <div className="pdf-unavailable">
            <DocumentPdf24Regular />
            <p>
              {error ||
                "The original PDF is unavailable. The retrieved passage is shown above."}
            </p>
          </div>
        ) : (
          <>
            {rendering && (
              <div className="pdf-loading" role="status">
                Loading page…
              </div>
            )}
            <canvas
              ref={canvas}
              aria-label={`PDF page ${currentPage}`}
              style={{ opacity: rendering ? 0.3 : 1 }}
            />
          </>
        )}
      </div>
    </aside>
  );
}
