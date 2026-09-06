import { memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Citation } from "../types";
import { pageLabel, prepareMarkdown } from "../lib/citations";

export const MarkdownMessage = memo(function MarkdownMessage({
  text,
  citations,
  streaming = false,
  onCitation,
}: {
  text: string;
  citations: Citation[];
  streaming?: boolean;
  onCitation: (citation: Citation) => void;
}) {
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        skipHtml
        components={{
          a: ({ href, children }) => {
            const citation = citations.find(
              (c) => href === `#evidence-${c.evidence_id}`,
            );
            if (citation)
              return (
                <button
                  className="citation-button"
                  onClick={() => onCitation(citation)}
                  aria-label={`Source ${citation.number}: ${citation.source}, ${pageLabel(citation)}`}
                >
                  {citation.number}
                </button>
              );
            if (!href || !/^(https?:|mailto:)/i.test(href))
              return <span>{children}</span>;
            return (
              <a href={href} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            );
          },
          img: ({ alt }) => <span>{alt}</span>,
        }}
      >
        {prepareMarkdown(text, citations, streaming)}
      </ReactMarkdown>
    </div>
  );
});
