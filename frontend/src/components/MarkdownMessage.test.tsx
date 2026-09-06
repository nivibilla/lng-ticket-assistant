import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MarkdownMessage } from "./MarkdownMessage";
import type { Citation } from "../types";

const citation: Citation = {
  evidence_id: "chunk-7",
  chunk_id: 7,
  source: "KB Articles.pdf",
  source_id: "pdf",
  page_start: 3,
  page_end: 4,
  content: "Reconnect the VPN client.",
  pdf_url: "/api/sources/pdf",
  number: 1,
};

afterEach(cleanup);
describe("safe Markdown answers", () => {
  it("opens the exact source from an inline citation", () => {
    const open = vi.fn();
    render(
      <MarkdownMessage
        text="Reconnect. [[cite:chunk-7]]"
        citations={[citation]}
        onCitation={open}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", {
        name: "Source 1: KB Articles.pdf, Pages 3–4",
      }),
    );
    expect(open).toHaveBeenCalledWith(citation);
  });
  it("does not execute HTML or load remote images and rejects unsafe links", () => {
    const { container } = render(
      <MarkdownMessage
        text={
          '<script>alert(1)</script>\n\n<img src="x" onerror="alert(1)">\n\n![image](https://example.com/private)\n\n[bad](javascript:alert(1))'
        }
        citations={[]}
        onCitation={vi.fn()}
      />,
    );
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("a")).toBeNull();
  });
});
