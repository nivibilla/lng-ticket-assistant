import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import CitationPanel from "./CitationPanel";
import type { Citation } from "../types";

const mocks = vi.hoisted(() => ({ getPage: vi.fn(), destroy: vi.fn() }));
vi.mock("pdfjs-dist", () => ({
  GlobalWorkerOptions: {},
  getDocument: () => ({
    promise: Promise.resolve({ numPages: 9, getPage: mocks.getPage }),
    destroy: mocks.destroy,
  }),
}));
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

beforeEach(() => {
  mocks.getPage.mockReset().mockResolvedValue({
    getViewport: ({ scale }: { scale: number }) => ({
      width: 600 * scale,
      height: 800 * scale,
    }),
    render: () => ({ promise: Promise.resolve(), cancel: vi.fn() }),
  });
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("source sidebar", () => {
  it("opens the cited page, changes pages, zooms, and switches citations", async () => {
    const { rerender } = render(
      <CitationPanel citation={citation} onClose={vi.fn()} />,
    );
    await screen.findByText("Page 3 of 9");
    expect(mocks.getPage).toHaveBeenCalledWith(3);
    expect(
      screen.getByRole("link", { name: "Open original PDF in new tab" }),
    ).toHaveAttribute("href", "/api/sources/pdf#page=3");
    fireEvent.click(screen.getByRole("button", { name: "Next PDF page" }));
    await screen.findByText("Page 4 of 9");
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    expect(
      screen.getByRole("button", { name: "Fit PDF to width" }),
    ).toHaveTextContent("125%");
    rerender(
      <CitationPanel
        citation={{ ...citation, evidence_id: "chunk-8", page_start: 8 }}
        onClose={vi.fn()}
      />,
    );
    await screen.findByText("Page 8 of 9");
    expect(
      screen.getByRole("button", { name: "Fit PDF to width" }),
    ).toHaveTextContent("100%");
    await waitFor(() => expect(mocks.getPage).toHaveBeenCalledWith(8));
  });
  it("keeps evidence visible if the original PDF is missing", () => {
    const close = vi.fn();
    render(
      <CitationPanel
        citation={{ ...citation, pdf_url: null }}
        onClose={close}
      />,
    );
    expect(screen.getByText("Reconnect the VPN client.")).toBeVisible();
    expect(screen.getByText(/original PDF is unavailable/)).toBeVisible();
    expect(screen.queryByRole("link")).toBeNull();
    fireEvent.click(
      screen.getByRole("button", { name: "Close source details" }),
    );
    expect(close).toHaveBeenCalledOnce();
  });
});
