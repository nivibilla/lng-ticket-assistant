import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  act,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { getConversation, resetConversation, sendMessage } from "./lib/api";
import type { ChatEvent, Message } from "./types";

vi.mock("./lib/api", () => ({
  getConversation: vi.fn(),
  resetConversation: vi.fn(),
  sendMessage: vi.fn(),
}));
const user: Message = {
  id: "u1",
  role: "user",
  content: "VPN fails",
  created_at: "2026-09-06T12:00:00Z",
  status: "pending",
  citations: [],
};
const answer: Message = {
  id: "a1",
  role: "assistant",
  content: "Reconnect the VPN.",
  created_at: "2026-09-06T12:00:10Z",
  status: "complete",
  citations: [],
};
let emit: (event: ChatEvent) => void;
let complete: () => void;

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getConversation).mockResolvedValue({ messages: [], busy: false });
  vi.mocked(resetConversation).mockResolvedValue({ messages: [], busy: false });
  vi.mocked(sendMessage).mockImplementation((_text, signal, onEvent) => {
    emit = onEvent;
    return new Promise((resolve, reject) => {
      complete = resolve;
      signal.addEventListener("abort", () =>
        reject(new DOMException("Aborted", "AbortError")),
      );
    });
  });
});
afterEach(cleanup);

async function start() {
  render(<App />);
  const input = screen.getByRole("textbox", {
    name: "Message L&G Support Assistant",
  });
  await waitFor(() => expect(input).toBeEnabled());
  fireEvent.change(input, { target: { value: "VPN fails" } });
  fireEvent.click(screen.getByRole("button", { name: "Send message" }));
}

describe("chat lifecycle", () => {
  it("pulses during retrieval, streams a draft and reconciles the final answer", async () => {
    await start();
    expect(screen.getByRole("status")).toHaveTextContent("Thinking");
    act(() => {
      emit({ type: "accepted", message: user });
      emit({ type: "status", phase: "searching" });
    });
    expect(screen.getByRole("status")).toHaveTextContent(
      "Searching the knowledge base",
    );
    act(() => {
      emit({ type: "message_start", message_id: "model1" });
      emit({ type: "delta", message_id: "model1", text: "Reconnect " });
    });
    expect(screen.getByText("Reconnect")).toBeVisible();
    await act(async () => {
      emit({ type: "complete", message: answer, user_message_id: "u1" });
      complete();
    });
    expect(screen.getByText("Reconnect the VPN.")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Stop reply" })).toBeNull();
  });
  it("shows retry on a failed turn and reuses its server ID", async () => {
    await start();
    await act(async () => {
      emit({ type: "accepted", message: user });
      emit({
        type: "error",
        message: "Please try again.",
        user_message_id: "u1",
      });
      complete();
    });
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(vi.mocked(sendMessage).mock.calls[1][3]).toBe("u1");
    expect(screen.getAllByText("VPN fails")).toHaveLength(1);
  });
  it("restores committed history on load and resets it", async () => {
    vi.mocked(getConversation).mockResolvedValue({
      messages: [{ ...user, status: "complete" }, answer],
      busy: false,
    });
    render(<App />);
    await screen.findByText("Reconnect the VPN.");
    fireEvent.click(screen.getByRole("button", { name: "Reset conversation" }));
    fireEvent.click(screen.getByRole("button", { name: "Start fresh" }));
    await screen.findByText("A little help. A lot less hassle.");
    expect(resetConversation).toHaveBeenCalledOnce();
  });
  it("ignores late completion after reset", async () => {
    await start();
    act(() => emit({ type: "accepted", message: user }));
    fireEvent.click(screen.getByRole("button", { name: "Reset conversation" }));
    fireEvent.click(screen.getByRole("button", { name: "Start fresh" }));
    await screen.findByText("A little help. A lot less hassle.");
    act(() =>
      emit({ type: "complete", message: answer, user_message_id: "u1" }),
    );
    expect(screen.queryByText("Reconnect the VPN.")).toBeNull();
  });
  it("stops a reply and restores the server’s retryable message", async () => {
    await start();
    act(() => emit({ type: "accepted", message: user }));
    vi.mocked(getConversation).mockResolvedValue({
      messages: [
        {
          ...user,
          status: "failed",
          error: "Reply stopped. You can try again.",
        },
      ],
      busy: false,
    });
    fireEvent.click(screen.getByRole("button", { name: "Stop reply" }));
    await screen.findByRole("button", { name: "Try again" });
    expect(vi.mocked(sendMessage).mock.calls[0][1].aborted).toBe(true);
  });
});
