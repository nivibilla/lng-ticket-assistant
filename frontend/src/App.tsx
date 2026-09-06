import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  Alert24Regular,
  Chat24Regular,
  Chat24Filled,
  Calendar24Regular,
  Call24Regular,
  Cloud24Regular,
  Apps24Regular,
  MoreHorizontal24Regular,
  Search20Regular,
  ChevronLeft20Regular,
  ChevronRight20Regular,
  ChevronDown16Regular,
  Compose20Regular,
  Filter20Regular,
  ArrowCounterclockwise20Regular,
  Bot24Regular,
  Sparkle20Regular,
  Send24Regular,
  TextFont20Regular,
  Emoji20Regular,
  Attach20Regular,
  Add20Regular,
  Video20Regular,
  PanelRight20Regular,
  Document20Regular,
  Checkmark16Regular,
  Stop20Filled,
  Copy20Regular,
  CheckmarkCircle16Filled,
  Dismiss20Regular,
  ArrowDown20Regular,
  Pin16Regular,
  ShieldCheckmark20Regular,
} from "@fluentui/react-icons";
import type { Citation, Message } from "./types";
import { getConversation, resetConversation, sendMessage } from "./lib/api";
import { MarkdownMessage } from "./components/MarkdownMessage";
const CitationPanel = lazy(() => import("./components/CitationPanel"));

const starters = [
  { label: "I can’t connect to the VPN", detail: "Connection & remote access" },
  { label: "I need help with my password", detail: "Accounts & sign-in" },
  { label: "Outlook isn’t working", detail: "Email & Microsoft 365" },
];
const welcome: Message = {
  id: "welcome",
  role: "assistant",
  content:
    "Hi, I’m your L&G Support Assistant. Tell me what’s going wrong, and I’ll help you find the next step using our IT knowledge base.",
  created_at: new Date().toISOString(),
  status: "complete",
  citations: [],
};

function BotAvatar({ large = false }: { large?: boolean }) {
  return (
    <span className={`bot-avatar${large ? " large" : ""}`}>
      <Bot24Regular />
      <span className="presence" />
    </span>
  );
}

function TeamsMark() {
  return (
    <svg
      width="29"
      height="28"
      viewBox="0 0 32 30"
      aria-label="Microsoft Teams"
    >
      <circle cx="22" cy="5" r="4" fill="#7b83eb" />
      <circle cx="29" cy="9" r="2.5" fill="#5059c9" />
      <path d="M24 13h7v8a3.5 3.5 0 0 1-7 0z" fill="#5059c9" />
      <path d="M13 11h14v11a7 7 0 0 1-14 0z" fill="#7b83eb" />
      <rect x="1" y="7" width="19" height="19" rx="2" fill="#4b53bc" />
      <path d="M5 12h11v2h-4v9H9v-9H5z" fill="white" />
    </svg>
  );
}

function Thinking({ phase }: { phase: string }) {
  return (
    <div className="thinking" role="status">
      <span className="thinking-dots">
        <i />
        <i />
        <i />
      </span>
      <span>
        {phase === "searching" ? "Searching the knowledge base…" : "Thinking…"}
      </span>
    </div>
  );
}

function timeLabel(value: string) {
  return new Date(value).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}
function dayLabel(value: string) {
  const date = new Date(value);
  return date.toDateString() === new Date().toDateString()
    ? "Today"
    : date.toLocaleDateString([], { day: "numeric", month: "long" });
}

function MessageRow({
  message,
  onCitation,
  onRetry,
  retryable,
}: {
  message: Message;
  onCitation: (c: Citation) => void;
  onRetry: (m: Message) => void;
  retryable: boolean;
}) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );
  return (
    <div className={`message-row ${message.role}`}>
      {message.role === "assistant" && <BotAvatar />}
      <div className="message-body">
        <div className="message-meta">
          {message.role === "assistant" && (
            <>
              <strong>L&G Support Assistant</strong>
              <span className="ai-tag">AI</span>
            </>
          )}
          <time dateTime={message.created_at}>
            {timeLabel(message.created_at)}
          </time>
        </div>
        <div className="message-bubble">
          {message.role === "user" ? (
            <div className="user-text">{message.content}</div>
          ) : (
            <MarkdownMessage
              text={message.content}
              citations={message.citations}
              onCitation={onCitation}
            />
          )}
          {message.role === "assistant" && (
            <button
              className="copy-message icon-button"
              aria-label={copied ? "Copied" : "Copy reply"}
              onClick={() => {
                void navigator.clipboard
                  .writeText(
                    message.content.replace(/\[\[cite:[^\]]+\]\]/g, ""),
                  )
                  .then(() => {
                    setCopied(true);
                    timer.current = setTimeout(() => setCopied(false), 2000);
                  })
                  .catch(() => setCopied(false));
              }}
            >
              {copied ? <Checkmark16Regular /> : <Copy20Regular />}
            </button>
          )}
        </div>
        {message.citations.length > 0 && (
          <div className="source-chips">
            {message.citations.map((citation) => (
              <button
                key={citation.evidence_id}
                onClick={() => onCitation(citation)}
                aria-label={`View source ${citation.number}`}
              >
                <Document20Regular />
                <span>{citation.source}</span>
                <span className="chip-page">
                  p. {citation.page_start ?? "—"}
                </span>
                <span className="chip-number">{citation.number}</span>
              </button>
            ))}
          </div>
        )}
        {message.status === "failed" && (
          <div className="message-error" role="alert">
            <span>{message.error || "Reply interrupted."}</span>
            {retryable && (
              <button onClick={() => onRetry(message)}>Try again</button>
            )}
          </div>
        )}
      </div>
      {message.role === "user" && message.status === "complete" && (
        <CheckmarkCircle16Filled className="sent-check" aria-label="Sent" />
      )}
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);
  const [phase, setPhase] = useState("thinking");
  const [draft, setDraft] = useState("");
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [error, setError] = useState("");
  const [resetOpen, setResetOpen] = useState(false);
  const [copiedNotice, setCopiedNotice] = useState("");
  const [awayFromBottom, setAwayFromBottom] = useState(false);
  const scroll = useRef<HTMLDivElement>(null);
  const composer = useRef<HTMLTextAreaElement>(null);
  const controller = useRef<AbortController | null>(null);
  const generation = useRef(0);
  const activeDraftId = useRef("");
  const opener = useRef<HTMLElement | null>(null);
  const busyRef = useRef(false);
  const polling = useRef<ReturnType<typeof setTimeout> | null>(null);

  const restore = useCallback(async (signal?: AbortSignal) => {
    const turn = generation.current;
    const state = await getConversation(signal);
    if (signal?.aborted || turn !== generation.current) return;
    setMessages(state.messages);
    setReady(true);
    setError("");
    setBusy(state.busy);
    busyRef.current = state.busy;
    if (state.busy)
      polling.current = setTimeout(() => {
        void restore(signal).catch((error) => {
          if (signal?.aborted || turn !== generation.current) return;
          busyRef.current = false;
          setBusy(false);
          setError(
            error instanceof Error
              ? error.message
              : "Connection interrupted. Please try again.",
          );
          setMessages((previous) =>
            previous.map((message) =>
              message.status === "pending"
                ? {
                    ...message,
                    status: "failed",
                    error: "The connection was interrupted. Please try again.",
                  }
                : message,
            ),
          );
        });
      }, 600);
  }, []);

  useEffect(() => {
    const abort = new AbortController();
    void restore(abort.signal).catch((error) => {
      if (!abort.signal.aborted) setError(error.message);
    });
    return () => {
      abort.abort();
      controller.current?.abort();
      if (polling.current) clearTimeout(polling.current);
    };
  }, [restore]);

  useEffect(() => {
    if (!awayFromBottom && scroll.current)
      scroll.current.scrollTop = scroll.current.scrollHeight;
  }, [messages, draft, busy, awayFromBottom]);
  useEffect(() => {
    if (composer.current) {
      composer.current.style.height = "auto";
      composer.current.style.height = `${Math.min(composer.current.scrollHeight, 160)}px`;
    }
  }, [input]);

  const openCitation = useCallback((citation: Citation) => {
    opener.current = document.activeElement as HTMLElement;
    setActiveCitation(citation);
  }, []);
  const closeCitation = useCallback(() => {
    setActiveCitation(null);
    opener.current?.focus();
  }, []);

  async function submit(text: string, retryId?: string) {
    const clean = text.trim();
    if (!clean || busyRef.current || !ready) return;
    if (polling.current) clearTimeout(polling.current);
    busyRef.current = true;
    setBusy(true);
    setDraft("");
    setPhase("thinking");
    setError("");
    setInput("");
    setAwayFromBottom(false);
    activeDraftId.current = "";
    const turn = ++generation.current;
    const abort = new AbortController();
    controller.current = abort;
    const optimisticId = retryId || `pending-${turn}`;
    let acceptedId = optimisticId;
    let accepted = false;
    setMessages((previous) =>
      retryId
        ? previous.map((m) =>
            m.id === retryId
              ? { ...m, status: "pending", error: undefined }
              : m,
          )
        : [
            ...previous,
            {
              id: optimisticId,
              role: "user",
              content: clean,
              created_at: new Date().toISOString(),
              status: "pending",
              citations: [],
            },
          ],
    );
    try {
      await sendMessage(
        clean,
        abort.signal,
        (event) => {
          if (turn !== generation.current) return;
          switch (event.type) {
            case "accepted":
              acceptedId = event.message.id;
              accepted = true;
              setMessages((previous) =>
                previous.map((m) =>
                  m.id === optimisticId ? event.message : m,
                ),
              );
              break;
            case "status":
              setPhase(event.phase);
              if (event.clear_draft) setDraft("");
              break;
            case "message_start":
              activeDraftId.current = event.message_id;
              setDraft("");
              break;
            case "delta":
              if (activeDraftId.current !== event.message_id) {
                activeDraftId.current = event.message_id;
                setDraft(event.text);
              } else setDraft((previous) => previous + event.text);
              break;
            case "complete":
              setMessages((previous) => [
                ...previous.map((m) =>
                  m.id === event.user_message_id
                    ? { ...m, status: "complete" as const }
                    : m,
                ),
                event.message,
              ]);
              setDraft("");
              break;
            case "error":
              setMessages((previous) =>
                previous.map((m) =>
                  m.id === event.user_message_id
                    ? { ...m, status: "failed", error: event.message }
                    : m,
                ),
              );
              setDraft("");
              break;
          }
        },
        retryId,
      );
    } catch (error) {
      if (turn !== generation.current) return;
      const message = abort.signal.aborted
        ? "Reply stopped. You can try again."
        : error instanceof Error
          ? error.message
          : "Connection interrupted. Please try again.";
      if (!accepted) {
        // A rejected request has no server message to retry; put the text back in the composer.
        setMessages((previous) =>
          retryId
            ? previous.map((m) =>
                m.id === retryId
                  ? { ...m, status: "failed", error: message }
                  : m,
              )
            : previous.filter((m) => m.id !== optimisticId),
        );
        setInput(clean);
        setError(message);
      } else
        setMessages((previous) =>
          previous.map((m) =>
            m.id === acceptedId && m.status === "pending"
              ? { ...m, status: "failed", error: message }
              : m,
          ),
        );
      setDraft("");
      if (abort.signal.aborted) {
        try {
          await restore();
        } catch {
          /* Keep the stopped state while disconnected. */
        }
      }
    } finally {
      if (turn === generation.current) {
        setBusy(false);
        busyRef.current = false;
        controller.current = null;
        composer.current?.focus();
      }
    }
  }

  async function reset() {
    const turn = ++generation.current;
    controller.current?.abort();
    if (polling.current) clearTimeout(polling.current);
    setResetOpen(false);
    setDraft("");
    setActiveCitation(null);
    try {
      await resetConversation();
      if (generation.current === turn) {
        setMessages([]);
        setError("");
        setInput("");
        setBusy(false);
        busyRef.current = false;
      }
    } catch (error) {
      setError(
        error instanceof Error
          ? error.message
          : "Couldn’t reset the conversation.",
      );
    } finally {
      if (generation.current === turn) {
        setBusy(false);
        busyRef.current = false;
        controller.current = null;
      }
    }
    composer.current?.focus();
  }

  const hasMessages = messages.length > 0;
  const visibleMessages = hasMessages
    ? [{ ...welcome, created_at: messages[0].created_at }, ...messages]
    : [];
  return (
    <div className={`teams-app${activeCitation ? " has-source" : ""}`}>
      <header className="topbar">
        <div className="app-identity">
          <TeamsMark />
          <span>Microsoft Teams</span>
        </div>
        <div className="topbar-center">
          <div className="history-controls" aria-hidden="true">
            <ChevronLeft20Regular />
            <ChevronRight20Regular />
          </div>
          <div className="global-search" role="search">
            <Search20Regular />
            <input
              aria-label="Search"
              placeholder="Search (Ctrl+Alt+E)"
              readOnly
              tabIndex={-1}
            />
            <span className="search-shortcut">⌘ E</span>
          </div>
        </div>
        <div className="topbar-right">
          <span className="demo-label">Demo</span>
          <MoreHorizontal24Regular className="decorative-icon" />
          <div className="user-avatar" title="You">
            YO
            <span className="presence" />
          </div>
        </div>
      </header>
      <nav className="app-rail" aria-label="Apps">
        <div className="rail-items">
          <div className="rail-item" aria-disabled="true">
            <Alert24Regular />
            <span>Activity</span>
          </div>
          <button
            className="rail-item selected"
            aria-current="page"
            onClick={() => composer.current?.focus()}
          >
            <Chat24Filled />
            <span>Chat</span>
          </button>
          <div className="rail-item" aria-disabled="true">
            <Calendar24Regular />
            <span>Calendar</span>
          </div>
          <div className="rail-item" aria-disabled="true">
            <Call24Regular />
            <span>Calls</span>
          </div>
          <div className="rail-item" aria-disabled="true">
            <Cloud24Regular />
            <span>OneDrive</span>
          </div>
          <div className="rail-item rail-more" aria-disabled="true">
            <MoreHorizontal24Regular />
          </div>
        </div>
        <div className="rail-item rail-apps" aria-disabled="true">
          <Apps24Regular />
          <span>Apps</span>
        </div>
      </nav>
      <aside className="chat-list" aria-label="Chat list">
        <div className="chat-list-heading">
          <h1>Chat</h1>
          <div className="list-actions">
            <MoreHorizontal24Regular className="decorative-icon" />
            <Filter20Regular className="decorative-icon" />
            <button
              className="icon-button"
              onClick={() =>
                hasMessages ? setResetOpen(true) : composer.current?.focus()
              }
              aria-label="New chat"
            >
              <Compose20Regular />
            </button>
          </div>
        </div>
        <div className="chat-filters">
          <span>Unread</span>
          <span className="active-filter">Chats</span>
          <span>Channels</span>
          <ChevronDown16Regular />
        </div>
        <div className="list-shortcut">
          <Sparkle20Regular />
          <span>Copilot</span>
        </div>
        <div className="list-shortcut">
          <span className="at-icon">@</span>
          <span>Mentions</span>
        </div>
        <div className="list-section">
          <ChevronDown16Regular />
          <span>Favourites</span>
        </div>
        <button
          className="chat-list-entry"
          aria-current="true"
          onClick={() => composer.current?.focus()}
        >
          <BotAvatar />
          <span className="chat-entry-copy">
            <strong>L&G Support Assistant</strong>
            <span>{busy ? "Thinking…" : "Your IT support assistant"}</span>
          </span>
          <Pin16Regular />
        </button>
        <div className="list-section lower-section">
          <ChevronDown16Regular />
          <span>Chats</span>
        </div>
        <p className="list-empty">Your conversations will appear here.</p>
        <div className="workspace-label">
          <span className="workspace-mark">L&G</span>
          <div>
            <strong>Legal & General</strong>
            <span>Support workspace</span>
          </div>
        </div>
      </aside>
      <main className="chat-main">
        <header className="conversation-header">
          <div className="conversation-identity">
            <BotAvatar />
            <div>
              <h2>
                L&G Support Assistant <span className="ai-tag">AI</span>
              </h2>
              <span className="header-subtitle">Here to help</span>
            </div>
          </div>
          <div className="conversation-tabs">
            <span className="selected-tab">Chat</span>
            <span>Shared</span>
            <Add20Regular />
          </div>
          <div className="conversation-actions">
            <Video20Regular className="decorative-icon optional-icon" />
            <Call24Regular className="decorative-icon optional-icon" />
            <span className="vertical-divider" />
            <button
              className="icon-button"
              disabled={!hasMessages && !busy}
              title="Reset conversation"
              aria-label="Reset conversation"
              onClick={() => setResetOpen(true)}
            >
              <ArrowCounterclockwise20Regular />
            </button>
            {activeCitation && (
              <button
                className="icon-button active-icon"
                onClick={closeCitation}
                aria-label="Hide sources"
              >
                <PanelRight20Regular />
              </button>
            )}
            <MoreHorizontal24Regular className="decorative-icon" />
          </div>
        </header>
        <div className="chat-content">
          <div
            className="conversation-scroll"
            ref={scroll}
            onScroll={() => {
              const element = scroll.current;
              if (element)
                setAwayFromBottom(
                  element.scrollHeight -
                    element.scrollTop -
                    element.clientHeight >
                    80,
                );
            }}
            role="log"
            aria-label="Conversation"
            aria-live="off"
          >
            {!hasMessages && (
              <div className="empty-conversation">
                <div className="welcome-avatar">
                  <BotAvatar large />
                  <span className="welcome-sparkle">
                    <Sparkle20Regular />
                  </span>
                </div>
                <span className="welcome-eyebrow">
                  YOUR IT SUPPORT ASSISTANT
                </span>
                <h2>A little help. A lot less hassle.</h2>
                <p>
                  Hi, I’m your L&G Support Assistant.
                  <br />
                  Tell me what’s going wrong. We’ll find your next step.
                </p>
                <div className="starter-prompts">
                  {starters.map((starter) => (
                    <button
                      key={starter.label}
                      onClick={() => void submit(starter.label)}
                      disabled={!ready || busy}
                    >
                      <Chat24Regular />
                      <strong>{starter.label}</strong>
                      <span>{starter.detail}</span>
                      <ChevronRight20Regular className="starter-arrow" />
                    </button>
                  ))}
                </div>
                <div className="grounding-note">
                  <ShieldCheckmark20Regular />
                  <span>Answers grounded in our IT knowledge base</span>
                </div>
              </div>
            )}
            {visibleMessages.map((message, index) => (
              <div key={message.id}>
                {(index === 0 ||
                  new Date(message.created_at).toDateString() !==
                    new Date(
                      visibleMessages[index - 1].created_at,
                    ).toDateString()) && (
                  <div className="day-divider">
                    <span>{dayLabel(message.created_at)}</span>
                  </div>
                )}
                <MessageRow
                  message={message}
                  onCitation={openCitation}
                  onRetry={(m) => void submit(m.content, m.id)}
                  retryable={!busy && message.id === messages.at(-1)?.id}
                />
              </div>
            ))}
            {busy && (
              <div className="message-row assistant streaming-row">
                <BotAvatar />
                <div className="message-body">
                  <div className="message-meta">
                    <strong>L&G Support Assistant</strong>
                    <span className="ai-tag">AI</span>
                  </div>
                  {draft ? (
                    <div className="message-bubble streaming-bubble">
                      <MarkdownMessage
                        text={draft}
                        citations={[]}
                        streaming
                        onCitation={openCitation}
                      />
                      <span className="stream-cursor" />
                    </div>
                  ) : (
                    <Thinking phase={phase} />
                  )}
                </div>
              </div>
            )}
          </div>
          {awayFromBottom && (
            <button
              className="jump-to-bottom"
              onClick={() => {
                setAwayFromBottom(false);
                scroll.current?.scrollTo({
                  top: scroll.current.scrollHeight,
                  behavior: "smooth",
                });
              }}
            >
              <ArrowDown20Regular />
              Latest messages
            </button>
          )}
          <div className="composer-area">
            {error && (
              <div className="connection-error" role="alert">
                <span>{error}</span>
                {!ready && (
                  <button
                    onClick={() =>
                      void restore().catch((error) => setError(error.message))
                    }
                  >
                    Reconnect
                  </button>
                )}
                <button
                  className="icon-button"
                  aria-label="Dismiss error"
                  onClick={() => setError("")}
                >
                  <Dismiss20Regular />
                </button>
              </div>
            )}
            <form
              className={`composer${busy ? " is-busy" : ""}`}
              onSubmit={(event) => {
                event.preventDefault();
                void submit(input);
              }}
            >
              <textarea
                ref={composer}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder={
                  ready ? "Type a message" : "Connecting to your assistant…"
                }
                aria-label="Message L&G Support Assistant"
                rows={1}
                maxLength={12000}
                disabled={!ready}
                onKeyDown={(event) => {
                  if (
                    event.key === "Enter" &&
                    !event.shiftKey &&
                    !event.nativeEvent.isComposing
                  ) {
                    event.preventDefault();
                    void submit(input);
                  }
                }}
              />
              <div className="composer-toolbar">
                <div className="composer-tools" aria-hidden="true">
                  <TextFont20Regular />
                  <Emoji20Regular />
                  <Attach20Regular />
                  <Add20Regular />
                </div>
                <span className="composer-hint">
                  Shift + Enter for a new line
                </span>
                {busy ? (
                  <button
                    type="button"
                    className="send-button stop-button"
                    aria-label="Stop reply"
                    onClick={() => {
                      controller.current?.abort();
                      if (!controller.current)
                        setCopiedNotice(
                          "This reply is running in another tab. Reset the conversation to stop it.",
                        );
                    }}
                  >
                    <Stop20Filled />
                  </button>
                ) : (
                  <button
                    type="submit"
                    className="send-button"
                    aria-label="Send message"
                    disabled={!input.trim() || !ready}
                  >
                    <Send24Regular />
                  </button>
                )}
              </div>
            </form>
            <div className="composer-caption">
              <Sparkle20Regular />
              <span>
                AI-generated answers. Check the sources for important details.
              </span>
            </div>
          </div>
        </div>
      </main>
      {activeCitation && (
        <>
          <button
            className="source-backdrop"
            aria-label="Close sources"
            onClick={closeCitation}
            tabIndex={-1}
          />
          <Suspense
            fallback={
              <aside className="citation-panel panel-skeleton" role="status">
                Opening source…
              </aside>
            }
          >
            <CitationPanel citation={activeCitation} onClose={closeCitation} />
          </Suspense>
        </>
      )}
      {resetOpen && (
        <div className="modal-backdrop" onClick={() => setResetOpen(false)}>
          <div
            className="reset-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="reset-title"
            onClick={(event) => event.stopPropagation()}
            onKeyDown={(event) => {
              if (event.key === "Escape") setResetOpen(false);
              if (event.key === "Tab") {
                event.preventDefault();
                const buttons = event.currentTarget.querySelectorAll("button");
                (document.activeElement === buttons[0]
                  ? buttons[1]
                  : buttons[0]
                ).focus();
              }
            }}
          >
            <h2 id="reset-title">Start a fresh conversation?</h2>
            <p>
              This will clear this chat and its sources. Any reply in progress
              will stop.
            </p>
            <div>
              <button
                autoFocus
                className="secondary-button"
                onClick={() => {
                  setResetOpen(false);
                  composer.current?.focus();
                }}
              >
                Cancel
              </button>
              <button className="primary-button" onClick={() => void reset()}>
                Start fresh
              </button>
            </div>
          </div>
        </div>
      )}
      {copiedNotice && (
        <div className="toast" role="status">
          {copiedNotice}
          <button
            className="icon-button"
            onClick={() => setCopiedNotice("")}
            aria-label="Dismiss notice"
          >
            <Dismiss20Regular />
          </button>
        </div>
      )}
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        {messages.at(-1)?.role === "assistant"
          ? "L&G Support Assistant replied. The answer is available in the conversation."
          : ""}
      </div>
    </div>
  );
}
