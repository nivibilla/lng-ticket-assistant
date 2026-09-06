export interface Citation {
  evidence_id: string;
  chunk_id: number;
  source: string;
  source_id: string;
  page_start: number | null;
  page_end: number | null;
  content: string;
  pdf_url: string | null;
  number: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  status: "pending" | "complete" | "failed";
  citations: Citation[];
  error?: string;
}

export interface Conversation {
  messages: Message[];
  busy: boolean;
}
export type ChatEvent =
  | { type: "accepted"; message: Message }
  | { type: "status"; phase: "thinking" | "searching"; clear_draft?: boolean }
  | { type: "message_start"; message_id: string }
  | { type: "delta"; message_id: string; text: string }
  | { type: "complete"; message: Message; user_message_id: string }
  | { type: "error"; message: string; user_message_id: string };
