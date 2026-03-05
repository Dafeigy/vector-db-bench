export type MessageRole = 'user' | 'assistant' | 'tool' | 'system';

export interface BaseMessage {
  id: string;
  role: MessageRole;
  timestamp: string;
}

export interface UserMessage extends BaseMessage {
  role: 'user';
  content: string;
  iteration: number;
  messageCount: number;
}

export interface AssistantMessage extends BaseMessage {
  role: 'assistant';
  content: string;
  thinkingContent?: string;
  iteration: number;
  hasToolCalls: boolean;
  toolCallCount?: number;
  durationMs?: number;
}

export interface ToolMessage extends BaseMessage {
  role: 'tool';
  toolName: string;
  toolCallId: string;
  arguments?: string;
  result?: string;
  resultType?: string;
  durationMs?: number;
  index: number;
}

export interface SystemMessage extends BaseMessage {
  role: 'system';
  content: string;
  model?: string;
  workDir?: string;
}

export type ChatMessage = UserMessage | AssistantMessage | ToolMessage | SystemMessage;

export interface JsonlEvent {
  event: 'session_start' | 'llm_request' | 'llm_response' | 'tool_call' | 'tool_result';
  timestamp: string;
}

export interface SessionStartEvent extends JsonlEvent {
  event: 'session_start';
  model: string;
  work_dir: string;
}

export interface LlmRequestEvent extends JsonlEvent {
  event: 'llm_request';
  iteration: number;
  message_count: number;
}

export interface LlmResponseEvent extends JsonlEvent {
  event: 'llm_response';
  iteration: number;
  has_tool_calls: boolean;
  tool_call_count?: number;
  content: string;
  thinking_content?: string;
  duration_ms: number;
}

export interface ToolCallEvent extends JsonlEvent {
  event: 'tool_call';
  index: number;
  tool: string;
  arguments: string;
  tool_call_id: string;
}

export interface ToolResultEvent extends JsonlEvent {
  event: 'tool_result';
  index: number;
  tool: string;
  tool_call_id: string;
  result: string | { content: string };
  type: string;
  duration_ms: number;
}

export type AnyJsonlEvent = SessionStartEvent | LlmRequestEvent | LlmResponseEvent | ToolCallEvent | ToolResultEvent;

export interface LeaderboardEntry {
  name: string;
  path: string;
  turn: number;
}
