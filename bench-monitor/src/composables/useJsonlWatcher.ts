import { ref, computed, onUnmounted } from 'vue';
import type { 
  ChatMessage, 
  LeaderboardEntry, 
  AnyJsonlEvent,
  SessionStartEvent,
  LlmRequestEvent,
  LlmResponseEvent,
  ToolCallEvent,
  ToolResultEvent,
  ToolMessage
} from '../types/message';

const generateId = () => Math.random().toString(36).substring(2, 15);

function parseToolArguments(args: string): string {
  try {
    return JSON.stringify(JSON.parse(args), null, 2);
  } catch {
    return args;
  }
}

function parseToolResult(result: string | { content: string }): string {
  if (typeof result === 'string') {
    try {
      const parsed = JSON.parse(result);
      if (typeof parsed === 'object' && parsed !== null) {
        return JSON.stringify(parsed, null, 2);
      }
      return result;
    } catch {
      return result;
    }
  }
  if (typeof result === 'object' && result !== null && 'content' in result) {
    return result.content;
  }
  return JSON.stringify(result, null, 2);
}

function eventToMessage(event: AnyJsonlEvent): ChatMessage | null {
  const baseId = generateId();
  
  switch (event.event) {
    case 'session_start': {
      const e = event as SessionStartEvent;
      return {
        id: baseId,
        role: 'system',
        timestamp: e.timestamp,
        content: `会话开始 - 模型: ${e.model}`,
        model: e.model,
        workDir: e.work_dir,
      };
    }
    
    case 'llm_request': {
      const e = event as LlmRequestEvent;
      return {
        id: baseId,
        role: 'user',
        timestamp: e.timestamp,
        content: `[迭代 ${e.iteration}] LLM请求 - 消息数: ${e.message_count}`,
        iteration: e.iteration,
        messageCount: e.message_count,
      };
    }
    
    case 'llm_response': {
      const e = event as LlmResponseEvent;
      let content = e.content;
      if (!content && e.thinking_content) {
        content = '[无内容]';
      }
      return {
        id: baseId,
        role: 'assistant',
        timestamp: e.timestamp,
        content,
        thinkingContent: e.thinking_content,
        iteration: e.iteration,
        hasToolCalls: e.has_tool_calls,
        toolCallCount: e.tool_call_count,
        durationMs: e.duration_ms,
      };
    }
    
    case 'tool_call': {
      const e = event as ToolCallEvent;
      return {
        id: baseId,
        role: 'tool',
        timestamp: e.timestamp,
        toolName: e.tool,
        toolCallId: e.tool_call_id,
        arguments: parseToolArguments(e.arguments),
        index: e.index,
      };
    }
    
    case 'tool_result': {
      const e = event as ToolResultEvent;
      return {
        id: baseId,
        role: 'tool',
        timestamp: e.timestamp,
        toolName: e.tool,
        toolCallId: e.tool_call_id,
        result: parseToolResult(e.result),
        resultType: e.type,
        durationMs: e.duration_ms,
        index: e.index,
      };
    }
    
    default:
      return null;
  }
}

function parseJsonl(content: string): ChatMessage[] {
  const lines = content.trim().split('\n');
  const messages: ChatMessage[] = [];
  const pendingToolCalls = new Map<string, ToolMessage>();
  
  for (const line of lines) {
    if (!line.trim()) continue;
    try {
      const event = JSON.parse(line) as AnyJsonlEvent;
      
      if (event.event === 'tool_call') {
        const e = event as ToolCallEvent;
        const message: ToolMessage = {
          id: generateId(),
          role: 'tool',
          timestamp: e.timestamp,
          toolName: e.tool,
          toolCallId: e.tool_call_id,
          arguments: parseToolArguments(e.arguments),
          index: e.index,
        };
        pendingToolCalls.set(e.tool_call_id, message);
      } else if (event.event === 'tool_result') {
        const e = event as ToolResultEvent;
        const existingCall = pendingToolCalls.get(e.tool_call_id);
        if (existingCall) {
          existingCall.result = parseToolResult(e.result);
          existingCall.resultType = e.type;
          existingCall.durationMs = e.duration_ms;
          messages.push(existingCall);
          pendingToolCalls.delete(e.tool_call_id);
        } else {
          const message: ToolMessage = {
            id: generateId(),
            role: 'tool',
            timestamp: e.timestamp,
            toolName: e.tool,
            toolCallId: e.tool_call_id,
            result: parseToolResult(e.result),
            resultType: e.type,
            durationMs: e.duration_ms,
            index: e.index,
          };
          messages.push(message);
        }
      } else {
        const message = eventToMessage(event);
        if (message) {
          messages.push(message);
        }
      }
    } catch (e) {
      console.warn('Failed to parse JSONL line:', e);
    }
  }
  
  for (const remaining of pendingToolCalls.values()) {
    messages.push(remaining);
  }
  
  return messages;
}

export function useJsonlWatcher() {
  const messages = ref<ChatMessage[]>([]);
  const isConnected = ref(false);
  const currentFile = ref<string>('');
  const isLoading = ref(false);
  const error = ref<string | null>(null);
  
  const lastFileSize = ref(0);
  let pollInterval: ReturnType<typeof setInterval> | null = null;
  
  async function fetchMessages(url: string): Promise<ChatMessage[]> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch: ${response.status}`);
    }
    const content = await response.text();
    return parseJsonl(content);
  }
  
  async function startWatching(fileUrl: string) {
    stopWatching();
    
    currentFile.value = fileUrl;
    isLoading.value = true;
    error.value = null;
    
    try {
      const initialMessages = await fetchMessages(fileUrl);
      messages.value = initialMessages;
      isConnected.value = true;
      
      const response = await fetch(fileUrl);
      const content = await response.text();
      lastFileSize.value = content.length;
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Unknown error';
      isConnected.value = false;
    } finally {
      isLoading.value = false;
    }
    
    pollInterval = setInterval(async () => {
      if (!currentFile.value) return;
      
      try {
        const newMessages = await fetchMessages(currentFile.value);
        messages.value = newMessages;
        isConnected.value = true;
        error.value = null;
      } catch (e) {
        error.value = e instanceof Error ? e.message : 'Unknown error';
        isConnected.value = false;
      }
    }, 1000);
  }
  
  function stopWatching() {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
    isConnected.value = false;
  }
  
  onUnmounted(() => {
    stopWatching();
  });
  
  return {
    messages,
    isConnected,
    currentFile,
    isLoading,
    error,
    startWatching,
    stopWatching,
  };
}

export function useLeaderboardScanner() {
  const entries = ref<LeaderboardEntry[]>([]);
  const isLoading = ref(false);
  
  async function scanLeaderboard(baseUrl: string = '/leaderboard') {
    isLoading.value = true;
    entries.value = [];
    
    try {
      const response = await fetch(baseUrl);
      const text = await response.text();
      
      const dirRegex = /<a href="([^"]+\/)">([^<]+)<\/a>/g;
      let match;
      
      while ((match = dirRegex.exec(text)) !== null) {
        const dirName = match[2];
        const dirPath = match[1];
        
        if (!dirName || !dirPath) continue;
        
        const turnMatch = dirName.match(/-turn-(\d+)$/);
        if (turnMatch && turnMatch[1]) {
          const turn = parseInt(turnMatch[1], 10);
          const modelName = dirName.replace(/-turn-\d+$/, '');
          
          entries.value.push({
            name: `${modelName} (Turn ${turn})`,
            path: `${baseUrl}/${dirPath}agent_log.jsonl`,
            turn,
          });
        }
      }
      
      entries.value.sort((a, b) => {
        if (a.name !== b.name) {
          return a.name.localeCompare(b.name);
        }
        return a.turn - b.turn;
      });
    } catch (e) {
      console.error('Failed to scan leaderboard:', e);
    } finally {
      isLoading.value = false;
    }
  }
  
  return {
    entries,
    isLoading,
    scanLeaderboard,
  };
}
