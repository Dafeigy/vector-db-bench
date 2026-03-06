import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface JsonlEvent {
  event: string
  iteration?: number
  message_count?: number
  has_tool_calls?: boolean
  tool_call_count?: number
  content?: string
  thinking_content?: string | null
  duration_ms?: number
  timestamp: string
  index?: number
  tool?: string
  arguments?: string
  tool_call_id?: string
  result?: any
  model?: string
  work_dir?: string
}

export interface Message {
  id: string
  type: 'user' | 'assistant' | 'tool' | 'system'
  content: string
  timestamp: string
  iteration?: number
  toolName?: string
  thinkingContent?: string | null
  toolCalls?: { name: string; args: any }[]
  toolResults?: { tool: string; result: any }[]
}

export const useMessageStore = defineStore('messages', () => {
  const messages = ref<Message[]>([])
  const isConnected = ref(false)

  function parseEvent(event: JsonlEvent): Message | null {
    const baseId = `${event.timestamp}-${event.event}`
    
    switch (event.event) {
      case 'llm_request':
        return {
          id: baseId,
          type: 'system',
          content: `[Iteration ${event.iteration}] LLM Request (${event.message_count} messages)`,
          timestamp: event.timestamp,
          iteration: event.iteration
        }
      
      case 'llm_response':
        return {
          id: baseId,
          type: 'assistant',
          content: event.content || '',
          timestamp: event.timestamp,
          iteration: event.iteration,
          thinkingContent: event.thinking_content,
          toolCalls: event.has_tool_calls && event.tool_call_count 
            ? Array.from({ length: event.tool_call_count }, () => ({ name: 'pending', args: {} }))
            : undefined
        }
      
      case 'tool_call':
        return {
          id: baseId,
          type: 'tool',
          content: '',
          timestamp: event.timestamp,
          iteration: event.iteration,
          toolName: event.tool || 'unknown',
          toolCalls: [{
            name: event.tool || 'unknown',
            args: event.arguments ? JSON.parse(event.arguments) : {}
          }]
        }
      
      case 'tool_result':
        return {
          id: baseId,
          type: 'tool',
          content: '',
          timestamp: event.timestamp,
          iteration: event.iteration,
          toolName: event.tool || 'unknown',
          toolResults: [{
            tool: event.tool || 'unknown',
            result: event.result
          }]
        }
      
      case 'session_start':
        return {
          id: baseId,
          type: 'system',
          content: `Session started: ${event.model || 'unknown'} (${event.work_dir || 'unknown'})`,
          timestamp: event.timestamp
        }
      
      case 'build_project':
      case 'run_correctness_test':
      case 'run_benchmark':
        return {
          id: baseId,
          type: 'tool',
          content: '',
          timestamp: event.timestamp,
          iteration: event.iteration,
          toolName: event.event.replace(/_/g, ' '),
          toolResults: event.result ? [{ tool: event.event, result: event.result }] : []
        }
      
      default:
        return null
    }
  }

  function loadAll(lines: string[]) {
    messages.value = []
    lines.forEach((line) => {
      try {
        const event = JSON.parse(line) as JsonlEvent
        const message = parseEvent(event)
        if (message) {
          messages.value.push(message)
        }
      } catch (e) {}
    })
  }

  return {
    messages,
    isConnected,
    loadAll
  }
})
