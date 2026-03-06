<script setup lang="ts">
import type { Message } from '../stores/messages'

defineProps<{
  message: Message
}>()

function formatJson(obj: any): string {
  return JSON.stringify(obj, null, 2)
}

function formatTimestamp(ts: string): string {
  const date = new Date(ts)
  return date.toLocaleTimeString()
}
</script>

<template>
  <div class="message-wrapper">
    <template v-if="message.type === 'system'">
      <div class="message-card system">
        <div class="system-message">
          {{ message.content }}
        </div>
      </div>
    </template>
    
    <template v-else-if="message.type === 'user'">
      <div class="message-card user">
        <div class="user-message">
          <span class="iteration-badge" v-if="message.iteration">
            #{{ message.iteration }}
          </span>
          {{ message.content }}
        </div>
      </div>
    </template>
    
    <template v-else-if="message.type === 'assistant'">
      <div class="message-card assistant">
        <div class="assistant-message">
          <div v-if="message.thinkingContent" class="thinking-block">
            <details>
              <summary class="thinking-summary">
                💭 Thinking ({{ message.thinkingContent.length }} chars)
              </summary>
              <pre class="thinking-content">{{ message.thinkingContent }}</pre>
            </details>
          </div>
          
          <div v-if="message.content" class="content">
            {{ message.content }}
          </div>
          
          <div v-if="message.toolCalls && message.toolCalls.length > 0" class="tool-calls">
            <div class="tool-call-header">
              🔧 Tool Calls ({{ message.toolCalls.length }})
            </div>
            <div v-for="(call, idx) in message.toolCalls" :key="idx" class="tool-call-item">
              <code class="tool-name">{{ call.name }}</code>
              <pre v-if="call.args" class="tool-args">{{ formatJson(call.args) }}</pre>
            </div>
          </div>
          
          <div v-if="message.iteration" class="iteration-info">
            Iteration {{ message.iteration }} • {{ formatTimestamp(message.timestamp) }}
          </div>
        </div>
      </div>
    </template>
    
    <template v-else-if="message.type === 'tool'">
      <div class="message-card tool">
        <div class="tool-message">
          <div class="tool-header">
            <span class="tool-icon">🔧</span>
            <span class="tool-name">{{ message.toolName }}</span>
            <span v-if="message.iteration" class="iteration-badge">#{{ message.iteration }}</span>
          </div>
          
          <div v-if="message.toolCalls && message.toolCalls.length > 0" class="tool-arguments">
            <div class="section-label">Arguments:</div>
            <pre class="json-content">{{ formatJson(message.toolCalls[0].args) }}</pre>
          </div>
          
          <div v-if="message.toolResults && message.toolResults.length > 0" class="tool-results">
            <div class="section-label">Result:</div>
            <pre class="json-content">{{ formatJson(message.toolResults[0].result) }}</pre>
          </div>
          
          <div class="timestamp">
            {{ formatTimestamp(message.timestamp) }}
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.message-wrapper {
  margin-bottom: 1rem;
}

.message-card {
  padding: 0.75rem 1rem;
  border-radius: 0.5rem;
  max-width: 85%;
}

.message-card.system {
  background: #f9fafb;
  max-width: 100%;
  text-align: center;
}

.message-card.user {
  background: #3b82f6;
  color: white;
  margin-left: auto;
}

.message-card.assistant {
  background: #f3f4f6;
  margin-right: auto;
  max-width: 100%;
}

.message-card.tool {
  background: #ecfdf5;
  margin-right: auto;
  max-width: 100%;
}

.system-message {
  color: #6b7280;
  font-size: 0.875rem;
}

.user-message {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.iteration-badge {
  background: #e5e7eb;
  padding: 0.125rem 0.5rem;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 500;
}

.assistant-message {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.thinking-block {
  border-left: 3px solid #8b5cf6;
  padding-left: 0.75rem;
}

.thinking-summary {
  cursor: pointer;
  color: #8b5cf6;
  font-size: 0.875rem;
  font-weight: 500;
}

.thinking-content {
  margin-top: 0.5rem;
  padding: 0.75rem;
  background: #f3f4f6;
  border-radius: 0.5rem;
  font-size: 0.75rem;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
}

.tool-calls {
  margin-top: 0.5rem;
}

.tool-call-header {
  font-weight: 600;
  color: #059669;
  margin-bottom: 0.5rem;
}

.tool-call-item {
  padding: 0.5rem;
  margin-bottom: 0.5rem;
  background: #ecfdf5;
  border-radius: 0.375rem;
}

.tool-name {
  color: #059669;
  font-weight: 600;
}

.tool-args {
  margin-top: 0.25rem;
  font-size: 0.75rem;
  overflow-x: auto;
}

.iteration-info {
  font-size: 0.75rem;
  color: #9ca3af;
}

.tool-message {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.tool-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.tool-icon {
  font-size: 1rem;
}

.tool-name {
  font-weight: 600;
  color: #059669;
}

.section-label {
  font-size: 0.75rem;
  font-weight: 500;
  color: #6b7280;
  margin-bottom: 0.25rem;
}

.json-content {
  padding: 0.5rem;
  background: #f9fafb;
  border-radius: 0.375rem;
  font-size: 0.75rem;
  overflow-x: auto;
}

.timestamp {
  font-size: 0.75rem;
  color: #9ca3af;
}
</style>
