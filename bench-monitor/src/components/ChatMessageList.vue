<script setup lang="ts">
import { ref, watch, nextTick } from 'vue';
import type { ChatMessage } from '../types/message';

const props = defineProps<{
  messages: ChatMessage[];
}>();

const messagesContainer = ref<HTMLElement | null>(null);
const collapsedSections = ref<Record<string, boolean>>({});

watch(() => props.messages.length, async () => {
  await nextTick();
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight;
  }
});

function getRoleIcon(role: string): string {
  switch (role) {
    case 'user':
      return '👤';
    case 'assistant':
      return '🤖';
    case 'tool':
      return '🔧';
    case 'system':
      return 'ℹ️';
    default:
      return '•';
  }
}

function getRoleLabel(role: string, toolName?: string): string {
  switch (role) {
    case 'user':
      return 'User';
    case 'assistant':
      return 'Assistant';
    case 'tool':
      return toolName ? `Tool: ${toolName}` : 'Tool';
    case 'system':
      return 'System';
    default:
      return role;
  }
}

function toggleSection(id: string, event: MouseEvent) {
  event.stopPropagation();
  event.preventDefault();
  const current = collapsedSections.value[id];
  collapsedSections.value[id] = current === undefined ? false : !current;
}

function isCollapsed(id: string): boolean {
  const val = collapsedSections.value[id];
  return val === undefined || val === true;
}

function formatTimestamp(timestamp: string): string {
  try {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  } catch {
    return timestamp;
  }
}

function formatDuration(ms?: number): string {
  if (!ms) return '';
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}
</script>

<template>
  <div class="chat-container" ref="messagesContainer">
    <div v-if="messages.length === 0" class="empty-state">
      <div class="empty-icon">📋</div>
      <div class="empty-text">Select a log file to view messages</div>
    </div>
    
    <div v-else class="messages-list">
      <div 
        v-for="(message, index) in messages" 
        :key="index" 
        class="message"
        :class="message.role"
      >
        <div class="message-header">
          <span class="role-icon">{{ getRoleIcon(message.role) }}</span>
          <span class="role-label">{{ message.role === 'tool' ? getRoleLabel(message.role, (message as any).toolName) : getRoleLabel(message.role) }}</span>
          <span class="timestamp">{{ formatTimestamp(message.timestamp) }}</span>
          <span v-if="message.role === 'assistant' && message.durationMs" class="duration">
            {{ formatDuration(message.durationMs) }}
          </span>
          <span v-if="message.role === 'tool' && (message as any).durationMs" class="duration">
            {{ formatDuration((message as any).durationMs) }}
          </span>
        </div>
        
        <div class="message-content">
          <!-- System message -->
          <template v-if="message.role === 'system'">
            <div class="system-content">{{ message.content }}</div>
            <div v-if="message.model" class="meta-info">Model: {{ message.model }}</div>
          </template>
          
          <!-- User message -->
          <template v-else-if="message.role === 'user'">
            <div class="user-content">{{ message.content }}</div>
          </template>
          
          <!-- Assistant message -->
          <template v-else-if="message.role === 'assistant'">
            <div v-if="message.thinkingContent" class="thinking">
              <div class="thinking-label">Thinking:</div>
              <pre class="thinking-content">{{ message.thinkingContent }}</pre>
            </div>
            <div v-if="message.content" class="assistant-content">{{ message.content }}</div>
            <div v-if="message.hasToolCalls" class="tool-badge">
              🔧 {{ message.toolCallCount || 0 }} tool call(s)
            </div>
          </template>
          
          <!-- Tool message -->
          <template v-else-if="message.role === 'tool'">
            <div v-if="message.arguments" class="tool-section collapsible">
              <div 
                class="section-header clickable" 
                @click="toggleSection(`args-${index}`, $event)"
              >
                <span class="collapse-icon">{{ isCollapsed(`args-${index}`) ? '▶' : '▼' }}</span>
                <span class="section-label">Arguments</span>
              </div>
              <pre v-if="!isCollapsed(`args-${index}`)" class="code-block">{{ message.arguments }}</pre>
            </div>
            <div v-if="message.result" class="tool-section collapsible">
              <div 
                class="section-header clickable" 
                @click="toggleSection(`result-${index}`, $event)"
              >
                <span class="collapse-icon">{{ isCollapsed(`result-${index}`) ? '▶' : '▼' }}</span>
                <span class="section-label">Result</span>
              </div>
              <pre v-if="!isCollapsed(`result-${index}`)" class="code-block">{{ message.result }}</pre>
            </div>
          </template>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat-container {
  flex: 1;
  height: 100%;
  overflow-y: auto;
  background: #1e1e1e;
  display: flex;
  flex-direction: column;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100%;
  color: #666;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.empty-text {
  font-size: 14px;
}

.messages-list {
  padding: 16px;
  width: 100%;
  flex: 1;
}

.message {
  margin-bottom: 24px;
  border-radius: 8px;
  overflow: hidden;
}

.message.user {
  background: #2d2d30;
}

.message.assistant {
  background: #252526;
}

.message.tool {
  background: #1e1e1e;
  border-left: 3px solid #f5a623;
}

.message.system {
  background: #0d1117;
  border-left: 3px solid #58a6ff;
}

.message-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  background: rgba(0, 0, 0, 0.2);
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.role-icon {
  font-size: 14px;
}

.role-label {
  font-weight: 600;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.message.user .role-label {
  color: #4fc3f7;
}

.message.assistant .role-label {
  color: #81c784;
}

.message.tool .role-label {
  color: #f5a623;
}

.message.system .role-label {
  color: #58a6ff;
}

.timestamp {
  margin-left: auto;
  font-size: 11px;
  color: #666;
}

.duration {
  font-size: 11px;
  color: #888;
  padding: 2px 6px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 4px;
}

.message-content {
  padding: 14px;
}

.system-content {
  color: #58a6ff;
  font-size: 13px;
}

.meta-info {
  margin-top: 8px;
  font-size: 11px;
  color: #666;
}

.user-content {
  color: #e0e0e0;
  font-size: 14px;
  line-height: 1.6;
}

.thinking {
  margin-bottom: 12px;
  padding: 12px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 6px;
}

.thinking-label {
  font-size: 11px;
  font-weight: 600;
  color: #ff9800;
  text-transform: uppercase;
  margin-bottom: 8px;
}

.thinking-content {
  font-size: 12px;
  color: #b0b0b0;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
  font-family: inherit;
}

.assistant-content {
  color: #e0e0e0;
  font-size: 14px;
  line-height: 1.6;
}

.tool-badge {
  display: inline-block;
  margin-top: 8px;
  padding: 4px 8px;
  background: rgba(245, 166, 35, 0.2);
  color: #f5a623;
  font-size: 12px;
  border-radius: 4px;
}

.tool-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.tool-name {
  font-weight: 600;
  color: #f5a623;
  font-size: 13px;
}

.tool-duration {
  font-size: 11px;
  color: #666;
}

.tool-section {
  margin-top: 10px;
}

.tool-section.collapsible .section-header {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  padding: 4px 0;
}

.tool-section.collapsible .section-header:hover {
  opacity: 0.8;
}

.collapse-icon {
  font-size: 10px;
  color: #f5a623;
}

.section-label {
  font-size: 11px;
  font-weight: 600;
  color: #888;
  text-transform: uppercase;
}

.code-block {
  background: #0d1117;
  padding: 10px;
  border-radius: 4px;
  font-size: 12px;
  color: #c9d1d9;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
  font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
  overflow-x: auto;
}
</style>
