<script setup lang="ts">
import { onMounted, onUnmounted } from 'vue'
import { useMessageStore } from './stores/messages'
import ChatContainer from './components/ChatContainer.vue'

const store = useMessageStore()
let eventSource: EventSource | null = null

async function loadData() {
  try {
    const res = await fetch('/api/lines')
    const data = await res.json()
    if (data.lines) {
      store.loadAll(data.lines)
    }
  } catch (err) {
    console.error('Failed to load data:', err)
  }
}

function connectSSE() {
  console.log('Connecting to SSE...')
  eventSource = new EventSource('/api/events')
  
  eventSource.onopen = () => {
    console.log('SSE connected')
    store.isConnected = true
  }
  
  eventSource.onmessage = (event) => {
    console.log('SSE message received:', event.data)
    loadData()
  }
  
  eventSource.onerror = (err) => {
    console.error('SSE error:', err)
    console.error('ReadyState:', eventSource?.readyState)
    store.isConnected = false
    eventSource?.close()
    setTimeout(connectSSE, 3000)
  }
  
  eventSource.onmessageerror = (err) => {
    console.error('SSE message error:', err)
  }
}

onMounted(() => {
  loadData()
  connectSSE()
})

onUnmounted(() => {
  eventSource?.close()
})
</script>

<template>
  <div class="app">
    <header class="header">
      <h1 class="title">JSONL Monitor</h1>
      <div class="status">
        <span 
          class="status-dot"
          :class="{ connected: store.isConnected }"
        ></span>
        <span class="status-text">
          {{ store.isConnected ? 'Live' : 'Disconnected' }}
        </span>
        <span class="message-count">
          {{ store.messages.length }} messages
        </span>
      </div>
    </header>
    <main class="main">
      <ChatContainer />
    </main>
  </div>
</template>

<style scoped>
.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  max-width: 1200px;
  margin: 0 auto;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.5rem;
  background: #ffffff;
  border-bottom: 1px solid #e5e7eb;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.title {
  font-size: 1.25rem;
  font-weight: 600;
  color: #111827;
  margin: 0;
}

.status {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #ef4444;
}

.status-dot.connected {
  background: #22c55e;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

.status-text {
  font-size: 0.875rem;
  color: #6b7280;
}

.message-count {
  font-size: 0.875rem;
  color: #9ca3af;
  padding: 0.25rem 0.75rem;
  background: #f3f4f6;
  border-radius: 9999px;
}

.main {
  flex: 1;
  overflow: hidden;
}
</style>
