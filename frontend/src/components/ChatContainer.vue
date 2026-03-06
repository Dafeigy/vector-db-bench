<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import { useMessageStore } from '../stores/messages'
import ChatMessage from './ChatMessage.vue'

const store = useMessageStore()
const containerRef = ref<HTMLElement | null>(null)

function scrollToBottom() {
  nextTick(() => {
    if (containerRef.value) {
      containerRef.value.scrollTop = containerRef.value.scrollHeight
    }
  })
}

watch(
  () => store.messages.length,
  () => {
    scrollToBottom()
  }
)
</script>

<template>
  <div class="chat-container" ref="containerRef">
    <div v-if="store.messages.length === 0" class="empty-state">
      <p>No messages yet. Start monitoring a JSONL file.</p>
    </div>
    <div v-else class="messages">
      <ChatMessage
        v-for="message in store.messages"
        :key="message.id"
        :message="message"
      />
    </div>
  </div>
</template>

<style scoped>
.chat-container {
  height: 100%;
  overflow-y: auto;
  padding: 1rem;
  background: #ffffff;
}

.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #9ca3af;
}

.messages {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
</style>
