<script setup lang="ts">
import { ref, onMounted } from 'vue';
import JsonlSelector from './components/JsonlSelector.vue';
import ChatMessageList from './components/ChatMessageList.vue';
import { useJsonlWatcher, useLeaderboardScanner } from './composables/useJsonlWatcher';

const {
  messages,
  isConnected,
  currentFile,
  isLoading,
  error,
  startWatching,
} = useJsonlWatcher();

const {
  entries,
  isLoading: isScanning,
  scanLeaderboard,
} = useLeaderboardScanner();

const sampleFiles = [
  { name: 'Sample (Demo)', path: '/sample.jsonl' },
];

onMounted(() => {
  scanLeaderboard();
});

function handleSelect(path: string) {
  startWatching(path);
}

function handleRefresh() {
  scanLeaderboard();
}

function getDisplayPath(path: string): string {
  if (path.startsWith('/leaderboard/')) {
    return path.replace('/leaderboard/', '').replace('/agent_log.jsonl', '');
  }
  return path;
}
</script>

<template>
  <div id="app" class="w-full h-full">
    <header class="app-header h-[10%]">
      <h1>AGENT MONITOR</h1>
      <p class="subtitle">Real-time Monitor for Vector-bench</p>
    </header>
    
    <div class="app-body h-[85%]">
      <aside class="sidebar">
        <JsonlSelector
          :entries="entries"
          :is-loading="isScanning"
          :selected-path="currentFile"
          @select="handleSelect"
          @refresh="handleRefresh"
        />
        
        <div class="sample-files">
          <div class="sample-header">Sample Files</div>
          <div 
            v-for="file in sampleFiles" 
            :key="file.path"
            class="file-item"
            :class="{ selected: currentFile === file.path }"
            @click="handleSelect(file.path)"
          >
            {{ file.name }}
          </div>
        </div>
      </aside>
      
      <main class="main-content">
        <ChatMessageList :messages="messages" />
      </main>
    </div>
    
    <footer class="status-bar h-[5%]">
      <span class="current-file">
        {{ currentFile ? getDisplayPath(currentFile) : 'No file selected' }}
      </span>
      <span class="connection-status" :class="{ connected: isConnected }">
        {{ isConnected ? '● Connected' : '○ Disconnected' }}
      </span>
      <span v-if="isLoading" class="loading">Loading...</span>
      <span v-if="error" class="error">{{ error }}</span>
    </footer>
  </div>
</template>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body {
  height: 100%;
  overflow: hidden;
}

#app {
  height: 100%;
}
</style>

<style scoped>
.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.app-header {
  background: #252526;
  padding: 16px 24px;
  border-bottom: 1px solid #333;
}

.app-header h1 {
  font-size: 20px;
  font-weight: 600;
  color: #fff;
  margin-bottom: 4px;
}

.subtitle {
  font-size: 12px;
  color: #888;
}

.app-body {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.sidebar {
  width: 260px;
  display: flex;
  flex-direction: column;
  background: #1e1e1e;
  border-right: 1px solid #333;
}

.sample-files {
  margin-top: auto;
  border-top: 1px solid #333;
  padding: 8px 0;
}

.sample-header {
  padding: 6px 16px;
  font-size: 11px;
  font-weight: 600;
  color: #666;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.file-item {
  padding: 8px 16px 8px 24px;
  cursor: pointer;
  font-size: 13px;
  color: #aaa;
  transition: all 0.15s;
}

.file-item:hover {
  background: #2a2d2e;
  color: #fff;
}

.file-item.selected {
  background: #094771;
  color: #fff;
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 8px 16px;
  background: #007acc;
  color: #fff;
  font-size: 12px;
}

.current-file {
  flex: 1;
  opacity: 0.9;
}

.connection-status {
  padding: 2px 8px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 4px;
}

.connection-status.connected {
  background: rgba(46, 204, 113, 0.3);
}

.loading {
  color: #ffeb3b;
}

.error {
  color: #ff5252;
}
</style>
