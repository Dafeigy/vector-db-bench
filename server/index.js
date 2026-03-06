import express from 'express';
import cors from 'cors';
import { watch } from 'fs';
import { statSync } from 'fs';
import { readFile } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = 3005;

app.use(cors());

const jsonlFile = join(__dirname, '..', 'example.jsonl');

let watchers = new Set();
let lastMtime = 0;
let lastSize = 0;

function notifyClients() {
  console.log('Notifying clients, count:', watchers.size)
  const message = JSON.stringify({ event: 'update' });
  watchers.forEach(client => {
    client.write(`data: ${message}\n\n`);
    console.log('Sent to client');
  });
}

app.get('/api/lines', async (req, res) => {
  try {
    const content = await readFile(jsonlFile, 'utf-8');
    const lines = content.split('\n').filter(line => line.trim());
    res.json({ lines, total: lines.length });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/events', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.flushHeaders();

  console.log('New SSE client connected');
  watchers.add(res);

  const stats = statSync(jsonlFile);
  lastMtime = stats.mtimeMs;
  lastSize = stats.size;

  req.on('close', () => {
    watchers.delete(res);
  });
});

let fsWatcher = watch(jsonlFile, (eventType) => {
  if (eventType === 'change') {
    console.log('File changed, notifying clients');
    notifyClients();
  }
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
