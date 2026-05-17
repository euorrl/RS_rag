<script setup>
import { computed, nextTick, ref } from 'vue'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
const MAX_QUESTION_LENGTH = 500

const messages = ref([
  {
    id: crypto.randomUUID(),
    role: 'assistant',
    content: '你好，我是 RS RAG 助手。你可以直接输入遥感相关问题。',
  },
])
const question = ref('')
const sessionId = ref('')
const isStreaming = ref(false)
const errorMessage = ref('')
const chatBody = ref(null)
const abortController = ref(null)

const questionLength = computed(() => question.value.trim().length)
const canSend = computed(
  () =>
    questionLength.value > 0 &&
    questionLength.value <= MAX_QUESTION_LENGTH &&
    !isStreaming.value,
)
const lengthState = computed(() =>
  questionLength.value > MAX_QUESTION_LENGTH
    ? '超出限制'
    : `${questionLength.value}/${MAX_QUESTION_LENGTH}`,
)

function scrollToBottom() {
  nextTick(() => {
    if (!chatBody.value) {
      return
    }
    chatBody.value.scrollTop = chatBody.value.scrollHeight
  })
}

function createMessage(role, content = '') {
  return {
    id: crypto.randomUUID(),
    role,
    content,
  }
}

function parseSseBlock(block) {
  const event = {
    type: 'message',
    data: '',
  }

  for (const line of block.split('\n')) {
    if (line.startsWith('event:')) {
      event.type = line.slice(6).trim()
    }

    if (line.startsWith('data:')) {
      const data = line.startsWith('data: ') ? line.slice(6) : line.slice(5)
      event.data += `${data}\n`
    }
  }

  event.data = event.data.replace(/\n$/, '')
  return event
}

function applySseEvent(event, assistantMessage) {
  if (event.type === 'session') {
    const payload = JSON.parse(event.data)
    sessionId.value = payload.session_id || sessionId.value
    return
  }

  if (event.type === 'delta') {
    assistantMessage.content += event.data
    scrollToBottom()
    return
  }

  if (event.type === 'error') {
    errorMessage.value = event.data || '后端生成答案失败'
  }
}

async function readStream(response, assistantMessage) {
  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      break
    }

    buffer += decoder.decode(value, { stream: true })
    const blocks = buffer.split('\n\n')
    buffer = blocks.pop() || ''

    for (const block of blocks) {
      if (!block.trim()) {
        continue
      }
      applySseEvent(parseSseBlock(block), assistantMessage)
    }
  }

  if (buffer.trim()) {
    applySseEvent(parseSseBlock(buffer), assistantMessage)
  }
}

async function sendQuestion() {
  const text = question.value.trim()
  if (!canSend.value) {
    return
  }

  errorMessage.value = ''
  question.value = ''
  isStreaming.value = true
  abortController.value = new AbortController()

  const userMessage = createMessage('user', text)
  const assistantMessage = createMessage('assistant')
  messages.value.push(userMessage, assistantMessage)
  scrollToBottom()

  try {
    const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: text,
        session_id: sessionId.value || null,
      }),
      signal: abortController.value.signal,
    })

    if (!response.ok || !response.body) {
      throw new Error(`请求失败：${response.status}`)
    }

    await readStream(response, assistantMessage)
  } catch (error) {
    if (error.name !== 'AbortError') {
      errorMessage.value = error.message || '请求后端失败'
      assistantMessage.content = assistantMessage.content || '请求失败，请稍后重试。'
    }
  } finally {
    isStreaming.value = false
    abortController.value = null
    scrollToBottom()
  }
}

function stopStreaming() {
  abortController.value?.abort()
  isStreaming.value = false
}

async function startNewChat() {
  if (isStreaming.value) {
    stopStreaming()
  }

  const oldSessionId = sessionId.value
  sessionId.value = ''
  messages.value = [
    {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '新的对话已开始。你可以继续提问。',
    },
  ]
  errorMessage.value = ''

  if (!oldSessionId) {
    return
  }

  try {
    await fetch(`${API_BASE_URL}/api/chat/session/${oldSessionId}`, {
      method: 'DELETE',
    })
  } catch {
    errorMessage.value = '本地对话已清空，但后端会话清理失败。'
  }
}

function handleKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    sendQuestion()
  }
}
</script>

<template>
  <main class="app-shell">
    <aside class="side-panel">
      <div>
        <p class="eyebrow">RS RAG</p>
        <h1>遥感知识问答</h1>
        <p class="summary">
          连接后端 RAG 服务，按会话保存上下文，并以流式方式返回答案。
        </p>
      </div>

      <div class="status-list">
        <div class="status-item">
          <span class="status-dot"></span>
          <span>后端接口：{{ API_BASE_URL }}</span>
        </div>
        <div class="status-item">
          <span class="status-dot"></span>
          <span>会话：{{ sessionId || '未创建' }}</span>
        </div>
      </div>
    </aside>

    <section class="chat-panel" aria-label="遥感问答聊天窗口">
      <header class="chat-header">
        <div>
          <p class="eyebrow">Chat</p>
          <h2>问答窗口</h2>
        </div>
        <button class="secondary-button" type="button" @click="startNewChat">
          新对话
        </button>
      </header>

      <div ref="chatBody" class="chat-body">
        <article
          v-for="message in messages"
          :key="message.id"
          class="message-row"
          :class="`message-row-${message.role}`"
        >
          <div class="message-bubble">
            <span class="message-role">
              {{ message.role === 'user' ? '你' : '助手' }}
            </span>
            <p v-if="message.content">{{ message.content }}</p>
            <p v-else class="thinking-text">正在生成答案...</p>
          </div>
        </article>
      </div>

      <footer class="composer">
        <p v-if="errorMessage" class="error-message">{{ errorMessage }}</p>
        <div class="input-row">
          <textarea
            v-model="question"
            placeholder="输入你的问题，例如：什么是遥感？"
            rows="3"
            :maxlength="MAX_QUESTION_LENGTH + 50"
            :disabled="isStreaming"
            @keydown="handleKeydown"
          ></textarea>
          <button
            v-if="isStreaming"
            class="send-button stop-button"
            type="button"
            @click="stopStreaming"
          >
            停止
          </button>
          <button
            v-else
            class="send-button"
            type="button"
            :disabled="!canSend"
            @click="sendQuestion"
          >
            发送
          </button>
        </div>
        <div class="composer-meta">
          <span>Enter 发送，Shift + Enter 换行</span>
          <span :class="{ danger: questionLength > MAX_QUESTION_LENGTH }">
            {{ lengthState }}
          </span>
        </div>
      </footer>
    </section>
  </main>
</template>
