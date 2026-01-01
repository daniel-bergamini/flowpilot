import { useEffect, useState, useRef } from 'react'
import { Button, ToggleSwitch } from '@/components/common'

interface LogResponse {
  success: boolean
  output?: string
  error?: string
}

interface WebSocketMessage {
  type: string
  data?: {
    line?: string
    status?: string
    message?: string
  }
}

export function DiagnosticsTmux() {
  const [logLines, setLogLines] = useState<string[]>([])
  const [isPaused, setIsPaused] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [isConnecting, setIsConnecting] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const [streamStatus, setStreamStatus] = useState<string>('stopped')

  const logContainerRef = useRef<HTMLPreElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const seenLinesRef = useRef<Set<string>>(new Set())
  const lastLogAtRef = useRef<number>(0)
  const lastPollAtRef = useRef<number>(0)

  const hasTimestamp = (line: string) => {
    return /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}/.test(line)
  }

  const formatTimestamp = () => {
    const now = new Date()
    const pad = (value: number, size = 2) => value.toString().padStart(size, '0')
    return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} `
      + `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}.${pad(now.getMilliseconds(), 3)}`
  }

  const formatLine = (line: string) => {
    if (!line || hasTimestamp(line)) {
      return line
    }
    return `${formatTimestamp()} ${line}`
  }

  const appendLines = (lines: string[]) => {
    if (!lines.length) return
    setLogLines(prev => {
      const next = [...prev]
      let appended = 0
      for (const raw of lines) {
        const line = raw.trim()
        if (!line) continue
        if (seenLinesRef.current.has(line)) continue
        seenLinesRef.current.add(line)
        next.push(formatLine(line))
        appended += 1
      }
      if (appended > 0) {
        lastLogAtRef.current = Date.now()
      }
      return next.slice(-2000)
    })
  }

  // Load initial logs
  useEffect(() => {
    const fetchInitialLogs = async () => {
      try {
        const response = await fetch('/api/manager-logs')
        if (response.ok) {
          const data: LogResponse = await response.json()
          if (data.success && data.output) {
            const lines = data.output.split('\n').filter(line => line.trim())
            seenLinesRef.current = new Set(lines)
            appendLines(lines)
          }
        }
      } catch (err) {
        console.error('Failed to fetch initial logs:', err)
      }
    }
    fetchInitialLogs()
  }, [])

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(`${protocol}//${window.location.hostname}:8089`)

      ws.onopen = () => {
        console.log('WebSocket connected')
        setError(null)
        setIsConnecting(false)
        wsRef.current = ws
      }

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)

          if (message.type === 'log_line' && message.data?.line && !isPaused) {
            const newLine = message.data.line
            if (seenLinesRef.current.has(newLine)) {
              return
            }
            seenLinesRef.current.add(newLine)
            lastLogAtRef.current = Date.now()
            setLogLines(prev => {
              const newLines = [...prev, formatLine(newLine)]
              return newLines.slice(-2000)
            })
          } else if (message.type === 'log_stream_status' && message.data?.status) {
            setStreamStatus(message.data.status)
            if (message.data.status === 'error' && message.data.message) {
              setError(message.data.message)
            }
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err)
        }
      }

      ws.onerror = (event) => {
        console.error('WebSocket error:', event)
        setError('WebSocket connection error')
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected')
        wsRef.current = null

        // Attempt to reconnect after 3 seconds
        if (!reconnectTimeoutRef.current) {
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectTimeoutRef.current = null
            setIsConnecting(true)
            connectWebSocket()
          }, 3000)
        }
      }

      return ws
    }

    const ws = connectWebSocket()

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
        reconnectTimeoutRef.current = null
      }
      ws.close()
    }
  }, [isPaused])

  // Poll logs when websocket is unavailable or idle
  useEffect(() => {
    const pollLogs = async () => {
      if (isPaused) return
      const wsReady = wsRef.current && wsRef.current.readyState === WebSocket.OPEN
      const idleMs = Date.now() - lastLogAtRef.current
      if (wsReady && idleMs < 4000) {
        return
      }
      const now = Date.now()
      if (now - lastPollAtRef.current < 4000) {
        return
      }
      lastPollAtRef.current = now
      try {
        const response = await fetch('/api/manager-logs')
        if (!response.ok) return
        const data: LogResponse = await response.json()
        if (data.success && data.output) {
          const lines = data.output.split('\n').filter(line => line.trim())
          appendLines(lines)
        }
      } catch (err) {
        console.error('Failed to poll logs:', err)
      }
    }

    const interval = setInterval(pollLogs, 4000)
    return () => clearInterval(interval)
  }, [isPaused])

  // Start streaming when component mounts
  useEffect(() => {
    const startStreaming = async () => {
      try {
        const response = await fetch('/api/manager-logs/stream/start', { method: 'POST' })
        if (response.ok) {
          const data = await response.json()
          if (data.success) {
            setStreamStatus('started')
          }
        }
      } catch (err) {
        console.error('Failed to start log streaming:', err)
      }
    }

    startStreaming()

    // Cleanup: stop streaming when unmounting
    return () => {
      fetch('/api/manager-logs/stream/stop', { method: 'POST' }).catch(() => {})
    }
  }, [])

  // Auto-scroll to bottom when new lines arrive
  useEffect(() => {
    if (autoScroll && logContainerRef.current && !isPaused) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logLines, autoScroll, isPaused])

  // Detect manual scrolling
  const handleScroll = () => {
    if (logContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = logContainerRef.current
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 50
      setAutoScroll(isAtBottom)
    }
  }

  const handlePauseToggle = async () => {
    if (!isPaused) {
      // Pausing - stop the stream
      try {
        const response = await fetch('/api/manager-logs/stream/stop', { method: 'POST' })
        if (response.ok) {
          setIsPaused(true)
          setStreamStatus('stopped')
        }
      } catch (err) {
        console.error('Failed to stop streaming:', err)
      }
    } else {
      // Resuming - start the stream
      try {
        const response = await fetch('/api/manager-logs/stream/start', { method: 'POST' })
        if (response.ok) {
          setIsPaused(false)
          setStreamStatus('started')
        }
      } catch (err) {
        console.error('Failed to start streaming:', err)
      }
    }
  }

  const handleClear = () => {
    setLogLines([])
    seenLinesRef.current.clear()
    lastLogAtRef.current = 0
  }

  const handleRefresh = async () => {
    setIsConnecting(true)
    setError(null)

    // Fetch fresh logs
    try {
      const response = await fetch('/api/manager-logs')
      if (response.ok) {
        const data: LogResponse = await response.json()
        if (data.success && data.output) {
          const lines = data.output.split('\n').filter(line => line.trim())
          appendLines(lines)
        }
      }
    } catch (err) {
      setError('Failed to fetch logs')
    } finally {
      setIsConnecting(false)
    }
  }

  // Filter lines based on search query
  const filteredLines = searchQuery
    ? logLines.filter(line => line.toLowerCase().includes(searchQuery.toLowerCase()))
    : logLines

  const displayText = filteredLines.join('\n')

  return (
    <>
      <div className="diagnostics-controls">
        <div className="search-container">
          <input
            type="text"
            className="search-input"
            placeholder="Search logs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {searchQuery && (
            <button
              className="search-clear"
              onClick={() => setSearchQuery('')}
              aria-label="Clear search"
            >
              ✕
            </button>
          )}
        </div>

        <div className="control-buttons">
          <Button
            variant={isPaused ? 'primary' : 'secondary'}
            size="small"
            onClick={handlePauseToggle}
            icon={<span aria-hidden="true">{isPaused ? '▶' : '⏸'}</span>}
          >
            {isPaused ? 'Resume' : 'Pause'}
          </Button>

          <Button
            variant="secondary"
            size="small"
            onClick={handleClear}
            icon={<span aria-hidden="true">🗑</span>}
          >
            Clear
          </Button>

          <Button
            variant="primary"
            size="small"
            onClick={handleRefresh}
            disabled={isConnecting}
            icon={<span aria-hidden="true">↻</span>}
          >
            Refresh
          </Button>

          <ToggleSwitch
            checked={autoScroll}
            onChange={setAutoScroll}
            label="Auto-scroll"
            size="compact"
            className="diagnostics-toggle"
          />
        </div>
      </div>

      <div className="diagnostics-content console-content">
        {error && (
          <div className="console-error">
            {error}
          </div>
        )}

        <div className="console-status">
          <span className="console-status-label">Status:</span>
          <span className={`console-status-value ${streamStatus === 'started' && !isPaused ? 'streaming' : 'paused'}`}>
            {isPaused ? 'Paused' : streamStatus === 'started' ? 'Streaming' : isConnecting ? 'Connecting...' : 'Stopped'}
          </span>
          {searchQuery && (
            <>
              <span className="console-status-separator">•</span>
              <span className="console-status-label">Showing:</span>
              <span className="console-status-value">
                {filteredLines.length} / {logLines.length} lines
              </span>
            </>
          )}
          <span className="console-status-separator">•</span>
          <span className="console-status-label">Total:</span>
          <span className="console-status-value">{logLines.length} lines</span>
        </div>

        <pre
          ref={logContainerRef}
          className="console-output"
          onScroll={handleScroll}
          aria-live={isPaused ? 'off' : 'polite'}
        >
          {displayText || (isConnecting ? 'Connecting to log stream...' : 'No logs available')}
        </pre>
      </div>
    </>
  )
}
