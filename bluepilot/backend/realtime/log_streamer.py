#!/usr/bin/env python3
"""
BluePilot Log Streamer

Streams manager logs in real-time via WebSocket
"""

import logging
from datetime import datetime
import threading
from typing import Optional

from bluepilot.backend.logs import parse_manager_log_line, read_tmux_logs

logger = logging.getLogger(__name__)


class LogStreamer:
    """Streams manager logs to WebSocket clients"""

    def __init__(self, websocket_broadcaster):
        """
        Initialize log streamer

        Args:
            websocket_broadcaster: WebSocketBroadcaster instance
        """
        self.broadcaster = websocket_broadcaster
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sock = None
        self._tmux_last_line = None
        self._tmux_target = None
        self._use_tmux = False

    def start(self):
        """Start streaming logs"""
        with self._lock:
            if self.running:
                logger.debug("Log streamer already running")
                return False

            try:
                import cereal.messaging as messaging

                self._sock = messaging.sub_sock('logMessage', timeout=1000, conflate=True)
                self._use_tmux = False
            except Exception as exc:
                logger.error("Unable to access logMessage stream: %s", exc)
                self._use_tmux = True

            self.running = True
            self._stop_event.clear()

            target = self._read_tmux if self._use_tmux else self._read_logs
            self.thread = threading.Thread(target=target, daemon=True)
            self.thread.start()

            logger.info("Log streamer started")
            self._broadcast_status('started')
            return True

    def stop(self):
        """Stop streaming logs"""
        with self._lock:
            if not self.running:
                logger.debug("Log streamer not running")
                return False

            self.running = False
            self._stop_event.set()

            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                finally:
                    self._sock = None

            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1.5)
                self.thread = None

            logger.info("Log streamer stopped")
            self._broadcast_status('stopped')
            return True

    def is_running(self):
        """Check if streamer is running"""
        return self.running

    def _read_logs(self):
        """Read logs from messaging stream and broadcast"""
        try:
            import cereal.messaging as messaging
        except Exception as exc:
            logger.error("Unable to import messaging for log stream: %s", exc)
            self._broadcast_status('error', 'messaging unavailable')
            self.running = False
            return

        sock = self._sock or messaging.sub_sock('logMessage', timeout=1000, conflate=True)

        try:
            while self.running:
                msg = messaging.recv_one_or_none(sock)
                if msg is None:
                    if self._stop_event.wait(timeout=0.1):
                        break
                    continue

                if msg.which() != 'logMessage':
                    continue

                formatted = parse_manager_log_line(msg.logMessage)
                if formatted:
                    from bluepilot.backend.realtime.websocket import WebSocketEvent
                    self.broadcaster.broadcast(WebSocketEvent.LOG_LINE, {
                        'line': formatted
                    })

        except Exception as exc:
            logger.error("Error reading log stream: %s", exc)
            self._broadcast_status('error', 'Log stream error')
        finally:
            self.running = False
            self._stop_event.set()
            if self._sock and sock is not self._sock:
                try:
                    sock.close()
                except Exception:
                    pass
            self._sock = None

    def _read_tmux(self):
        """Read logs from tmux output and broadcast."""
        from bluepilot.backend.realtime.websocket import WebSocketEvent

        poll_interval = 1.0
        while self.running:
            ok, output = read_tmux_logs(max_lines=2000, target=self._tmux_target, with_timestamps=False)
            if ok and output:
                lines = [line for line in output.splitlines() if line.strip()]
                new_lines = []
                if self._tmux_last_line and self._tmux_last_line in lines:
                    idx = lines.index(self._tmux_last_line)
                    new_lines = lines[idx + 1:]
                else:
                    new_lines = lines[-200:]

                for line in new_lines:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    formatted = f"\x1b[2m{timestamp}\x1b[0m {line}".rstrip()
                    self.broadcaster.broadcast(WebSocketEvent.LOG_LINE, {'line': formatted})

                if lines:
                    self._tmux_last_line = lines[-1]
            else:
                if ok is False and output:
                    self._broadcast_status('error', output)

            if self._stop_event.wait(timeout=poll_interval):
                break

    def _broadcast_status(self, status, message=None):
        """Broadcast stream status change"""
        from bluepilot.backend.realtime.websocket import WebSocketEvent
        data = {'status': status}
        if message:
            data['message'] = message
        self.broadcaster.broadcast(WebSocketEvent.LOG_STREAM_STATUS, data)


# Global log streamer instance
_log_streamer: Optional[LogStreamer] = None


def get_log_streamer(websocket_broadcaster=None):
    """Get or create global log streamer instance"""
    global _log_streamer

    if _log_streamer is None and websocket_broadcaster is not None:
        _log_streamer = LogStreamer(websocket_broadcaster)

    return _log_streamer
