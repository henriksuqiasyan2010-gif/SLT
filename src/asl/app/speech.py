"""Non-blocking English speech with pyttsx3 (temporary; Phase 7 adds the Armenian audio files).

Each call runs in its own thread with its own engine (SAPI5 needs COM initialised per thread);
a call while the previous word is still being spoken is skipped instead of queued.
"""

import logging
import threading

log = logging.getLogger(__name__)


class Speaker:
    def __init__(self, on_error=None):
        """on_error(details: str) is called from the speech thread if speaking fails."""
        self.on_error = on_error
        self._busy = threading.Lock()

    def speak(self, text: str) -> bool:
        """Start speaking `text` in the background; False if still busy with the previous text."""
        if not text or not self._busy.acquire(blocking=False):
            return False
        threading.Thread(target=self._run, args=(text,), daemon=True, name="speech").start()
        return True

    def _run(self, text: str) -> None:
        try:
            try:
                import comtypes
                comtypes.CoInitialize()
            except (ImportError, OSError):
                pass
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:  # noqa: BLE001 - no voice must never break the app
            log.exception("speech failed")
            if self.on_error:
                self.on_error(f"{type(e).__name__}: {e}")
        finally:
            self._busy.release()
