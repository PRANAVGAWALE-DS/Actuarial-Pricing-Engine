from __future__ import annotations

# ---------------------------------------------------------------------------
# stdlib imports
# ---------------------------------------------------------------------------
import atexit
import concurrent.futures
import copy
import gc
import hashlib
import html as html_module
import json
import logging
import math
import os
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import wraps
from io import BytesIO
from typing import Any

# ---------------------------------------------------------------------------
# third-party imports (required)
# ---------------------------------------------------------------------------
import pandas as pd
import plotly.express as px
import plotly.io as pio
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# third-party imports (optional)
# ---------------------------------------------------------------------------
try:
    import psutil  # noqa: F401

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# =============================================================================
# LOGGING SETUP
# =============================================================================

_root_logger = logging.getLogger()
if not _root_logger.hasHandlers():
    class UTF8SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                stream = self.stream
                terminator = self.terminator
                try:
                    stream.write(msg + terminator)
                except UnicodeEncodeError:
                    safe_msg = msg.encode("ascii", errors="backslashreplace").decode(
                        "ascii"
                    )
                    stream.write(safe_msg + terminator)
                stream.flush()
            except Exception:
                self.handleError(record)

    _handler = UTF8SafeStreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    _root_logger.addHandler(_handler)
    _root_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

for _noisy_logger in ("streamlit", "streamlit.deprecation_util"):
    logging.getLogger(_noisy_logger).setLevel(logging.ERROR)

pio.templates.default = "plotly"

load_dotenv(override=False)


@st.cache_resource
def _log_optional_deps_once() -> None:
    """Log optional dependency status exactly ONCE per server lifetime."""
    if PSUTIL_AVAILABLE:
        logger.info("psutil available - memory monitoring enabled")
    else:
        logger.warning("psutil not available - memory monitoring disabled")


# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

APP_VERSION = "3.5.12"

BMI_UNDERWEIGHT = 18.5
BMI_NORMAL = 25.0
BMI_OVERWEIGHT = 30.0

RISK_LOW_THRESHOLD = 30
RISK_MEDIUM_THRESHOLD = 60

MAX_FILE_SIZE_MB = 5
MAX_BATCH_ROWS = 50000

BATCH_PROCESS_SIZE = 100
BATCH_UPDATE_INTERVAL = 10
MAX_BATCH_TIME_SECONDS = 300
MAX_RESULTS_MEMORY = 50000
BYTES_PER_ROW_ESTIMATE = 1200

RATE_LIMIT_SECONDS = 0.5

BMI_METHOD_DIRECT = "Enter BMI directly"
BMI_METHOD_CALCULATE = "Calculate from height/weight"

MAX_HISTORY_SIZE = 10
HEALTH_CHECK_TTL = 60
COMPARISON_LIMIT = 5
MAX_EXPORT_RECORDS = 50
EXPORT_WARNING_THRESHOLD = 25

VALID_REGIONS = ["northeast", "northwest", "southeast", "southwest"]
VALID_SEX = ["male", "female"]
VALID_SMOKER = ["no", "yes"]

MIN_TIMESTAMP = 0
MAX_TIMESTAMP = 253402300799  # Year 9999

# =============================================================================
# ENVIRONMENT VARIABLE HELPERS
# =============================================================================


def get_env_int(key: str, default: int, min_val: int = 1) -> int:
    """Safely parse integer environment variables with validation."""
    try:
        value = int(os.getenv(key, str(default)))
        return max(min_val, value)
    except (ValueError, TypeError):
        logger.warning(f"Invalid {key}, using default {default}")
        return default


def _sanitize_api_key(raw: str | None) -> str | None:
    """Sanitise an API key loaded from environment / .env file."""
    if not raw:
        return None

    # Strip surrounding whitespace only, then strip enclosing quote characters.
    # Do NOT split on internal whitespace — tokens such as OAuth bearer values
    # may legitimately contain spaces; key.split()[0] would silently truncate them.
    key = raw.strip().strip("\"'")

    if key != raw.strip():
        logger.warning(
            "API_KEY had surrounding quote characters that were stripped. "
            "If the key contains intentional leading/trailing quotes, "
            "check your .env file."
        )

    # Warn if the original raw value had trailing content after stripping the key
    # (e.g. an inline .env comment separated by whitespace).
    raw_stripped = raw.strip()
    if " " in raw_stripped or "\t" in raw_stripped:
        logger.warning(
            "API_KEY appears to contain whitespace. "
            f"Using stripped value ({len(key)} chars). "
            "If your key genuinely contains spaces, this is expected. "
            "Otherwise check your .env file for inline comments: "
            "API_KEY=\"your-key\"  # comment"
        )

    try:
        key.encode("ascii")
    except UnicodeEncodeError as _ue:
        logger.error(
            f"API_KEY contains non-ASCII char at position {_ue.start} "
            f"({repr(key[_ue.start])!s}). "
            "HTTP Authorization headers must be 7-bit ASCII (RFC 7230). "
            "Remove em-dashes or trailing comments from the API_KEY line "
            "in your .env file. Correct format: API_KEY=your-actual-key"
        )
        return None

    return key if key else None


API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
API_TIMEOUT = get_env_int("API_TIMEOUT", 20)
HEALTH_TIMEOUT = get_env_int("HEALTH_TIMEOUT", 2)
MAX_RETRIES = get_env_int("MAX_RETRIES", 1, min_val=0)
API_KEY = _sanitize_api_key(os.getenv("API_KEY"))
METRICS_TOKEN: str | None = os.getenv("METRICS_TOKEN")
PREDICTION_MIN = float(os.getenv("PREDICTION_MIN", "100"))
PREDICTION_MAX = float(os.getenv("PREDICTION_MAX", "500000"))

KNOWN_MODELS: frozenset[str] = frozenset(
    {
        "xgboost_median",
        "xgboost_high_value_specialist",
        "lightgbm",
        "random_forest",
        "hybrid",
        "hybrid_xgboost_median_v6.3.1",
        "unknown",
    }
)

_KNOWN_MODEL_PREFIXES: tuple[str, ...] = (
    "xgboost_",
    "lightgbm_",
    "random_forest_",
    "hybrid_",
    "ensemble_",
)


def _is_known_model(model_used: str) -> bool:
    """Return True if model_used is a recognised backend model identifier."""
    if model_used in KNOWN_MODELS:
        return True
    return any(model_used.startswith(prefix) for prefix in _KNOWN_MODEL_PREFIXES)


P99_ALERT_THRESHOLD_MS: float = float(os.getenv("P99_ALERT_THRESHOLD_MS", "20000"))

MAX_BATCH_CSV_BYTES = 5 * 1024 * 1024  # 5 MB

# =============================================================================
# CHART CONFIGURATION
# =============================================================================

# Base template — static values only.  Theme-sensitive keys (font.color,
# hoverlabel) are injected at render time by _chart_layout().
# BUG-3 FIX: added default margin so axis labels and vline annotations are
# never clipped.  margin.t=60 gives heading room; margin.b=70 prevents the
# x-axis label from being cut off; margin.r=20 avoids legend truncation.
CHART_LAYOUT_TEMPLATE = {
    "plot_bgcolor": "rgba(0,0,0,0)",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "font": dict(family="Arial, sans-serif", size=12),
    "title_font_size": 18,
    "hovermode": "x unified",
    "margin": dict(t=60, b=70, l=60, r=20),
}

CHART_AXIS_CONFIG = {
    "showgrid": True,
    "gridwidth": 1,
    "gridcolor": "rgba(128,128,128,0.2)",
}


# ---------------------------------------------------------------------------
# THEME-4 / THEME-5: Runtime theme helpers
# ---------------------------------------------------------------------------

def _get_theme() -> str:
    """Best-effort active-theme detection from Python.

    NOTE: st.get_option("theme.base") returns the config.toml base ("light")
    regardless of the hamburger menu selection.  There is no public Streamlit
    Python API that returns the active user-selected theme.  This function
    returns "light" unconditionally; it exists as a hook for future upgrades
    (e.g. session_state bridge via custom component).  All background/sidebar
    theming is handled client-side by _inject_theme_detector() + html.st-dark.
    """
    return "light"


def _chart_layout(**extra: Any) -> dict:
    """Return a Plotly layout dict merging CHART_LAYOUT_TEMPLATE.

    THEME-4 FIX: hoverlabel bgcolor was hardcoded "white" — tooltip rendered
    as a white box on dark chart backgrounds.  Replaced with a dark semi-
    transparent rgba value that reads correctly on both light and dark themes
    without requiring Python-side theme detection.

    SHALLOW-MERGE FIX (Point 4): ``dict(CHART_LAYOUT_TEMPLATE)`` produces a
    shallow copy, so ``extra=dict(font=dict(size=14))`` previously wiped the
    entire base ``font`` dict (losing ``family="Arial, sans-serif"``).
    The layout is now deep-copied and nested ``extra`` dicts are recursively
    merged rather than replaced wholesale.

    DARK-MODE FIX (v3.5.11): font.color was hardcoded to the light-palette
    value "#2d3748" regardless of the dark_mode toggle, making axis labels,
    titles, and annotations invisible on the dark (#0e1117) app background.
    Now reads st.session_state.dark_mode and switches font colour accordingly.

    Args:
        **extra: Additional layout kwargs merged over the base template.
                 Nested dicts (e.g. ``font=dict(size=14)``) are merged
                 key-by-key, not replaced.
    """
    is_dark: bool = st.session_state.get("dark_mode", False)
    layout: dict[str, Any] = copy.deepcopy(CHART_LAYOUT_TEMPLATE)
    # DARK-MODE FIX: choose font colour based on active palette.
    layout["font"]["color"] = "#e2e8f0" if is_dark else "#2d3748"
    layout["hoverlabel"] = dict(
        bgcolor="rgba(30, 33, 48, 0.92)",
        font_size=13,
        font_color="#f0f0f5",
        bordercolor="rgba(255, 255, 255, 0.12)",
    )
    for key, val in extra.items():
        if key in layout and isinstance(layout[key], dict) and isinstance(val, dict):
            # Deep-merge: overlay caller keys onto the existing nested dict
            # instead of replacing it, preserving keys the caller omits.
            layout[key] = {**layout[key], **val}
        else:
            layout[key] = val
    return layout


def _primary_color() -> str:
    """Chart primary line/marker colour — switches with dark_mode toggle."""
    return "#7c8aed" if st.session_state.get("dark_mode", False) else "#5a67d8"


def _primary_dark_color() -> str:
    """Chart primary-dark marker colour — switches with dark_mode toggle."""
    return "#a3b0f5" if st.session_state.get("dark_mode", False) else "#4c51bf"


def _marker_border_color() -> str:
    """Marker/line border colour.

    Dark mode: near-black border so coloured markers pop on the dark background.
    Light mode: white border so coloured markers stand out on light bars.
    """
    return "#0e1117" if st.session_state.get("dark_mode", False) else "white"


def _marker_line_color() -> str:
    """Bar/pie marker outline.

    Dark mode: subtle white-ish semi-transparent outline legible on dark bars.
    Light mode: dark navy legible on light bars.
    """
    return "rgba(255,255,255,0.30)" if st.session_state.get("dark_mode", False) else "rgb(8,48,107)"


def _vline_threshold_color() -> str:
    """Threshold vline colour for What-If charts — readable on both palettes."""
    return "rgba(180,180,180,0.85)" if st.session_state.get("dark_mode", False) else "gray"


def _vline_current_color() -> str:
    """'Current value' vline colour for What-If charts."""
    return "#7c8aed" if st.session_state.get("dark_mode", False) else "blue"


# =============================================================================
# UTILITY CONVERSION FUNCTIONS
# =============================================================================


def _safe_convert(val: Any, converter: callable, default: Any) -> Any:
    """Safely convert values with NaN/None/empty string handling."""
    try:
        if pd.isna(val) or val is None or val == "":
            return default
        return converter(val)
    except (ValueError, TypeError, AttributeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    return _safe_convert(val, lambda x: int(float(x)), default)


def _safe_float(val: Any, default: float = 0.0) -> float:
    return _safe_convert(val, float, default)


def _safe_str(val: Any, default: str = "") -> str:
    return _safe_convert(val, lambda x: str(x).strip().lower(), default)


def safe_html(value: Any) -> str:
    """Escape a value before embedding it inside an unsafe_allow_html block."""
    return html_module.escape(str(value))


def _mask_url_credentials(url: str | None) -> str:
    """Return URL with password replaced by '***'. Safe to display in UI."""
    if not url:
        return "not set"
    try:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(url)
        if parsed.password:
            username = parsed.username or ""
            # urlparse strips brackets from IPv6 addresses (e.g. [::1] → ::1).
            # Re-wrap with brackets when a colon is present so the reconstructed
            # netloc is a valid URL (http://user:***@::1:8080 is ambiguous;
            # http://user:***@[::1]:8080 is correct RFC-2732 syntax).
            raw_host = parsed.hostname or ""
            host = f"[{raw_host}]" if ":" in raw_host else raw_host
            port = f":{parsed.port}" if parsed.port else ""
            if username:
                userinfo = f"{username}:***"
            else:
                # Password-only auth (http://:secret@host): preserve the colon
                # so the auth block remains visible as `:***@host`.
                userinfo = ":***"
            netloc = f"{userinfo}@{host}{port}"
            masked = urlunparse(parsed._replace(netloc=netloc))
            return masked
        return url
    except Exception:
        return "[URL parsing error]"


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class FormData:
    """User input form data with validation."""

    age: int = 30
    sex: str = "male"
    bmi: float = 25.0
    children: int = 0
    smoker: str = "no"
    region: str = "northeast"

    def __post_init__(self) -> None:
        try:
            age_val = int(self.age)
            if not (18 <= age_val <= 100):
                raise ValueError("Age must be between 18 and 100")
            self.age = age_val
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid age: {str(e)}") from e

        try:
            bmi_val = float(self.bmi)
            if not (10.0 <= bmi_val <= 60.0):
                raise ValueError("BMI must be between 10.0 and 60.0")
            self.bmi = round(bmi_val, 1)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid BMI: {str(e)}") from e

        try:
            children_val = int(self.children)
            if not (0 <= children_val <= 10):
                raise ValueError("Children must be between 0 and 10")
            self.children = children_val
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid children count: {str(e)}") from e

        sex_val = str(self.sex).strip().lower()
        if sex_val not in VALID_SEX:
            raise ValueError(f"Sex must be one of: {', '.join(VALID_SEX)}")
        self.sex = sex_val

        smoker_val = str(self.smoker).strip().lower()
        if smoker_val not in VALID_SMOKER:
            raise ValueError(f"Smoker must be one of: {', '.join(VALID_SMOKER)}")
        self.smoker = smoker_val

        region_val = str(self.region).strip().lower()
        if region_val not in VALID_REGIONS:
            raise ValueError(f"Region must be one of: {', '.join(VALID_REGIONS)}")
        self.region = region_val

    def to_dict(self) -> dict[str, int | str | float]:
        return asdict(self)

    def to_api_payload(self) -> dict[str, int | str | float]:
        return asdict(self)


@dataclass
class Prediction:
    """Prediction result with metadata."""

    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str
    prediction: float
    model_used: str
    timestamp: datetime

    @classmethod
    def from_form_data(
        cls,
        form_data: FormData,
        prediction: float,
        model_used: str,
        timestamp: datetime | None = None,
    ) -> "Prediction":
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        return cls(
            age=form_data.age,
            sex=form_data.sex,
            bmi=form_data.bmi,
            children=form_data.children,
            smoker=form_data.smoker,
            region=form_data.region,
            prediction=prediction,
            model_used=model_used,
            timestamp=timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "age": self.age,
            "sex": self.sex,
            "bmi": self.bmi,
            "children": self.children,
            "smoker": self.smoker,
            "region": self.region,
            "prediction": self.prediction,
            "model_used": self.model_used,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================


class RateLimitError(Exception):
    """Raised when the API returns HTTP 429 Too Many Requests.

    429 is a capacity signal — the server is healthy but temporarily
    overloaded.  Must NOT increment the CircuitBreaker failure counter.
    """


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


class CircuitBreaker:
    """Circuit breaker pattern for API resilience.

    All state mutations are protected by a ``threading.Lock`` so that
    concurrent Streamlit sessions cannot race on ``failure_count``,
    ``state``, or ``last_failure_time``.

    Timeout recovery uses ``time.monotonic()`` (Issue 6) so NTP steps,
    DST transitions, or leap seconds cannot permanently stuck the breaker.
    ``last_failure_time`` (wall clock) is retained solely for display.

    Thundering-herd prevention (Issue 2): when the first thread transitions
    ``"open"`` → ``"half_open"`` it is the sole probe thread.  Every
    subsequent thread that finds the state already ``"half_open"`` is
    rejected exactly like ``"open"`` — it does not get a free pass through.
    """

    def __init__(self, failure_threshold: int = 10, timeout: int = 60) -> None:
        self._lock = threading.Lock()
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        # Monotonic timestamp used for timeout comparison — immune to clock
        # adjustments (time.time() could step backward and strand the breaker).
        self._last_failure_mono: float | None = None
        # Wall-clock timestamp kept only for human-readable display in get_state().
        self.last_failure_time: float | None = None
        self.state = "closed"

    def call(self, func, *args, **kwargs):
        with self._lock:
            current_state = self.state
            if current_state == "open":
                if (
                    self._last_failure_mono is not None
                    and time.monotonic() - self._last_failure_mono > self.timeout
                ):
                    # This thread wins the probe slot; all others keep seeing
                    # "half_open" and are rejected below (thundering-herd fix).
                    self.state = "half_open"
                    current_state = "half_open"
                else:
                    raise ConnectionError(
                        "Service temporarily unavailable (circuit breaker open)"
                    )
            elif current_state == "half_open":
                # A probe is already in flight from the thread that transitioned
                # open→half_open.  Reject all other threads to prevent a burst
                # of concurrent probe requests hitting the recovering backend.
                raise ConnectionError(
                    "Service temporarily unavailable (circuit breaker probing)"
                )

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except RateLimitError:
            # Capacity signal — server is healthy; do not count against CB.
            raise
        except requests.exceptions.HTTPError as exc:
            # 4xx errors are client-side mistakes, not server failures.
            # Counting them would let bad form data trip the circuit breaker.
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and 400 <= status < 500:
                raise
            self.on_failure(current_state=current_state)
            raise
        except Exception:
            self.on_failure(current_state=current_state)
            raise

    def on_success(self) -> None:
        with self._lock:
            self.failure_count = 0
            self._last_failure_mono = None
            self.last_failure_time = None
            self.state = "closed"

    def on_failure(self, current_state: str = "closed") -> None:
        with self._lock:
            self.failure_count += 1
            self._last_failure_mono = time.monotonic()
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )

    def reset(self) -> None:
        """Atomically reset breaker to closed state (admin use only)."""
        with self._lock:
            self.failure_count = 0
            self._last_failure_mono = None
            self.last_failure_time = None
            self.state = "closed"

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            state_snapshot = self.state
            failure_count_snapshot = self.failure_count
            last_failure_time_snapshot = self.last_failure_time

        try:
            last_failure_iso = None
            if last_failure_time_snapshot:
                if MIN_TIMESTAMP <= last_failure_time_snapshot <= MAX_TIMESTAMP:
                    last_failure_iso = datetime.fromtimestamp(
                        last_failure_time_snapshot
                    ).isoformat()
                else:
                    last_failure_iso = "out_of_range"
        except (ValueError, OSError, OverflowError):
            last_failure_iso = "invalid"

        return {
            "state": state_snapshot,
            "failure_count": failure_count_snapshot,
            "last_failure": last_failure_iso,
        }


# =============================================================================
# API CLIENT
# =============================================================================


class APIClient:
    """HTTP client for insurance prediction API."""

    def __init__(
        self,
        base_url: str,
        timeout: int = API_TIMEOUT,
        health_timeout: int = HEALTH_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        api_key: str | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("API base URL must be provided")

        from ipaddress import AddressValueError, IPv4Address, IPv6Address
        from urllib.parse import urlparse as _urlparse

        _parsed = _urlparse(base_url)
        _hostname = _parsed.hostname or ""

        # Determine whether the URL is safe to use over plain HTTP.
        # Allowed without HTTPS:
        #   • loopback addresses: localhost, 127.0.0.1, ::1
        #   • RFC-1918 private IPv4: 10.x, 172.16–31.x, 192.168.x
        #   • link-local: 169.254.x.x
        #   • non-dotted hostnames (Docker Compose service names,
        #     Kubernetes cluster-DNS, internal VPC aliases, e.g. "backend-api")
        def _is_private_or_local(hostname: str) -> bool:
            _loopback = {"localhost", "127.0.0.1", "::1"}
            if hostname in _loopback:
                return True
            # Non-dotted single-label hostnames are internal service names.
            if hostname and "." not in hostname:
                return True
            # Attempt IP address classification.
            try:
                addr = IPv4Address(hostname)
                return addr.is_private or addr.is_loopback or addr.is_link_local
            except AddressValueError:
                pass
            try:
                addr = IPv6Address(hostname)
                return addr.is_private or addr.is_loopback or addr.is_link_local
            except AddressValueError:
                pass
            return False

        _is_internal = _is_private_or_local(_hostname)
        if not (base_url.startswith("https://") or _is_internal):
            raise ValueError(
                f"API URL must use HTTPS or resolve to a private/internal network "
                f"address (localhost, RFC-1918 IP, or a single-label Docker/k8s "
                f"service name). Got: {base_url!r}"
            )

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.health_timeout = health_timeout
        self.api_key = api_key
        self.circuit_breaker = CircuitBreaker()

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            read=0,
            # 429 is included so urllib3 backs off on rate-limit responses
            # instead of propagating them immediately (Issue 7).
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "HEAD", "POST"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20,
            pool_block=False,
        )

        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        if api_key:
            try:
                api_key.encode("latin-1")
            except UnicodeEncodeError as _e:
                _char = repr(api_key[_e.start]) if _e.start < len(api_key) else "?"
                raise ValueError(
                    f"API_KEY contains a non-latin-1 character ({_char}) at "
                    f"position {_e.start}. HTTP headers must be ASCII. "
                    "Check your .env file: remove em-dashes or trailing "
                    "inline comments from the API_KEY line."
                ) from _e

        logger.info(f"API Client initialized: {self.base_url} (auth: {bool(api_key)})")

    def _safe_json(self, resp: requests.Response) -> dict[str, Any]:
        try:
            return resp.json()
        except ValueError as e:
            logger.warning(f"JSON parsing failed: {e}")
            return {"detail": resp.text[:200]}

    def health_check(self) -> dict[str, Any]:
        try:
            resp = self.session.get(
                f"{self.base_url}/health", timeout=self.health_timeout
            )
            # Parse the body first so non-2xx responses still surface the
            # backend's detail payload rather than losing it to raise_for_status.
            body = self._safe_json(resp)
            if not resp.ok:
                logger.error(
                    f"Health check HTTP {resp.status_code}: "
                    f"{body.get('detail', resp.text[:200])}"
                )
                resp.raise_for_status()
            return body
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check error: {type(e).__name__}")
            raise

    def predict_single(self, form_data: FormData) -> dict[str, Any]:
        def _make_request():
            payload = form_data.to_api_payload()
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            resp = self.session.post(
                f"{self.base_url}/api/v1/predict",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            if resp.status_code == 429:
                raise RateLimitError(
                    "API rate limit reached (429). Wait a moment and retry."
                )
            # Read body before raise_for_status so the backend's detailed
            # validation message is preserved rather than discarded.
            body = self._safe_json(resp)
            if not resp.ok:
                logger.warning(
                    f"predict_single HTTP {resp.status_code}: "
                    f"{body.get('detail', resp.text[:200])}"
                )
                resp.raise_for_status()
            return body

        try:
            return self.circuit_breaker.call(_make_request)
        except RateLimitError:
            raise
        except requests.exceptions.Timeout:
            raise TimeoutError("Request timed out")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 400:
                raise ValueError("Invalid request data")
            if status == 503:
                raise ConnectionError("Service unavailable")
            raise
        except requests.exceptions.RequestException as e:
            raise ConnectionError("Failed to connect to service") from e

    def predict_batch(self, records: list[dict]) -> dict[str, Any]:
        def _make_request():
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Timeout is bounded by MAX_BATCH_TIME_SECONDS so a pathologically
            # large record list cannot produce an arbitrarily long synchronous
            # block.  The caller (display_batch_predictions) is responsible for
            # chunking at BATCH_PROCESS_SIZE before calling this method.
            raw_timeout = len(records) * 0.05 + 10
            bounded_timeout = int(min(
                max(self.timeout, raw_timeout),
                MAX_BATCH_TIME_SECONDS,
            ))

            resp = self.session.post(
                f"{self.base_url}/api/v1/predict/batch",
                json={"records": records},
                headers=headers,
                timeout=bounded_timeout,
            )
            if resp.status_code == 429:
                raise RateLimitError(
                    "API rate limit reached (429) during batch. "
                    "Reduce batch size or retry later."
                )
            # Read body before raise_for_status so backend validation detail
            # is surfaced to the caller rather than silently discarded.
            body = self._safe_json(resp)
            if not resp.ok:
                logger.warning(
                    f"predict_batch HTTP {resp.status_code}: "
                    f"{body.get('detail', resp.text[:200])}"
                )
                resp.raise_for_status()
            return body

        try:
            return self.circuit_breaker.call(_make_request)
        except RateLimitError:
            raise
        except requests.exceptions.Timeout:
            raise TimeoutError("Batch request timed out")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 400:
                raise ValueError("Invalid batch request data")
            if status == 503:
                raise ConnectionError("Service unavailable")
            raise
        except requests.exceptions.RequestException as e:
            raise ConnectionError("Failed to connect to service") from e

    def get_metrics(self) -> dict[str, Any]:
        try:
            headers = {}
            # /api/v1/metrics requires METRICS_TOKEN, not the prediction API_KEY.
            # Fall back to api_key only when METRICS_TOKEN is unset (local/dev mode).
            _token = METRICS_TOKEN or self.api_key
            if _token:
                headers["Authorization"] = f"Bearer {_token}"
            resp = self.session.get(
                f"{self.base_url}/api/v1/metrics",
                headers=headers,
                timeout=self.health_timeout,
            )
            resp.raise_for_status()
            return self._safe_json(resp)
        except requests.exceptions.RequestException as e:
            logger.error(f"Metrics fetch failed: {type(e).__name__}")
            return {}

    def close(self) -> None:
        try:
            self.session.close()
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}", exc_info=True)
            if isinstance(e, (RuntimeError, OSError)):
                raise


# Module-level registry of every APIClient ever created by get_api_client().
# A single atexit hook (registered below, once at module load) closes all of
# them on process exit.  This prevents the atexit accumulation problem (Issue 7)
# where each cache-clear + re-create cycle appended a new atexit entry, keeping
# the old client's connection pool alive indefinitely via the atexit reference.
_managed_api_clients: list["APIClient"] = []


def _close_all_managed_clients() -> None:
    """Close every APIClient created during this process lifetime."""
    for _c in _managed_api_clients:
        try:
            _c.close()
        except Exception:
            pass
    _managed_api_clients.clear()


atexit.register(_close_all_managed_clients)


# =============================================================================
# CACHING & API CLIENT MANAGEMENT
# =============================================================================


@st.cache_resource
def get_api_client() -> APIClient:
    """Get or create cached API client.

    ``@st.cache_resource`` returns the same instance for the process lifetime.
    If the cache is cleared (e.g. Admin Panel → Clear Cache), this function
    re-executes and creates a *new* ``APIClient`` with a fresh session pool.

    The outgoing client's session is closed immediately when a replacement is
    created so connection pools are not held open until process exit.
    The new client is tracked in ``_managed_api_clients`` for final cleanup
    via the single module-level ``atexit`` hook.
    """
    if not API_BASE_URL:
        raise ValueError("API_URL environment variable must be set")

    # Eagerly close any previously cached client whose session pool is now
    # orphaned after a cache clear.
    for _old in _managed_api_clients:
        try:
            _old.close()
        except Exception:
            pass
    _managed_api_clients.clear()

    client = APIClient(
        API_BASE_URL,
        timeout=API_TIMEOUT,
        health_timeout=HEALTH_TIMEOUT,
        max_retries=MAX_RETRIES,
        api_key=API_KEY,
    )
    _managed_api_clients.append(client)
    return client


# =============================================================================
# API HEALTH CHECK
# =============================================================================


@st.cache_data(ttl=HEALTH_CHECK_TTL, show_spinner=False)
def check_api_health_cached() -> (
    tuple[bool, str | None, str | None, str | None, str | None]
):
    """Check API health with caching."""
    try:
        client = get_api_client()
        health_data = client.health_check()
        status = health_data.get("status", "unknown")
        is_healthy = str(status).lower() in ("healthy", "ok", "true", "1")
        model_name = health_data.get("model_name", "unknown")
        pipeline_version = health_data.get("pipeline_version")
        hybrid_version = health_data.get("hybrid_version")

        _cross_check_schema(health_data)

        if not is_healthy:
            error_msg = health_data.get("detail", "Service degraded")
            return False, model_name, error_msg, pipeline_version, hybrid_version

        return True, model_name, None, pipeline_version, hybrid_version
    except Exception as e:
        logger.error(f"Health check failed: {type(e).__name__}")
        return False, None, "Cannot connect to API", None, None


def _cross_check_schema(health_data: dict[str, Any]) -> None:
    """AUD-07: Cross-check API-reported valid values against local constants."""
    schema_checks = {
        "valid_regions": (VALID_REGIONS, "VALID_REGIONS"),
        "valid_sex": (VALID_SEX, "VALID_SEX"),
        "valid_smoker": (VALID_SMOKER, "VALID_SMOKER"),
    }
    for api_key, (local_values, const_name) in schema_checks.items():
        api_values = health_data.get(api_key)
        if api_values is None:
            continue
        api_set = set(str(v).strip().lower() for v in api_values)
        local_set = set(str(v).strip().lower() for v in local_values)
        if api_set != local_set:
            logger.warning(
                f"Schema mismatch for {const_name}: "
                f"API={sorted(api_set)} vs UI={sorted(local_set)}."
            )


def display_api_status() -> bool:
    """Display API connection status.

    Checks the circuit breaker state *before* consulting the TTL-cached health
    result.  When the breaker is open the cached 200 OK is stale by definition
    and must be bypassed (Issue 4 — health check / circuit breaker desync).
    """
    try:
        cb_state = get_api_client().circuit_breaker.get_state()
        if cb_state["state"] == "open":
            col1, col2 = st.columns([4, 1])
            with col1:
                st.error(
                    f"❌ API Unavailable: circuit breaker open "
                    f"({cb_state['failure_count']} consecutive failures)"
                )
            with col2:
                if st.button("🔄 Retry", key="retry_health_btn", width="stretch"):
                    check_api_health_cached.clear()
                    st.rerun()
            return False
    except Exception:
        # If we cannot even instantiate the client, fall through to the
        # cached health check which will surface the configuration error.
        pass

    is_healthy, model_name, error_msg, pipeline_version, hybrid_version = (
        check_api_health_cached()
    )

    if is_healthy:
        safe_model = safe_html(model_name or "unknown")
        version_str = (
            f" | Pipeline v{safe_html(pipeline_version)}" if pipeline_version else ""
        )
        st.success(f"✅ API Connected | Model: **{safe_model}**{version_str}")
    else:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.error(f"❌ API Unavailable: {safe_html(error_msg or 'unknown error')}")
        with col2:
            if st.button("🔄 Retry", key="retry_health_btn", width="stretch"):
                check_api_health_cached.clear()
                st.rerun()

    return is_healthy


# =============================================================================
# VALIDATION & SECURITY
# =============================================================================


def validate_upload(file) -> tuple[bool, str | None]:
    if file is None:
        return False, "No file provided"
    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"File too large (max {MAX_FILE_SIZE_MB}MB)"
    if not file.name.endswith(".csv"):
        return False, "Only CSV files allowed"
    return True, None


def check_csv_injection(df: pd.DataFrame) -> bool:
    """Enhanced CSV injection detection."""
    dangerous_starts = ("=", "+", "-", "@", "\t", "\r", "\n", "|", "#", "%", "~", "^")
    dangerous_patterns = [
        "cmd|", "powershell", "system(", "exec(", "eval(",
        "@sum", "@if", "@cmd", "dde", "|calc", "|cmd",
    ]

    for col in df.columns:
        col_str = str(col).strip()
        if col_str and col_str[0] in dangerous_starts:
            logger.warning(f"Dangerous column name detected: {col}")
            return True

    for col in df.select_dtypes(include="object").columns:
        for val in df[col].dropna():
            val_str = str(val)
            val_stripped = val_str.strip()
            val_lstripped = val_str.lstrip()

            if not val_stripped:
                continue

            try:
                float(val_stripped)
                continue
            except ValueError:
                pass

            if (val_stripped and val_stripped[0] in dangerous_starts) or (
                val_lstripped and val_lstripped[0] in dangerous_starts
            ):
                logger.warning(f"Dangerous value in {col}: {val_str[:20]}")
                return True

            val_lower = val_stripped.lower()
            if any(pattern in val_lower for pattern in dangerous_patterns):
                logger.warning(f"Command injection pattern in {col}")
                return True

    return False


def rate_limit_check(namespace: str = "default") -> bool:
    """Check if rate limit allows a request."""
    key = f"last_request_time_{namespace}"
    current = time.time()
    last_time = st.session_state.get(key, 0)

    if current - last_time < RATE_LIMIT_SECONDS:
        return False

    st.session_state[key] = current
    return True


def validate_bmi_calculation(
    height: float, weight: float
) -> tuple[bool, str | None, float | None]:
    try:
        height_val = float(height)
        weight_val = float(weight)
    except (ValueError, TypeError):
        return False, "Height and weight must be numbers", None

    if height_val <= 0 or weight_val <= 0:
        return False, "Height and weight must be positive", None

    if not (50 <= height_val <= 250):
        return False, "Height must be between 50-250 cm", None
    if not (20 <= weight_val <= 300):
        return False, "Weight must be between 20-300 kg", None

    bmi = round(weight_val / ((height_val / 100) ** 2), 1)

    if not math.isfinite(bmi):
        return False, "Invalid BMI calculation result", None

    if not (10 <= bmi <= 60):
        return False, f"Calculated BMI ({bmi:.1f}) outside valid range (10-60)", None

    return True, None, bmi


def validate_realtime(age: int, bmi: float, children: int) -> list[str]:
    warnings = []
    if age > 65:
        warnings.append("⚠️ Higher premiums expected for age 65+")
    if bmi > BMI_OVERWEIGHT:
        warnings.append("⚠️ Obesity range detected")
    if children > 3:
        warnings.append("⚠️ Large family may affect premiums")
    return warnings


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_bmi_category(bmi: float) -> str:
    try:
        bmi_val = float(bmi)
    except (ValueError, TypeError):
        return "Invalid"

    if bmi_val < BMI_UNDERWEIGHT:
        return "Underweight"
    elif bmi_val < BMI_NORMAL:
        return "Normal"
    elif bmi_val < BMI_OVERWEIGHT:
        return "Overweight"
    else:
        return "Obese"


def calculate_risk_score(pred: Prediction) -> int:
    score = 0
    if pred.smoker == "yes":
        score += 40
    if pred.bmi > BMI_OVERWEIGHT:
        score += 20
    elif pred.bmi > BMI_NORMAL:
        score += 10
    if pred.age > 65:
        score += 25
    elif pred.age > 50:
        score += 15
    if pred.children > 3:
        score += 10
    return min(100, score)


def get_risk_level(score: int) -> str:
    if score < RISK_LOW_THRESHOLD:
        return "Low"
    elif score < RISK_MEDIUM_THRESHOLD:
        return "Medium"
    return "High"


def safe_display(func):
    """Decorator for safe display function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            st.error(f"⚠️ Error displaying {func.__name__.replace('display_', '')}")
            logger.error(f"{func.__name__} failed: {e}", exc_info=True)
            with st.expander("Error Details"):
                st.code(str(e))
        except Exception as e:
            logger.critical(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise

    return wrapper


# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================


def init_session_state() -> None:
    """Initialize session state with defaults."""
    defaults = {
        "predictions_history": [],
        "history_validated": False,
        "show_admin": False,
        "last_prediction": None,
        "last_model_used": None,
        "last_form_data": None,
        "last_request_time_predict": 0.0,
        "last_request_time_analysis": 0.0,
        "last_request_time_default": 0.0,
        "uploaded_file_id": None,
        "uploaded_df": None,
        "prediction_error": None,
        "advanced_mode": False,
        "dark_mode": False,
        "bmi_method_radio": BMI_METHOD_DIRECT,
        "batch_result_csv": None,
        "batch_result_filename": None,
        "export_anchor_key": None,
        "export_anchor_csv": None,
        "export_anchor_json": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _is_valid_prediction(item: Any) -> bool:
    try:
        required_attrs = [
            "age", "sex", "bmi", "children", "smoker", "region",
            "prediction", "model_used", "timestamp",
        ]
        for attr in required_attrs:
            if not hasattr(item, attr):
                return False

        if not isinstance(item.age, int): return False
        if not isinstance(item.sex, str): return False
        if not isinstance(item.bmi, (int, float)): return False
        if not isinstance(item.children, int): return False
        if not isinstance(item.smoker, str): return False
        if not isinstance(item.region, str): return False
        if not isinstance(item.prediction, (int, float)): return False
        if not isinstance(item.model_used, str): return False

        return True
    except (AttributeError, TypeError):
        return False


def get_history() -> list[Prediction]:
    """Get validated prediction history with single-pass validation."""
    history = st.session_state.get("predictions_history", [])

    if not isinstance(history, list):
        logger.warning(f"History is not a list: {type(history)}")
        st.session_state.predictions_history = []
        st.session_state.history_validated = False
        return []

    if st.session_state.get("history_validated", False) and isinstance(history, list):
        return history

    valid = [item for item in history if _is_valid_prediction(item)]

    if len(valid) != len(history):
        invalid_count = len(history) - len(valid)
        logger.info(
            f"History cleanup: {len(history)} total -> {len(valid)} valid "
            f"({invalid_count} removed)"
        )
        st.session_state.predictions_history = valid

    st.session_state.history_validated = True
    return valid


def get_history_dataframe(history: list) -> pd.DataFrame:
    """Build a DataFrame from the current prediction history snapshot."""
    if not history:
        return pd.DataFrame()
    return pd.DataFrame([p.to_dict() for p in history])


# =============================================================================
# PREDICTION LOGIC
# =============================================================================


def calculate_prediction(
    form_data: FormData, silent: bool = False
) -> tuple[float, str] | None:
    """Calculate insurance premium prediction."""
    try:
        client = get_api_client()
        response = client.predict_single(form_data)

        prediction = response.get("prediction")
        if prediction is None:
            raise ValueError("No prediction returned from API")

        prediction = float(prediction)

        if not math.isfinite(prediction):
            raise ValueError("Prediction is not a valid number")

        if not (PREDICTION_MIN <= prediction <= PREDICTION_MAX):
            raise ValueError(
                f"Prediction ${prediction:,.2f} outside expected range "
                f"[${PREDICTION_MIN:,.0f}–${PREDICTION_MAX:,.0f}]. "
                f"Verify model health."
            )

        model_used = response.get("model_used", "unknown")

        if not _is_known_model(model_used):
            logger.warning(
                f"Unexpected model_used='{model_used}' not in KNOWN_MODELS "
                f"and does not match known prefixes."
            )

        logger.info(f"Prediction successful: ${prediction:.2f} (model={model_used})")
        return prediction, model_used

    except RateLimitError:
        if not silent:
            st.warning("⏱️ Server is busy (rate limit). Wait a moment and try again.")
        logger.warning("Prediction rate-limited (429) — transient, CB unaffected")
    except UnicodeEncodeError as e:
        _char = repr(e.object[e.start]) if e.start < len(e.object) else "?"
        _msg = (
            f"Non-ASCII character {_char} at position {e.start} cannot be used "
            "in HTTP headers. Check your .env file: the API_KEY value must be "
            "plain ASCII with no em-dashes or trailing inline comments."
        )
        if not silent:
            st.error(f"🔑 API key encoding error — {_msg}")
        logger.error(f"API key UnicodeEncodeError: {_msg}")
    except TimeoutError:
        if not silent:
            st.error("⏱️ Request timed out. Please try again.")
        logger.error("Prediction timeout")
    except ValueError as e:
        if not silent:
            st.error(f"❌ {str(e)}")
        error_msg = str(e).encode("ascii", errors="replace").decode("ascii")
        logger.error(f"Prediction validation error: {error_msg}")
    except ConnectionError as e:
        if not silent:
            st.error(f"📡 {str(e)}")
        error_msg = str(e).encode("ascii", errors="replace").decode("ascii")
        logger.error(f"Prediction connection error: {error_msg}")
    except Exception as e:
        if not silent:
            st.error("⚠️ An error occurred. Please try again.")
        error_msg = str(e).encode("ascii", errors="replace").decode("ascii")
        logger.error(f"Prediction error: {error_msg}", exc_info=True)

    return None


def add_prediction_to_history(
    form_data: FormData, prediction: float, model_used: str
) -> None:
    """Add prediction to history with automatic trimming."""
    try:
        pred = Prediction.from_form_data(form_data, prediction, model_used)
        logger.info(
            f"Created Prediction: age={pred.age}, premium=${pred.prediction:,.2f}"
        )

        new_history = st.session_state.predictions_history.copy()
        if len(new_history) >= MAX_HISTORY_SIZE:
            new_history.pop(0)
        new_history.append(pred)

        st.session_state.predictions_history = new_history
        st.session_state.history_validated = False
        logger.info(f"History updated: {len(new_history)} items")

    except Exception as e:
        logger.error(f"Error adding to history: {e}", exc_info=True)


# =============================================================================
# BATCH PROCESSING
# =============================================================================


def process_batch_row_safe(
    row_dict: dict[str, Any], client: APIClient
) -> dict[str, Any]:
    """Safely process a single batch row with comprehensive validation."""
    try:
        required_keys = ["age", "sex", "bmi", "children", "smoker", "region"]

        for key in required_keys:
            if key not in row_dict:
                return {
                    **row_dict,
                    "predicted_cost": None,
                    "model": "error",
                    "imputed_fields": "",
                    "status": f"missing_column: {key}",
                }

        FIELD_DEFAULTS = {
            "age": (30, _safe_int, lambda v: _safe_int(v, 30)),
            "sex": ("male", _safe_str, lambda v: _safe_str(v, "male")),
            "bmi": (25.0, _safe_float, lambda v: _safe_float(v, 25.0)),
            "children": (0, _safe_int, lambda v: _safe_int(v, 0)),
            "smoker": ("no", _safe_str, lambda v: _safe_str(v, "no")),
            "region": ("northeast", _safe_str, lambda v: _safe_str(v, "northeast")),
        }
        imputed = []
        converted: dict[str, Any] = {}
        for field, (default, _, converter) in FIELD_DEFAULTS.items():
            raw = row_dict.get(field)
            is_missing = (
                raw is None
                or (isinstance(raw, float) and pd.isna(raw))
                or str(raw).strip() == ""
            )
            converted[field] = converter(raw)
            if is_missing:
                imputed.append(field)

        form_data = FormData(**converted)

        response = client.predict_single(form_data)
        status = "success_with_imputation" if imputed else "success"
        return {
            **row_dict,
            "predicted_cost": float(response.get("prediction")),
            "model": response.get("model_used", "unknown"),
            "imputed_fields": ",".join(imputed),
            "status": status,
        }

    except requests.exceptions.HTTPError as e:
        _sc = e.response.status_code if e.response is not None else None
        if _sc == 429:
            return {
                **row_dict,
                "predicted_cost": None,
                "model": "rate_limited",
                "imputed_fields": "",
                "status": "rate_limited: server busy, retry batch later",
            }
        return {
            **row_dict,
            "predicted_cost": None,
            "model": "error",
            "imputed_fields": "",
            "status": f"http_{_sc}: {str(e)[:40]}",
        }
    except ValueError as e:
        return {
            **row_dict,
            "predicted_cost": None,
            "model": "error",
            "imputed_fields": "",
            "status": f"validation: {str(e)[:50]}",
        }
    except Exception as e:
        return {
            **row_dict,
            "predicted_cost": None,
            "model": "error",
            "imputed_fields": "",
            "status": f"error: {str(e)[:50]}",
        }


# =============================================================================
# FORM HANDLERS
# =============================================================================


def handle_reset() -> None:
    """Reset form to default values and clear ALL prediction-related state."""
    init_session_state()
    st.session_state.update(
        {
            "age_input": 30,
            "sex_input": "male",
            "bmi_input": 25.0,
            "children_input": 0,
            "smoker_input": "no",
            "region_input": "northeast",
            "bmi_method_radio": BMI_METHOD_DIRECT,
            "height_input": 170,
            "weight_input": 70,
            "prediction_error": None,
            "last_prediction": None,
            "last_model_used": None,
            "last_form_data": None,
            "batch_result_csv": None,
            "batch_result_filename": None,
            "uploaded_file_id": None,
            "uploaded_df": None,
            "export_anchor_key": None,
            "export_anchor_csv": None,
            "export_anchor_json": None,
        }
    )


def handle_predict() -> None:
    """Handle prediction request."""
    init_session_state()

    if not rate_limit_check(namespace="predict"):
        st.session_state.prediction_error = "⏱️ Please wait before next request"
        return

    required_widget_keys = [
        "bmi_method_radio", "age_input", "sex_input",
        "children_input", "smoker_input", "region_input",
    ]
    if not all(k in st.session_state for k in required_widget_keys):
        st.session_state.prediction_error = "⚠️ Form not ready — please try again."
        return

    try:
        if st.session_state.bmi_method_radio == BMI_METHOD_CALCULATE:
            if "calculated_bmi" not in st.session_state:
                st.session_state.prediction_error = "❌ Calculate BMI first"
                return
            bmi_val = st.session_state.calculated_bmi
        else:
            bmi_val = st.session_state.bmi_input

        form_data = FormData(
            age=st.session_state.age_input,
            sex=st.session_state.sex_input,
            bmi=bmi_val,
            children=st.session_state.children_input,
            smoker=st.session_state.smoker_input,
            region=st.session_state.region_input,
        )

        with st.spinner("🤖 Calculating your premium..."):
            result = calculate_prediction(form_data)

        if result:
            prediction, model_used = result
            st.session_state.last_prediction = prediction
            st.session_state.last_model_used = model_used
            st.session_state.last_form_data = form_data
            st.session_state.prediction_error = None

            add_prediction_to_history(form_data, prediction, model_used)

            try:
                _anchor_pred = Prediction.from_form_data(
                    form_data, prediction, model_used,
                    timestamp=st.session_state.predictions_history[-1].timestamp,
                )
                _anchor_ts = _anchor_pred.timestamp.strftime("%Y%m%d_%H%M%S")
                _anchor_df = pd.DataFrame([_anchor_pred.to_dict()])
                st.session_state.export_anchor_key = _anchor_ts
                st.session_state.export_anchor_csv = _anchor_df.to_csv(index=False)
                st.session_state.export_anchor_json = json.dumps(
                    _anchor_pred.to_dict(), indent=2
                )
            except Exception as _e:
                logger.warning(f"Could not cache export anchor: {_e}")

    except ValueError as e:
        st.session_state.prediction_error = f"❌ {str(e)}"
    except Exception as e:
        st.session_state.prediction_error = "⚠️ An error occurred"
        logger.error(f"Predict callback error: {e}", exc_info=True)


# =============================================================================
# UI STYLING
# =============================================================================


def get_enhanced_theme_css_cached() -> str:
    """Thin wrapper: reads mutable session_state, delegates to lru_cache'd builder.

    v3.5.12 (BUG-3 fix): Moved all palette / CSS logic into _build_theme_css()
    which is decorated with @lru_cache(maxsize=2).  The CSS string is now built
    only once per (is_dark=True/False) value for the server process lifetime.
    """
    is_dark: bool = st.session_state.get("dark_mode", False)
    return _build_theme_css(is_dark)


from functools import lru_cache as _lru_cache  # noqa: E402


@_lru_cache(maxsize=2)
def _build_theme_css(is_dark: bool) -> str:
    """Pure function: build the full <style> block for the given palette.

    Called only by get_enhanced_theme_css_cached().  Cached per process so
    Streamlit reruns (widget interactions) never rebuild the CSS string.
    """

    if is_dark:
        bg_app         = "#0e1117"
        bg_app_end     = "#0e1117"
        bg_primary     = "#1e2130"
        bg_secondary   = "#262730"
        bg_tertiary    = "#2d3250"
        bg_sidebar     = "#0e1117"
        bg_sidebar_end = "#1a1f2e"
        text_primary   = "#fafafa"
        text_secondary = "#e2e8f0"
        text_muted     = "#a0aec0"
        text_inverse   = "#ffffff"
        border_light   = "rgba(255, 255, 255, 0.12)"
        border_medium  = "rgba(255, 255, 255, 0.22)"
        shadow_sm      = "0 2px 4px   rgba(0, 0, 0, 0.45)"
        shadow_md      = "0 4px 6px   rgba(0, 0, 0, 0.55)"
        shadow_lg      = "0 10px 30px rgba(0, 0, 0, 0.65)"
        shadow_xl      = "0 20px 50px rgba(0, 0, 0, 0.75)"
        overlay_light  = "rgba(255, 255, 255, 0.05)"
        overlay_medium = "rgba(255, 255, 255, 0.10)"
        tab_hover_bg   = "rgba(90, 103, 216, 0.28)"
        expander_hover = "rgba(90, 103, 216, 0.18)"
        # Dropdown portal colours (dark palette)
        portal_bg      = "#1e2130"
        portal_text    = "#fafafa"
        portal_hover   = "rgba(255, 255, 255, 0.10)"
        color_scheme   = "dark"
    else:
        bg_app         = "#e8eef5"
        bg_app_end     = "#dce4f0"
        bg_primary     = "#ffffff"
        bg_secondary   = "#f7fafc"
        bg_tertiary    = "#edf2f7"
        bg_sidebar     = "#ffffff"
        bg_sidebar_end = "#f7fafc"
        text_primary   = "#1a202c"
        text_secondary = "#2d3748"
        text_muted     = "#718096"
        text_inverse   = "#ffffff"
        border_light   = "rgba(128, 128, 128, 0.20)"
        border_medium  = "rgba(128, 128, 128, 0.35)"
        shadow_sm      = "0 2px 4px  rgba(0, 0, 0, 0.18)"
        shadow_md      = "0 4px 6px  rgba(0, 0, 0, 0.25)"
        shadow_lg      = "0 10px 30px rgba(0, 0, 0, 0.35)"
        shadow_xl      = "0 20px 50px rgba(0, 0, 0, 0.45)"
        overlay_light  = "rgba(128, 128, 128, 0.05)"
        overlay_medium = "rgba(128, 128, 128, 0.12)"
        tab_hover_bg   = "rgba(90, 103, 216, 0.12)"
        expander_hover = "rgba(90, 103, 216, 0.07)"
        # Dropdown portal colours (light palette)
        portal_bg      = "#ffffff"
        portal_text    = "#1a202c"
        portal_hover   = "rgba(128, 128, 128, 0.12)"
        color_scheme   = "light"

    # BUG-4 FIX: compute primary hex explicitly here so the toggle ON-state
    # CSS uses a hard-wired colour string rather than var(--primary).
    # var(--primary) resolves via var(--primary-color, #5a67d8); Streamlit can
    # override --primary-color with its own default red (#FF4B4B) when the
    # hamburger theme or OS dark-mode changes, causing the toggle track to
    # appear red instead of indigo.  Baking the literal hex into the stylesheet
    # is immune to that override.
    primary_hex = "#7c8aed" if is_dark else "#5a67d8"

    return f"""
    <style>
    /* ===== THEME VARIABLES (v3.5.11 — Python-driven palette, dark-mode aware) ===== */
    /* BUG-3/4 FIX: color-scheme is now scoped to the active palette value so
       the browser/OS renders UA controls (scrollbars, native inputs) in the
       correct palette, and so Styletron's prefers-color-scheme detection is
       NOT confused by a hard-wired "light" in dark-mode.
       NOTE: color-scheme on :root does NOT affect prefers-color-scheme media
       queries (those read the OS setting), but it does suppress browser-UA
       dark rendering artifacts when the app is in light mode. */
    :root {{
        color-scheme: {color_scheme};

        --primary:       var(--primary-color, #5a67d8);
        --primary-dark:  #4c51bf;
        --primary-light: #7c8aed;
        --success: #48bb78;
        --warning: #ed8936;
        --danger:  #f56565;

        --bg-app:         {bg_app};
        --bg-app-end:     {bg_app_end};
        --bg-primary:     {bg_primary};
        --bg-secondary:   {bg_secondary};
        --bg-tertiary:    {bg_tertiary};
        --bg-sidebar:     {bg_sidebar};
        --bg-sidebar-end: {bg_sidebar_end};

        --text-primary:   {text_primary};
        --text-secondary: {text_secondary};
        --text-muted:     {text_muted};
        --text-inverse:   {text_inverse};

        --border-light:  {border_light};
        --border-medium: {border_medium};

        --shadow-sm: {shadow_sm};
        --shadow-md: {shadow_md};
        --shadow-lg: {shadow_lg};
        --shadow-xl: {shadow_xl};

        --overlay-light:  {overlay_light};
        --overlay-medium: {overlay_medium};

        --tab-hover-bg:      {tab_hover_bg};
        --expander-hover-bg: {expander_hover};
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);

        /* Portal / dropdown colours — switched per palette by Python */
        --portal-bg:    {portal_bg};
        --portal-text:  {portal_text};
        --portal-hover: {portal_hover};
    }}

    .stApp {{
        background: linear-gradient(135deg, var(--bg-app) 0%, var(--bg-app-end) 100%) !important;
        color: var(--text-primary) !important;
    }}

    .main {{ background-color: transparent !important; }}

    .main .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 5rem !important;
        padding-right: 5rem !important;
        max-width: 100% !important;
        background-color: transparent !important;
    }}

    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {{
        color: var(--text-primary) !important;
    }}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
        color: var(--text-primary) !important;
    }}
    .stCodeBlock, code {{
        background-color: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
    }}
    .stTextInput input, .stNumberInput input,
    .stSelectbox select, .stTextArea textarea {{
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-light) !important;
    }}
    label, .stMarkdown label {{ color: var(--text-secondary) !important; }}

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {{
        width: 21rem !important;
        min-width: 21rem !important;
        flex-shrink: 0 !important;
        background: linear-gradient(180deg, var(--bg-sidebar) 0%, var(--bg-sidebar-end) 100%) !important;
        box-shadow: 2px 0 10px var(--border-light) !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        width: 21rem !important; min-width: 21rem !important;
        background-color: transparent !important;
    }}
    [data-testid="stSidebar"] .element-container {{
        will-change: auto; margin-bottom: 0.5rem;
    }}
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {{ color: var(--text-primary) !important; }}
    [data-testid="stSidebar"] .stMarkdown h3 {{
        color: var(--primary) !important;
        font-weight: 700;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--primary);
        margin-bottom: 1rem;
    }}
    [data-testid="stSidebar"] .stMarkdown strong,
    [data-testid="stSidebar"] .stMarkdown b {{ color: var(--text-primary) !important; }}
    [data-testid="stSidebar"] .stAlert {{
        min-height: 3.5rem;
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }}
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] [data-baseweb="select"] {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-light) !important;
    }}

    /* ===== BASEWEB INPUT CONTAINERS (BUG-4 FIX v3.5.13) ===== */
    [data-baseweb="base-input"] {{
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="base-input"] {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }}
    [data-baseweb="base-input"] input {{
        color: var(--text-primary) !important;
        caret-color: var(--text-primary) !important;
    }}

    /* ===== SELECTBOX — VALUE CONTAINER ===== */
    [data-baseweb="select"] > div:first-child {{
        background-color: var(--bg-primary) !important;
        border-color: var(--border-light) !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {{
        background-color: var(--bg-secondary) !important;
    }}
    /* Selected value text and placeholder */
    [data-baseweb="select"] [data-value],
    [data-baseweb="select"] input,
    [data-baseweb="select"] span {{
        color: var(--text-primary) !important;
    }}
    /* BUG-2 FIX (v3.5.11): The sidebar compound-selector fix covered sidebar
       selects but NOT main-content selects (e.g. What-If "Vary parameter").
       Styletron's <style data-hydrate> sets the inner span text to white after
       our block, winning the source-order battle for same-!important rules.
       This higher-specificity rule wins on specificity weight, not source order,
       so it beats Styletron for main-content selects as well as sidebar selects. */
    html body [data-baseweb="select"] > div > div > div,
    html body [data-baseweb="select"] > div > div > span,
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div[data-value],
    [data-testid="stSidebar"] [data-baseweb="select"] [aria-selected],
    [data-testid="stSidebar"] [data-baseweb="select"] > div > div > div,
    [data-testid="stSidebar"] [data-baseweb="select"] > div > div > span {{
        color: var(--text-primary) !important;
    }}
    /* Chevron / dropdown arrow icon */
    [data-baseweb="select"] svg {{
        fill: var(--text-muted) !important;
    }}

    /* ===== DROPDOWN PORTAL (POPOVER / LISTBOX) ===== */
    /* Root problem: BaseWeb's Styletron CSS-in-JS injects its theme styles
       into a <style data-hydrate> tag AFTER our <style> tag in React's render
       cycle, winning the same-!important source-order battle.
       v3.5.11 fix: CSS variables --portal-bg / --portal-text / --portal-hover
       are set per-palette by Python (_build_theme_css), so the correct values
       are baked directly into the stylesheet rather than relying on cascade.
       html body prefix provides maximum specificity without JS. */
    html body [data-baseweb="layer"] {{
        color-scheme: {color_scheme} !important;
    }}
    html body [data-baseweb="layer"],
    html body [data-baseweb="layer"] > div,
    html body [data-baseweb="popover"],
    html body [data-baseweb="popover"] > div,
    html body [data-baseweb="menu"],
    html body ul[role="listbox"],
    html body [role="listbox"] {{
        background: var(--portal-bg) !important;
        background-color: var(--portal-bg) !important;
        border-color: var(--border-light) !important;
    }}
    html body [data-baseweb="menu"] li,
    html body [data-baseweb="layer"] li,
    html body ul[role="listbox"] li,
    html body [role="listbox"] [role="option"] {{
        background: var(--portal-bg) !important;
        background-color: var(--portal-bg) !important;
        color: var(--portal-text) !important;
    }}
    html body [data-baseweb="menu"] [aria-selected="true"],
    html body [data-baseweb="menu"] li:hover,
    html body [role="listbox"] [role="option"]:hover,
    html body [role="listbox"] [aria-selected="true"] {{
        background: var(--portal-hover) !important;
        background-color: var(--portal-hover) !important;
        color: var(--portal-text) !important;
    }}

    /* ===== NUMBER INPUT STEPPER BUTTONS (- / +) ===== */
    [data-testid="stNumberInput"] button {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-light) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button {{
        background-color: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
    }}
    [data-testid="stNumberInput"] button:hover {{
        background-color: var(--overlay-medium) !important;
    }}

    /* ===== TOGGLE TRACK ===== */
    /* BUG-3/4 FIX (v3.5.11):
       1. color-scheme rule scoped to active palette at :root level (above) so
          UA rendering is correct without needing it on stCheckbox.
       2. Removed `color-scheme: light` from [data-testid="stCheckbox"] which
          was locking the Dark Mode toggle itself into light rendering regardless
          of the user's active palette.
       3. html body prefix raises specificity to beat Styletron's equal-!important
          source-order injection for toggle track background. */
    html body [data-testid="stCheckbox"] [role="switch"] {{
        background-color: var(--border-medium) !important;
        border-color: transparent !important;
        border-radius: 999px !important;
        transition: background-color 0.2s ease !important;
    }}
    /* BUG-4 FIX (v3.5.12): use hard-wired palette hex instead of
       var(--primary) which chains to var(--primary-color, #5a67d8).
       Streamlit can override --primary-color with its own red (#FF4B4B)
       when the hamburger/OS theme changes, turning the ON track red. */
    html body [data-testid="stCheckbox"] [role="switch"][aria-checked="true"] {{
        background-color: {primary_hex} !important;
    }}
    html body [data-testid="stCheckbox"] [role="switch"] > div {{
        background-color: #ffffff !important;
        border-radius: 50% !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.35) !important;
    }}

    /* ===== CHECKBOX (st.checkbox) ===== */
    /* NOTE: color-scheme is no longer set per-widget here (BUG-4 FIX).
       It is set on :root with the active palette value. The per-widget
       color-scheme was locking the Dark Mode toggle into light rendering. */
    [data-testid="stCheckbox"] input[type="checkbox"] {{
        accent-color: var(--primary) !important;
        cursor: pointer;
    }}
    /* NOTE — checkbox dark-square residual limitation:
       BaseWeb uses data-baseweb="checkbox" for BOTH st.toggle() AND
       st.checkbox() in Streamlit 1.50.  Styletron injects its dark-theme
       background on the indicator via <style data-hydrate> appended AFTER
       our <style> tag, winning source-order.  The checkbox visual indicator
       remains dark on OS-dark systems.  Mitigation: replace any user-facing
       st.checkbox() that shows toggle semantics with st.toggle() instead.
       The "Multivariate" checkbox in What-If has been migrated (BUG-5 FIX). */

    /* ===== TITLE HEADER ===== */
    .title-container {{
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: var(--text-inverse);
        padding: 3rem 2rem; border-radius: 20px; margin-bottom: 2rem;
        box-shadow: var(--shadow-xl); position: relative; overflow: hidden;
    }}
    .title-container::before {{
        content: ""; position: absolute; top: -50%; right: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, var(--overlay-medium) 0%, transparent 70%);
    }}
    .title-container h1 {{
        font-size: 2.5rem; margin: 0; font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        position: relative; z-index: 1;
        color: var(--text-inverse) !important;
    }}
    .title-container p {{
        font-size: 1.1rem; opacity: 0.95; margin: 0.5rem 0 0 0;
        position: relative; z-index: 1;
        color: var(--text-inverse) !important;
    }}

    /* ===== TABS ===== */
    html body .stTabs [data-baseweb="tab-list"],
    html body [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        gap: 8px; background-color: var(--bg-primary) !important;
        padding: 10px; border-radius: 12px; box-shadow: var(--shadow-sm);
    }}
    html body .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"],
    html body [data-testid="stTabs"] [data-baseweb="tab-list"] [data-baseweb="tab"] {{
        height: 50px; padding: 0 24px;
        background-color: transparent !important;
        border-radius: 8px; color: var(--text-secondary) !important;
        font-weight: 600; border: none; transition: var(--transition);
    }}
    html body .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:hover,
    html body [data-testid="stTabs"] [data-baseweb="tab-list"] [data-baseweb="tab"]:hover {{
        background-color: var(--tab-hover-bg) !important;
        color: var(--primary) !important; transform: translateY(-2px);
    }}
    html body .stTabs [data-baseweb="tab-list"] [aria-selected="true"][data-baseweb="tab"],
    html body [data-testid="stTabs"] [data-baseweb="tab-list"] [aria-selected="true"][data-baseweb="tab"] {{
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%) !important;
        color: var(--text-inverse) !important;
        box-shadow: var(--shadow-md); transform: translateY(-2px);
    }}
    [data-baseweb="tab-panel"] {{ background-color: transparent !important; }}

    /* ===== BUTTONS ===== */
    .stButton > button {{
        border-radius: 10px; font-weight: 600; transition: var(--transition);
        box-shadow: var(--shadow-sm);
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-light) !important;
    }}
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {{
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        color: var(--text-inverse) !important; border: none !important;
    }}
    .stButton > button:hover {{ transform: translateY(-2px); box-shadow: var(--shadow-md); }}
    button:focus-visible, a:focus-visible {{
        outline: 3px solid var(--primary); outline-offset: 2px;
    }}

    /* ===== METRICS ===== */
    [data-testid="metric-container"],
    [data-testid="stMetric"] {{
        background: var(--bg-primary) !important;
        padding: 1rem; border-radius: 12px;
        box-shadow: var(--shadow-sm); transition: var(--transition);
        border: 1px solid var(--border-light);
    }}
    [data-testid="metric-container"]:hover,
    [data-testid="stMetric"]:hover {{
        transform: translateY(-4px); box-shadow: var(--shadow-md);
    }}
    [data-testid="metric-container"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: var(--primary) !important; font-size: 1.8rem; font-weight: 700;
    }}
    [data-testid="metric-container"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricLabel"] {{
        color: var(--text-secondary) !important;
    }}

    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 8px;
        list-style: none;
    }}
    .streamlit-expanderHeader:hover,
    [data-testid="stExpander"] summary:hover {{
        background-color: var(--expander-hover-bg) !important;
        transform: translateX(4px);
    }}
    [data-testid="stExpander"] summary:focus,
    [data-testid="stExpander"] summary:focus-visible {{
        outline: 2px solid var(--primary) !important;
        outline-offset: 2px !important;
    }}
    .streamlit-expanderContent,
    [data-testid="stExpander"] details > div {{
        background-color: var(--bg-primary) !important;
        border: 1px solid var(--border-light) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }}
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {{
        color: var(--text-primary) !important;
    }}

    /* ===== DATAFRAMES ===== */
    .stDataFrame, [data-testid="stDataFrame"] {{
        background-color: var(--bg-primary) !important;
        border-radius: 8px; overflow: hidden;
    }}

    /* ===== THEMED HTML TABLE ===== */
    /* Mechanism D FIX (v3.5.11): All color and background-color declarations
       now carry !important so they cannot be silently overridden by any future
       Streamlit/BaseWeb global table rules, regardless of injection order. */
    .themed-table-wrapper {{
        width: 100%; overflow-x: auto; border-radius: 10px;
        border: 1px solid var(--border-light);
        box-shadow: var(--shadow-sm); margin-bottom: 1rem;
    }}
    .themed-table {{
        width: 100%; border-collapse: collapse;
        font-size: 0.9rem; font-family: Arial, sans-serif;
    }}
    .themed-table thead tr {{ background-color: var(--bg-secondary) !important; }}
    .themed-table th {{
        padding: 0.75rem 1rem; text-align: left; font-weight: 600;
        color: var(--text-secondary) !important;
        border-bottom: 2px solid var(--border-medium); white-space: nowrap;
    }}
    .themed-table tbody tr {{
        background-color: var(--bg-primary) !important;
        transition: background-color 0.15s ease;
    }}
    .themed-table tbody tr:nth-child(even) {{ background-color: var(--bg-secondary) !important; }}
    .themed-table tbody tr:hover {{ background-color: var(--overlay-medium) !important; }}
    .themed-table td {{
        padding: 0.65rem 1rem; color: var(--text-primary) !important;
        border-bottom: 1px solid var(--border-light); white-space: nowrap;
    }}
    .themed-table tbody tr:last-child td {{ border-bottom: none; }}
    .themed-table td.num, .themed-table th.num {{ text-align: right; }}

    /* ===== PLOTLY CHARTS ===== */
    .js-plotly-plot .plotly,
    .js-plotly-plot .plotly .modebar {{ background-color: transparent !important; }}

    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] {{
        background-color: var(--bg-primary) !important;
        border-radius: 10px;
    }}
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploader"] section {{
        background-color: var(--bg-primary) !important;
        border: 2px dashed var(--border-medium) !important;
        border-radius: 10px !important;
    }}
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] p {{
        color: var(--text-primary) !important;
    }}
    [data-testid="stFileUploaderDropzone"] svg {{
        fill: var(--text-muted) !important;
    }}
    /* BUG-5 FIX (v3.5.12): the "Browse files" button inside the dropzone was
       not targeted by any rule, so it inherited the browser/OS dark background
       in light mode — appearing near-black against the light dropzone area.
       Selectors cover both the known data-testid and the bare button fallback. */
    [data-testid="stFileUploaderDropzoneButton"],
    [data-testid="stFileUploaderDropzoneButton"] button,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneButton"] button {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-medium) !important;
        border-radius: 8px !important;
    }}
    [data-testid="stFileUploaderDropzoneButton"]:hover button,
    [data-testid="stFileUploaderDropzoneButton"] button:hover {{
        background-color: var(--overlay-medium) !important;
    }}

    /* ===== MISC ===== */
    .stDownloadButton > button {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-light) !important;
    }}
    .stProgress > div > div {{ background-color: var(--bg-secondary) !important; }}
    .stProgress > div > div > div {{ background-color: var(--primary) !important; }}
    .stSpinner > div {{ border-color: var(--primary) !important; }}
    [data-testid="stStatusWidget"] {{
        background-color: var(--bg-primary) !important;
        border: 1px solid var(--border-light) !important;
        color: var(--text-primary) !important;
    }}

    /* ===== PREDICTION RESULT ===== */
    .prediction-result {{
        background: linear-gradient(135deg, var(--success) 0%, #38a169 100%);
        padding: 1.5rem 2rem; border-radius: 12px;
        margin-bottom: 1rem; animation: pulse-glow 2s infinite;
    }}
    .prediction-result h3 {{ color: #ffffff !important; margin: 0; font-size: 1.4rem; }}
    @keyframes pulse-glow {{
        0%, 100% {{ box-shadow: 0 0 20px rgba(72, 187, 120, 0.4); }}
        50%       {{ box-shadow: 0 0 40px rgba(72, 187, 120, 0.8); }}
    }}

    /* ===== FOOTER ===== */
    .footer-container {{
        margin-top: 4rem; padding: 2rem;
        background: var(--bg-secondary); border-radius: 16px;
        color: var(--text-primary); text-align: center;
        box-shadow: var(--shadow-lg); border: 1px solid var(--border-light);
    }}
    .footer-container h3 {{ color: var(--text-primary) !important; }}
    .footer-container p  {{ color: var(--text-secondary) !important; }}
    .footer-container a  {{
        color: var(--primary-light); text-decoration: none;
        font-weight: 600; transition: var(--transition);
    }}
    .footer-container a:hover {{ color: var(--primary); text-decoration: underline; }}
    .footer-badge {{
        display: inline-block; background: var(--bg-tertiary);
        padding: 0.25rem 0.75rem; border-radius: 20px;
        font-size: 0.85rem; margin: 0 0.25rem;
        border: 1px solid var(--primary); color: var(--text-primary);
    }}

    /* ===== STREAMLIT NATIVE HEADER / TOOLBAR ===== */
    [data-testid="stHeader"] {{
        background-color: var(--bg-app) !important;
        border-bottom: 1px solid var(--border-light) !important;
    }}
    [data-testid="stToolbar"] {{
        background-color: transparent !important;
        color: var(--text-secondary) !important;
    }}
    [data-testid="stDecoration"] {{
        background: linear-gradient(90deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        height: 3px !important;
    }}
    [data-testid="stToolbar"] button {{
        color: var(--text-secondary) !important;
    }}
    [data-testid="stActionButtonIcon"] {{
        color: var(--text-secondary) !important;
    }}

    /* ===== WIDGET LABELS ===== */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    [data-testid="stWidgetLabel"] label {{
        color: var(--text-primary) !important;
    }}
    [data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"] svg {{
        fill: var(--text-muted) !important;
    }}
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] p,
    [data-testid="stRadio"] span {{
        color: var(--text-primary) !important;
    }}
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] p,
    [data-testid="stCheckbox"] span {{
        color: var(--text-primary) !important;
    }}

    /* ===== ACCESSIBILITY ===== */
    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
            scroll-behavior: auto !important;
        }}
    }}

    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {{
        .title-container {{ padding: 2rem 1.5rem; }}
        .title-container h1 {{ font-size: 1.8rem; }}
        .stTabs [data-baseweb="tab"] {{ height: 45px; padding: 0 16px; font-size: 0.9rem; }}
        [data-testid="stSidebar"] {{ width: 18rem !important; min-width: 18rem !important; }}
        .main .block-container {{ padding-left: 2rem !important; padding-right: 2rem !important; }}
    }}
    </style>
    """


def apply_enhanced_ui() -> None:
    """Apply enhanced UI styling."""
    st.markdown(get_enhanced_theme_css_cached(), unsafe_allow_html=True)




# =============================================================================
# UI COMPONENTS
# =============================================================================


def create_input_form() -> None:
    """Create sidebar input form with stable layout."""
    st.sidebar.markdown("### ⚙️ Mode")
    # BUG-6 FIX: value= is ignored when key= maps to an existing session_state
    # entry (already guaranteed by init_session_state). Removed to avoid the
    # misleading impression that value= drives initialization on every rerun.
    st.sidebar.toggle(
        "Advanced Mode",
        key="advanced_mode",
        help="Enable to access Compare, What-If, Batch, Stats, and Export features",
    )
    st.sidebar.toggle(
        "🌙 Dark Mode",
        key="dark_mode",
        help="Toggle dark mode. Also switch Streamlit's hamburger (≡) theme to match.",
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown("### 📋 Personal Information")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.number_input("Age", 18, 100, key="age_input", value=30)
    with col2:
        st.selectbox("Gender", VALID_SEX, key="sex_input", index=0)

    st.sidebar.markdown("**Body Mass Index (BMI)**")

    st.sidebar.radio(
        "Input method:",
        [BMI_METHOD_DIRECT, BMI_METHOD_CALCULATE],
        key="bmi_method_radio",
        index=0,
    )

    bmi_container = st.sidebar.container()

    with bmi_container:
        if st.session_state.bmi_method_radio == BMI_METHOD_DIRECT:
            st.number_input("BMI Value", 10.0, 60.0, key="bmi_input", value=25.0)
            st.info("ℹ️ BMI range: 10.0 - 60.0 (Normal: 18.5 - 25.0)")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("Height (cm)", 50, 250, key="height_input", value=170)
            with col2:
                st.number_input("Weight (kg)", 20, 300, key="weight_input", value=70)

            is_valid, error_msg, calculated_bmi = validate_bmi_calculation(
                st.session_state.height_input, st.session_state.weight_input
            )

            if is_valid and calculated_bmi is not None:
                st.success(
                    f"✓ BMI: **{calculated_bmi:.1f}** ({get_bmi_category(calculated_bmi)})"
                )
                st.session_state.calculated_bmi = calculated_bmi
            else:
                st.error(error_msg if error_msg else "❌ Invalid BMI calculation")

    st.sidebar.markdown("### 👥 Family & Lifestyle")
    st.sidebar.number_input("Dependents", 0, 10, key="children_input", value=0)
    st.sidebar.selectbox("Smoking", VALID_SMOKER, key="smoker_input", index=0)
    st.sidebar.selectbox("Region", VALID_REGIONS, key="region_input", index=0)

    if st.session_state.bmi_method_radio == BMI_METHOD_CALCULATE:
        bmi_for_warnings = st.session_state.get("calculated_bmi", 25.0)
    else:
        bmi_for_warnings = st.session_state.get("bmi_input", 25.0)

    warning_container = st.sidebar.container()
    with warning_container:
        warnings = validate_realtime(
            st.session_state.age_input,
            bmi_for_warnings,
            st.session_state.children_input,
        )

        if warnings:
            for warning in warnings:
                st.info(warning)
            remaining_warnings = 3 - len(warnings)
            if remaining_warnings > 0:
                spacer_height = remaining_warnings * 60
                st.markdown(
                    f'<div style="height: {spacer_height}px;"></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("✓ All inputs within normal ranges")
            st.markdown('<div style="height: 120px;"></div>', unsafe_allow_html=True)

    error_container = st.sidebar.container()
    with error_container:
        if st.session_state.get("prediction_error"):
            st.error(st.session_state.prediction_error)
        else:
            st.markdown('<div style="height: 60px;"></div>', unsafe_allow_html=True)

    st.sidebar.markdown("### 🚀 Actions")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.button(
            "🔮 Predict",
            type="primary",
            width="stretch",
            on_click=handle_predict,
            key="predict_btn",
        )
    with col2:
        st.button(
            "🔄 Reset",
            width="stretch",
            on_click=handle_reset,
            key="reset_btn",
        )


def display_header() -> None:
    """Display application header."""
    st.markdown(
        """
        <div class="title-container">
            <h1><span class="title-icon">🏥</span> Insurance Cost Predictor Pro</h1>
            <p>AI-powered premium estimation with advanced medical analytics</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_prediction_result(prediction: float, model_used: str) -> None:
    """Display prediction result with metrics."""
    escaped_prediction = safe_html(f"${prediction:,.2f}")
    st.markdown(
        f"""
        <div class="prediction-result">
            <h3>💰 Estimated Annual Premium: {escaped_prediction}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(f"**Model Used:** {safe_html(model_used)}", icon="🤖")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📅 Monthly", f"${prediction/12:,.0f}")
    with col2:
        st.metric("📊 Weekly", f"${prediction/52:,.0f}")
    with col3:
        st.metric("💰 Daily", f"${prediction/365:,.0f}")


def display_welcome_message() -> None:
    """Display welcome message for first-time users."""
    st.markdown("### 👋 Welcome to Insurance Cost Predictor Pro")

    st.markdown("#### ✨ Features:")
    st.markdown(
        """
    - 🤖 **Real-time ML predictions** - XGBoost, LightGBM, Random Forest ensemble
    - ⚖️ **Compare scenarios** - Side-by-side premium analysis
    - 🔬 **What-if analysis** - Medical-grade parameter sensitivity
    - 📁 **Batch CSV processing** - Optimized for large datasets
    - 📤 **Export results** - Download in CSV/JSON formats
    - 🛡️ **Circuit breaker** - API resilience and fault tolerance
    """
    )

    st.markdown("#### 🚀 Get Started:")
    st.markdown(
        """
    1. Fill in your personal information in the **sidebar**
    2. Click **"🔮 Predict"** to get your premium estimate
    3. Enable **Advanced Mode** to access all features
    4. Enable **Admin Panel** for system diagnostics
    """
    )



def display_footer() -> None:
    """Display application footer."""
    st.markdown(
        f"""
        <div class="footer-container">
            <h3 style="margin-bottom: 1rem;">🏥 Insurance Cost Predictor Pro</h3>
            <p style="margin-bottom: 1rem; opacity: 0.9;">
                Powered by advanced machine learning algorithms for accurate premium estimation
            </p>
            <div style="margin-bottom: 1rem;">
                <span class="footer-badge">XGBoost</span>
                <span class="footer-badge">LightGBM</span>
                <span class="footer-badge">Random Forest</span>
            </div>
            <hr style="border-color: rgba(128,128,128,0.2); margin: 1.5rem 0;">
            <p style="font-size: 0.9rem; opacity: 0.8;">
                Built with ❤️ using Streamlit | Plotly | Scikit-learn
            </p>
            <p style="font-size: 0.85rem; opacity: 0.7; margin-top: 0.5rem;">
                © 2026 Insurance Predictor Pro v{safe_html(APP_VERSION)}.
                For educational and analytical purposes only.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_metrics_panel() -> None:
    """Fetch /api/v1/metrics and surface operational drift/degradation alerts."""
    try:
        client = get_api_client()
        m = client.get_metrics()
    except Exception:
        st.caption("⚠️ Metrics unavailable")
        return

    if not m:
        st.caption("No metrics data returned.")
        return

    preds = m.get("predictions", {})
    lat = m.get("latency_ms", {})
    uptime_s = m.get("uptime_seconds", 0)

    error_rate = preds.get("error_rate_pct", 0.0)
    p99 = lat.get("p99_ms", 0.0)
    rejected = preds.get("rejected_overload", 0)

    if error_rate > 5.0:
        st.warning(f"⚠️ High error rate: {error_rate:.1f}% of predictions failing")
    if p99 > P99_ALERT_THRESHOLD_MS:
        st.warning(
            f"⚠️ Latency degradation: p99={p99:.0f}ms "
            f"(threshold {P99_ALERT_THRESHOLD_MS:.0f}ms)"
        )
    if rejected > 0:
        st.warning(f"⚠️ Concurrency cap hit: {rejected} request(s) rejected (429)")
    if error_rate <= 5.0 and p99 <= P99_ALERT_THRESHOLD_MS and rejected == 0:
        st.success("✅ API healthy — no anomalies detected")

    hours, rem = divmod(int(uptime_s), 3600)
    mins = rem // 60
    st.caption(f"Uptime: {hours}h {mins}m")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total predictions", f"{preds.get('total', 0):,}")
        st.metric("Errors", preds.get("errors", 0))
    with col2:
        st.metric("Error rate", f"{error_rate:.2f}%")
        st.metric("Rejected (429)", rejected)
    with col3:
        st.metric("p50 latency", f"{lat.get('p50_ms', 0):.0f}ms")
        st.metric("p99 latency", f"{p99:.0f}ms")


def display_mlflow_panel() -> None:
    """Show the last 5 MLflow training runs from the tracking server."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns/")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "insurance_prediction")

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        st.caption("mlflow not installed — pip install mlflow")
        return

    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None or experiment.lifecycle_stage != "active":
            all_exps = client.search_experiments(
                filter_string=f"name = '{experiment_name}'",
                order_by=["creation_time DESC"],
            )
            active = [e for e in all_exps if e.lifecycle_stage == "active"]
            if not active:
                st.caption(
                    f"No active experiment '{experiment_name}' found. "
                    "Run train.py to create one."
                )
                return
            experiment = active[0]

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=5,
        )

        if not runs:
            st.caption("No training runs recorded yet.")
            return

        rows = []
        for run in runs:
            metrics = run.data.metrics
            val_rmse = metrics.get("val_rmse")
            val_r2 = metrics.get("val_r2")
            rows.append(
                {
                    # BUG-NEW-3 FIX: 6 columns at white-space:nowrap exceed the
                    # 21rem sidebar width.  Streamlit's sidebar container has
                    # overflow:hidden at a higher stacking level so the
                    # themed-table-wrapper's overflow-x:auto scrollbar is
                    # suppressed — right columns become unreachable.
                    # Trim to 4 compact columns that fit comfortably.
                    # Status is always "FINISHED"; Time(s) dropped to save space.
                    "Run ID": run.info.run_id[:8],
                    "RMSE": f"${val_rmse:,.0f}" if val_rmse is not None else "—",
                    "R²": f"{val_r2:.4f}" if val_r2 is not None else "—",
                    "Started": datetime.fromtimestamp(
                        run.info.start_time / 1000
                    ).strftime("%m-%d %H:%M"),
                }
            )

        # BUG-CODE-2 FIX: st.dataframe() / Glide Data Grid does not inherit
        # CSS var() overrides — under OS dark the rows become invisible.
        # render_html_table() emits a themed HTML table that is always readable.
        render_html_table(pd.DataFrame(rows), numeric_cols=[])

        ui_url = os.environ.get(
            "MLFLOW_UI_URL",
            tracking_uri.replace("http://mlflow:5000", "http://localhost:5000"),
        )
        if ui_url.startswith("http"):
            st.markdown(f"[Open MLflow UI ↗]({safe_html(ui_url)})")

    except Exception as exc:
        st.caption(f"MLflow error: {type(exc).__name__}: {str(exc)[:80]}")
        logger.debug(f"MLflow panel error: {exc}", exc_info=True)


def display_admin_dashboard() -> None:
    """Display admin dashboard in sidebar."""
    # BUG-7 FIX: st.sidebar.checkbox was not migrated alongside the What-If
    # multivariate checkbox (BUG-5 in v3.5.11 changelog).  BaseWeb uses
    # data-baseweb="checkbox" for BOTH st.checkbox() and st.toggle(); Styletron
    # injects its dark-theme indicator background after our <style> tag, winning
    # source-order and rendering a dark filled square on OS-dark systems.
    # st.toggle() uses a pill-shaped track with unambiguous visual states and is
    # immune to the indicator CSS collision.
    st.sidebar.toggle(
        "🔧 Admin Panel",
        key="show_admin",
        help="Enable system diagnostics and admin tools",
    )

    if st.session_state.show_admin:
        with st.sidebar.expander("⚙️ System Status", expanded=False):
            is_healthy, model_name, _, pipeline_version, hybrid_version = (
                check_api_health_cached()
            )
            st.write(f"**API Status:** {'✅' if is_healthy else '❌'}")
            st.write(f"**Model:** {model_name or 'N/A'}")
            if pipeline_version:
                st.write(f"**Pipeline v:** {pipeline_version}")
            if hybrid_version:
                st.write(f"**Hybrid v:** {hybrid_version}")

            history = get_history()
            st.write(f"**History Size:** {len(history)}")
            st.write(f"**Max History:** {MAX_HISTORY_SIZE}")
            st.write(
                f"**Validated:** "
                f"{'Yes' if st.session_state.get('history_validated') else 'No'}"
            )

            try:
                client = get_api_client()
                cb = client.circuit_breaker.get_state()
                st.write(f"**Circuit Breaker:** {cb['state']}")
                st.write(f"**Failures:** {cb['failure_count']}")
                if cb["last_failure"]:
                    st.write(f"**Last Failure:** {cb['last_failure'][:19]}")
            except Exception:
                st.write("**Circuit Breaker:** Error")

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🗑️ Clear Cache", width="stretch", key="admin_clear_cache"):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    # BUG-6 FIX: _build_theme_css uses @lru_cache(maxsize=2),
                    # not @st.cache_data, so st.cache_data.clear() never reaches
                    # it.  Call .cache_clear() explicitly so a theme upgrade or
                    # palette change takes effect without a server restart.
                    _build_theme_css.cache_clear()
                    st.success("✅ Cache cleared")
            with col2:
                if st.button("♻️ Run GC", width="stretch", key="admin_gc"):
                    collected = gc.collect()
                    st.success(f"✅ Collected {collected}")
            with col3:
                if st.button("🔓 Reset CB", width="stretch", key="admin_reset_cb"):
                    try:
                        _client = get_api_client()
                        _client.circuit_breaker.reset()
                        st.success("✅ Circuit breaker reset to closed")
                        logger.info("Circuit breaker manually reset via admin panel")
                    except Exception as _e:
                        st.error(f"❌ CB reset failed: {_e}")

        with st.sidebar.expander("📊 Metrics & Alerts", expanded=False):
            display_metrics_panel()

        with st.sidebar.expander("🧪 MLflow Runs", expanded=False):
            display_mlflow_panel()


def render_html_table(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
) -> None:
    """Render a DataFrame as a themed HTML table."""
    if df.empty:
        st.info("No data to display")
        return

    if numeric_cols is None:
        numeric_cols = [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
    numeric_set = set(numeric_cols)

    import html as _html

    def _cell(tag: str, val: str, is_num: bool) -> str:
        cls = ' class="num"' if is_num else ""
        safe = _html.escape(str(val))
        return f"<{tag}{cls}>{safe}</{tag}>"

    header_cells = "".join(
        _cell("th", col, col in numeric_set) for col in df.columns
    )
    rows_html = ""
    for _, row in df.iterrows():
        cells = "".join(
            _cell("td", row[col], col in numeric_set) for col in df.columns
        )
        rows_html += f"<tr>{cells}</tr>"

    table_html = f"""
    <div class="themed-table-wrapper">
        <table class="themed-table">
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


@safe_display
def display_comparison() -> None:
    """Display comparison of multiple predictions."""
    st.subheader("⚖️ Premium Comparison")
    history = get_history()

    if len(history) < 2:
        if len(history) == 1:
            st.info("ℹ️ Make one more prediction to compare scenarios")
            p = history[0]
            st.write(
                f"**Prediction 1:** Age={p.age}, BMI={p.bmi:.1f}, "
                f"Premium=${p.prediction:,.2f}"
            )
        else:
            st.info("ℹ️ Make at least 2 predictions to compare")
        return

    data = []
    for i, p in enumerate(history[-COMPARISON_LIMIT:]):
        data.append(
            {
                "Scenario": i + 1,
                "Age": p.age,
                "BMI": round(p.bmi, 1),
                "Smoker": p.smoker,
                "Premium": p.prediction,
                "Time": p.timestamp.strftime("%H:%M:%S"),
            }
        )

    df = pd.DataFrame(data)
    display_df = df.copy()
    display_df["Premium"] = display_df["Premium"].apply(lambda x: f"${x:,.2f}")
    render_html_table(display_df, numeric_cols=["Scenario", "Age", "BMI"])

    chart_data = pd.DataFrame(
        [
            # BUG-1 FIX: cast Scenario to str so Plotly uses a categorical
            # axis instead of a continuous numeric one.  Integer values produce
            # fractional ticks (0.5, 1, 1.5, 2, 2.5) between bars; string
            # values force one discrete tick per bar.
            {"Scenario": str(i + 1), "Premium": p.prediction}
            for i, p in enumerate(history[-COMPARISON_LIMIT:])
        ]
    )
    fig = px.bar(
        chart_data,
        x="Scenario",
        y="Premium",
        title="Premium Comparison",
        labels={"Premium": "Premium ($)"},
        text="Premium",
        color="Premium",
        color_continuous_scale="Blues",
    )
    # THEME-6: marker_line_color uses _marker_line_color() instead of
    # hardcoded "rgb(8,48,107)" which is near-invisible on dark chart backgrounds.
    fig.update_traces(
        texttemplate="$%{text:,.0f}",
        textposition="outside",
        marker_line_color=_marker_line_color(),
        marker_line_width=1.5,
    )
    # THEME-4/5: _chart_layout() replaces **CHART_LAYOUT_TEMPLATE so that
    # font.color and hoverlabel colours are set via Plotly's layout API.
    # BUG-1 FIX (guard): xaxis type="category" is redundant once Scenario is
    # str, but keeps the axis categorical if the column is ever re-cast elsewhere.
    fig.update_layout(**_chart_layout(showlegend=False, xaxis=dict(type="category")))
    fig.update_xaxes(**CHART_AXIS_CONFIG)
    fig.update_yaxes(**CHART_AXIS_CONFIG)
    st.plotly_chart(fig, config={"displayModeBar": False}, key="comparison_chart")

    if st.button("🗑️ Clear History", key="clear_history_btn"):
        st.session_state.predictions_history = []
        st.session_state.history_validated = False
        st.session_state.last_prediction = None
        st.session_state.last_model_used = None
        st.session_state.last_form_data = None
        st.session_state.export_anchor_key = None
        st.session_state.export_anchor_csv = None
        st.session_state.export_anchor_json = None
        st.rerun()


def _get_analysis_parameters() -> dict[str, dict]:
    """Get parameter configuration for sensitivity analysis."""
    return {
        "age": {
            "values": [18, 21, 26, 30, 40, 50, 55, 60, 65, 70, 75, 80],
            "thresholds": {
                "26": "Insurance transition",
                "50": "Pre-existing conditions",
                "65": "Medicare age",
            },
            "label": "Age (years)",
            "medical_info": (
                "**Age Brackets:** 18-25 (low risk) | 26-49 (stable) | "
                "50-64 (chronic onset) | 65+ (Medicare/high cost)"
            ),
        },
        "bmi": {
            "values": [
                15.0, 17.0, 18.5, 22.0, 25.0, 27.5,
                30.0, 35.0, 40.0, 45.0, 50.0, 55.0,
            ],
            "thresholds": {
                "18.5": "Underweight",
                "25.0": "Overweight",
                "30.0": "Obese",
                "40.0": "Bariatric surgery",
            },
            "label": "BMI (kg/m²)",
            "medical_info": (
                "**BMI Categories:** <18.5 (underweight) | 18.5-25 (normal) | "
                "25-30 (overweight) | 30-40 (obese) | 40+ (severe obesity)"
            ),
        },
        "children": {
            "values": [0, 1, 2, 3, 4, 5, 6, 7, 10],
            "thresholds": {"3": "Large family threshold"},
            "label": "Number of Dependents",
            "medical_info": (
                "**Family Size:** Newborns cost $8-12K/yr | "
                "Multiple children compound costs non-linearly"
            ),
        },
    }


def _run_univariate_analysis(base: Prediction, param: str, param_config: dict) -> None:
    """Run univariate sensitivity analysis with parallelised API calls."""
    import threading as _threading

    values = param_config["values"]
    results = []
    failed_count = 0
    total_values = len(values)

    progress = st.progress(0, text="Starting analysis...")

    _api_lock = _threading.Lock()

    def _call_one(val):
        kwargs = {
            "age": base.age, "sex": base.sex, "bmi": base.bmi,
            "children": base.children, "smoker": base.smoker, "region": base.region,
        }
        kwargs[param] = val
        try:
            form = FormData(**kwargs)
        except ValueError:
            return val, None
        with _api_lock:
            result = calculate_prediction(form, silent=True)
            time.sleep(RATE_LIMIT_SECONDS)
        return val, result

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(4, total_values),
        thread_name_prefix="sweep_uni",
    ) as pool:
        future_map = {pool.submit(_call_one, val): val for val in values}
        done = 0
        for future in concurrent.futures.as_completed(future_map):
            val, result = future.result()
            done += 1
            progress.progress(done / total_values,
                              text=f"Processing {done}/{total_values}...")
            if result:
                results.append({param: val, "premium": result[0]})
            else:
                failed_count += 1

    results.sort(key=lambda r: r[param])
    progress.empty()

    if failed_count:
        st.warning(
            f"⚠️ {failed_count}/{total_values} data points failed "
            "(API errors — see logs)"
        )

    if results:
        df = pd.DataFrame(results)
        baseline_premium = base.prediction
        df["change_pct"] = (
            (df["premium"] - baseline_premium) / baseline_premium * 100
        ).round(1)

        fig = px.line(
            df,
            x=param,
            y="premium",
            markers=True,
            title=f"Premium vs {param_config['label']}",
            labels={param: param_config["label"], "premium": "Premium ($)"},
        )

        # THEME-6: line/marker colours use theme helpers instead of hardcoded hex.
        fig.update_traces(
            line=dict(color=_primary_color(), width=3),
            marker=dict(
                size=10,
                color=_primary_dark_color(),
                line=dict(color=_marker_border_color(), width=2),
            ),
        )

        for threshold_val, label in param_config["thresholds"].items():
            fig.add_vline(
                x=float(threshold_val),
                line_dash="dash",
                line_color=_vline_threshold_color(),
                line_width=2,
                annotation_text=label,
                annotation_position="top",
            )

        fig.add_vline(
            x=getattr(base, param),
            line_dash="dot",
            line_color=_vline_current_color(),
            line_width=2,
            annotation_text="Current",
            annotation_position="bottom",
        )

        # THEME-4/5: _chart_layout() instead of **CHART_LAYOUT_TEMPLATE
        fig.update_layout(**_chart_layout())
        fig.update_xaxes(**CHART_AXIS_CONFIG)
        fig.update_yaxes(**CHART_AXIS_CONFIG)

        st.plotly_chart(
            fig, config={"displayModeBar": False}, key="sensitivity_chart_uni"
        )

        st.markdown("**💡 Medical Cost Insights:**")
        st.info(param_config["medical_info"])

        display_df = df.copy()
        display_df["premium"] = display_df["premium"].apply(lambda x: f"${x:,.2f}")
        display_df["change_pct"] = display_df["change_pct"].apply(
            lambda x: f"{x:+.1f}%"
        )
        display_df.columns = [param_config["label"], "Premium", "% Change"]
        # BUG-CODE-1 FIX: st.dataframe() uses Glide Data Grid which does not
        # inherit CSS var() overrides.  Under OS dark / "Use system setting" the
        # grid renders light text on a white bg (our override) → invisible rows.
        # render_html_table() emits a plain HTML table that uses CSS variables
        # and is theme-aware, matching the multivariate impact table below.
        render_html_table(display_df, numeric_cols=[param_config["label"]])
    else:
        st.error("❌ All data points failed. Check API connectivity.")


def _run_multivariate_analysis(
    base: Prediction, param: str, param_config: dict
) -> None:
    """Run multivariate sensitivity analysis with parallelised API calls."""
    import threading as _threading

    values = param_config["values"]
    results = []
    failed_count = 0
    combinations = [(s, v) for s in ["no", "yes"] for v in values]
    total_runs = len(combinations)

    progress = st.progress(0, text="Starting multivariate analysis...")

    _api_lock = _threading.Lock()

    def _call_one(smoker_status, val):
        kwargs = {
            "age": base.age, "sex": base.sex, "bmi": base.bmi,
            "children": base.children, "smoker": smoker_status, "region": base.region,
        }
        kwargs[param] = val
        try:
            form = FormData(**kwargs)
        except ValueError:
            return smoker_status, val, None
        with _api_lock:
            result = calculate_prediction(form, silent=True)
            time.sleep(RATE_LIMIT_SECONDS)
        return smoker_status, val, result

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(4, total_runs),
        thread_name_prefix="sweep_multi",
    ) as pool:
        future_map = {
            pool.submit(_call_one, s, v): (s, v) for s, v in combinations
        }
        done = 0
        for future in concurrent.futures.as_completed(future_map):
            smoker_status, val, result = future.result()
            done += 1
            progress.progress(done / total_runs,
                              text=f"Processing {done}/{total_runs}...")
            if result:
                results.append(
                    {param: val, "premium": result[0], "smoker": smoker_status}
                )
            else:
                failed_count += 1

    results.sort(key=lambda r: (r["smoker"], r[param]))
    progress.empty()

    if failed_count:
        st.warning(
            f"⚠️ {failed_count}/{total_runs} data points failed "
            "(API errors — see logs)"
        )

    if results:
        df = pd.DataFrame(results)

        fig = px.line(
            df,
            x=param,
            y="premium",
            color="smoker",
            markers=True,
            title=f"Premium vs {param_config['label']} (Smoking Interaction)",
            labels={
                param: param_config["label"],
                "premium": "Premium ($)",
                "smoker": "Smoker",
            },
            color_discrete_map={"no": "#48bb78", "yes": "#f56565"},
        )

        # THEME-6: marker border uses theme helper
        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=10, line=dict(color=_marker_border_color(), width=2)),
        )

        for threshold_val, label in param_config["thresholds"].items():
            fig.add_vline(
                x=float(threshold_val),
                line_dash="dash",
                line_color=_vline_threshold_color(),
                line_width=2,
                annotation_text=label,
                annotation_position="top",
            )

        # THEME-4/5: _chart_layout() instead of **CHART_LAYOUT_TEMPLATE
        # BUG-3 FIX: legend y=1.02 places the legend *above* the plot area in
        # Plotly's paper coordinate system.  Without an explicit margin.t the
        # canvas is not tall enough for y>1, so Plotly clips the legend back
        # inside the top-right corner of the chart.  margin.t=100 gives the
        # required canvas headroom.
        fig.update_layout(
            **_chart_layout(
                margin=dict(t=100, b=70, l=60, r=20),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                )
            )
        )
        fig.update_xaxes(**CHART_AXIS_CONFIG)
        fig.update_yaxes(**CHART_AXIS_CONFIG)

        st.plotly_chart(
            fig, config={"displayModeBar": False}, key="sensitivity_chart_multi"
        )

        st.markdown("**📊 Smoking Impact Analysis:**")
        impact_df = df.pivot_table(index=param, columns="smoker", values="premium")
        if "no" in impact_df.columns and "yes" in impact_df.columns:
            impact_df["increase"] = (
                (impact_df["yes"] - impact_df["no"]) / impact_df["no"] * 100
            ).round(1)
            impact_df["increase"] = impact_df["increase"].apply(lambda x: f"+{x}%")

            # st.dataframe() uses Glide Data Grid which does not inherit our
            # injected CSS variables.  In dark mode with config.toml base="light"
            # it renders dark cell backgrounds with dark text — invisible.
            # Render as a plain HTML table using CSS variables instead so both
            # themes display correctly without any JavaScript detection.
            param_label = safe_html(param_config.get("label", param))

            header_cells = (
                f"<th>{safe_html(param_label)}</th>"
                f"<th>Non-smoker ($)</th>"
                f"<th>Smoker ($)</th>"
                f"<th>Smoking premium</th>"
            )
            rows_html = ""
            for idx_val, row in impact_df.iterrows():
                no_val = f"${row['no']:,.0f}" if pd.notna(row.get("no")) else "—"
                yes_val = f"${row['yes']:,.0f}" if pd.notna(row.get("yes")) else "—"
                inc_val = safe_html(str(row.get("increase", "—")))
                rows_html += (
                    f"<tr>"
                    f"<td>{safe_html(str(idx_val))}</td>"
                    f"<td>{safe_html(no_val)}</td>"
                    f"<td>{safe_html(yes_val)}</td>"
                    f"<td style='color: var(--danger); font-weight: 600;'>{inc_val}</td>"
                    f"</tr>"
                )

            st.markdown(
                f"""
                <style>
                .smoking-impact-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.875rem;
                    margin-top: 0.5rem;
                    background: var(--bg-primary);
                    border-radius: 8px;
                    overflow: hidden;
                    border: 1px solid var(--border-light);
                }}
                .smoking-impact-table th {{
                    background: var(--bg-tertiary);
                    color: var(--text-primary);
                    padding: 0.6rem 1rem;
                    text-align: left;
                    font-weight: 600;
                    border-bottom: 2px solid var(--border-medium);
                }}
                .smoking-impact-table td {{
                    padding: 0.5rem 1rem;
                    color: var(--text-primary);
                    border-bottom: 1px solid var(--border-light);
                }}
                .smoking-impact-table tr:last-child td {{
                    border-bottom: none;
                }}
                .smoking-impact-table tr:hover td {{
                    background: var(--overlay-light);
                }}
                </style>
                <table class="smoking-impact-table">
                  <thead><tr>{header_cells}</tr></thead>
                  <tbody>{rows_html}</tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.error("❌ All data points failed. Check API connectivity.")


@safe_display
def display_sensitivity_analysis() -> None:
    """Display sensitivity analysis tab."""
    st.subheader("🔬 What-If Analysis")
    history = get_history()

    if not history:
        st.info("ℹ️ Make a prediction first")
        return

    base = history[-1]
    st.write(
        f"**Base:** Age={base.age}, BMI={base.bmi:.1f}, "
        f"Smoker={base.smoker}, Premium=${base.prediction:,.2f}"
    )

    param_config = _get_analysis_parameters()

    col1, col2 = st.columns([3, 1])
    with col1:
        param = st.selectbox(
            "Vary parameter:", list(param_config.keys()), key="sensitivity_param"
        )
    with col2:
        # BUG-5 FIX: st.checkbox renders a dark filled square on OS-dark systems
        # because BaseWeb's indicator div shares data-baseweb="checkbox" with
        # st.toggle, making CSS targeting impossible without also affecting toggles.
        # st.toggle produces a pill shape with unambiguous visual state.
        multivariate = st.toggle(
            "Multivariate",
            help="Analyze interaction with smoking",
            key="multivariate_check",
        )

    if st.button("🔍 Run Analysis", type="primary", key="run_sensitivity_btn"):
        if not rate_limit_check(namespace="analysis"):
            st.warning("⏱️ Please wait")
            return

        if multivariate:
            _run_multivariate_analysis(base, param, param_config[param])
        else:
            _run_univariate_analysis(base, param, param_config[param])


@safe_display
def display_batch_predictions() -> None:
    """Display batch prediction interface with memory safety."""
    st.subheader("📁 Batch Predictions")

    timeout_minutes = MAX_BATCH_TIME_SECONDS // 60

    # THEME-3 FIX: replaced hardcoded #e6f2ff / #f0e6ff / #2d3748 with CSS
    # variables so the banner renders correctly in both light and dark themes.
    # BUG-2 FIX: added "Max {MAX_FILE_SIZE_MB}MB file size" to the Limits line
    # so the displayed constraints match the validate_upload() rejection threshold
    # (previously the widget showed Streamlit's native "200MB" label while the
    # server would reject anything over 5MB — confusing silent failure for users).
    st.markdown(
        f"""
    <div style="background: var(--bg-secondary);
                border: 1px solid var(--border-light);
                padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
        <p style="margin: 0; color: var(--text-secondary);">
            <strong>📋 Required CSV Columns:</strong>
            age, sex, bmi, children, smoker, region<br>
            <strong>⚡ Performance:</strong>
            Optimized for large datasets (10-100x faster than v1)<br>
            <strong>⏱️ Limits:</strong>
            Max {MAX_BATCH_ROWS:,} rows, {timeout_minutes} min timeout,
            {MAX_RESULTS_MEMORY:,} results max,
            {MAX_FILE_SIZE_MB}MB max file size
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    file = st.file_uploader("Choose CSV", type=["csv"], key="batch_file")

    if file:
        is_valid, error = validate_upload(file)
        if not is_valid:
            st.error(f"❌ {error}")
            return

        try:
            file_content = file.getvalue()
            file_hash = hashlib.md5(file_content).hexdigest()
            file_id = f"{file.name}_{file_hash[:16]}"

            if st.session_state.uploaded_file_id != file_id:
                df = pd.read_csv(BytesIO(file_content))
                st.session_state.uploaded_df = df
                st.session_state.uploaded_file_id = file_id
                logger.info(f"Uploaded new file: {file.name}")
            else:
                df = st.session_state.uploaded_df
                logger.info(f"Using cached file: {file.name}")

            required = ["age", "sex", "bmi", "children", "smoker", "region"]

            if not all(col in df.columns for col in required):
                st.error(f"❌ Missing columns. Need: {', '.join(required)}")
                return

            if check_csv_injection(df):
                st.error("❌ Potential CSV injection detected")
                return

            if len(df) > MAX_BATCH_ROWS:
                st.error(
                    f"❌ File too large. Max {MAX_BATCH_ROWS:,} rows, "
                    f"found {len(df):,}"
                )
                return

            st.success(f"✅ Loaded {len(df)} records")
            # BUG-CODE-2b FIX: st.dataframe() Glide Data Grid invisible in dark OS.
            render_html_table(df.head(), numeric_cols=["age", "bmi", "children"])

            estimated_mb = (len(df) * BYTES_PER_ROW_ESTIMATE) / (1024 * 1024)
            if estimated_mb > 100:
                st.warning(
                    f"⚠️ Large dataset: ~{estimated_mb:.0f}MB estimated memory usage"
                )
                proceed = st.checkbox(
                    "I understand this may take time and memory", key="memory_confirm"
                )
                if not proceed:
                    return
            elif len(df) > 100:
                st.warning(f"⚠️ {len(df)} records may take time")
                proceed = st.checkbox(
                    "Proceed with batch processing", key="batch_confirm"
                )
                if not proceed:
                    return

            if st.button("🚀 Process Batch", key="batch_process_btn"):
                if not rate_limit_check(namespace="default"):
                    st.warning("⏱️ Please wait")
                    return

                client = get_api_client()
                records = df.to_dict("records")

                max_results = min(len(records), MAX_RESULTS_MEMORY)
                if len(records) > max_results:
                    st.warning(
                        f"⚠️ Processing limited to {max_results:,} rows "
                        "(memory safety)"
                    )
                records_to_send = records[:max_results]

                start_time = time.time()

                with st.status("Processing batch...", expanded=True) as status:
                    try:
                        batch_payload = [
                            {
                                "age": _safe_int(r.get("age"), 30),
                                "sex": _safe_str(r.get("sex"), "male"),
                                "bmi": _safe_float(r.get("bmi"), 25.0),
                                "children": _safe_int(r.get("children"), 0),
                                "smoker": _safe_str(r.get("smoker"), "no"),
                                "region": _safe_str(r.get("region"), "northeast"),
                            }
                            for r in records_to_send
                        ]

                        st.write(
                            f"⚡ Sending {len(batch_payload):,} records to "
                            "batch endpoint..."
                        )
                        batch_response = client.predict_batch(batch_payload)

                        results = []
                        for item in batch_response.get("results", []):
                            idx = item.get("index", 0)
                            original_row = (
                                records_to_send[idx]
                                if idx < len(records_to_send)
                                else {}
                            )
                            results.append(
                                {
                                    **original_row,
                                    "predicted_cost": item.get("prediction"),
                                    "model": item.get("model_used", "unknown"),
                                    "status": item.get("status", "unknown"),
                                }
                            )

                        n_success = batch_response.get("successful", 0)
                        n_failed = batch_response.get("failed", 0)
                        elapsed = time.time() - start_time
                        st.write(
                            f"✅ Batch complete in {elapsed:.1f}s | "
                            f"{n_success} successful | {n_failed} failed"
                        )
                        status.update(label="Processing complete!", state="complete")
                        processing_status = "complete"

                    except Exception as batch_err:
                        logger.warning(
                            f"Batch endpoint failed ({batch_err}), "
                            "falling back to row-by-row processing"
                        )
                        st.warning(
                            "⚠️ Batch endpoint unavailable — switching to "
                            "row-by-row mode"
                        )
                        results = []
                        processing_status = "complete"
                        progress = st.progress(
                            0, text="Starting fallback processing..."
                        )
                        total_to_process = len(records_to_send)

                        for idx, row_dict in enumerate(records_to_send):
                            if time.time() - start_time > MAX_BATCH_TIME_SECONDS:
                                st.warning(f"⏱️ Timeout after {idx} records")
                                processing_status = "timeout"
                                break

                            results.append(
                                process_batch_row_safe(row_dict, client)
                            )

                            if (
                                (idx + 1) % BATCH_UPDATE_INTERVAL == 0
                                or idx == total_to_process - 1
                            ):
                                progress.progress(
                                    (idx + 1) / total_to_process,
                                    text=f"Processing {idx + 1}/{total_to_process}...",
                                )

                        progress.empty()
                        if processing_status == "timeout":
                            status.update(
                                label="Processing stopped (timeout)", state="error"
                            )
                        else:
                            status.update(
                                label="Processing complete (fallback)!",
                                state="complete",
                            )

                results_df = pd.DataFrame(results)

                success = len(
                    [r for r in results if r.get("status") == "success"]
                )

                if processing_status == "timeout":
                    st.warning(
                        f"⚠️ Processed {len(results)}/{len(df)} records "
                        f"({success} successful)"
                    )
                elif len(records) > max_results:
                    st.warning(
                        f"⚠️ Processed {len(results)}/{len(df)} records "
                        "(memory limit)"
                    )
                else:
                    st.success(
                        f"✅ Processed {len(results)} records ({success} successful)"
                    )

                # BUG-CODE-2c FIX: same Glide Data Grid visibility issue.
                render_html_table(results_df, numeric_cols=["predicted_cost"])

                batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_str = results_df.to_csv(index=False)
                filename_str = f"results_{batch_ts}.csv"

                if len(csv_str.encode("utf-8")) <= MAX_BATCH_CSV_BYTES:
                    st.session_state.batch_result_csv = csv_str
                    st.session_state.batch_result_filename = filename_str
                else:
                    st.warning(
                        f"⚠️ Result ({len(csv_str.encode('utf-8')) / (1024 * 1024):.1f} MB) "
                        f"exceeds {MAX_BATCH_CSV_BYTES // 1024 // 1024} MB cache limit. "
                        "Download immediately — it will not persist on rerun."
                    )
                    st.download_button(
                        "📥 Download Results (large — one-time)",
                        csv_str,
                        filename_str,
                        "text/csv",
                        key="batch_download_large_btn",
                    )
                    st.session_state.batch_result_csv = None
                    st.session_state.batch_result_filename = None

            if st.session_state.get("batch_result_csv") is not None:
                st.download_button(
                    "📥 Download Results",
                    st.session_state.batch_result_csv,
                    st.session_state.batch_result_filename,
                    "text/csv",
                    key="batch_download_btn",
                )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            logger.error(f"Batch processing error: {e}", exc_info=True)


@safe_display
def display_model_stats() -> None:
    """Display model statistics."""
    st.subheader("📊 Statistics")

    history = get_history()
    df = get_history_dataframe(history)

    if df.empty:
        st.info("ℹ️ No predictions yet")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", len(df))
    col2.metric("Avg", f"${df['prediction'].mean():,.0f}")
    col3.metric("Min", f"${df['prediction'].min():,.0f}")
    col4.metric("Max", f"${df['prediction'].max():,.0f}")

    with st.expander("📈 Premium Trend", expanded=False):
        trend_df = df[["timestamp", "prediction"]].copy()
        trend_df["timestamp"] = pd.to_datetime(trend_df["timestamp"])
        trend_df["prediction_num"] = range(1, len(trend_df) + 1)

        fig = px.line(
            trend_df,
            x="prediction_num",
            y="prediction",
            markers=True,
            title="Premium History",
            labels={"prediction_num": "Prediction #", "prediction": "Premium ($)"},
        )
        # THEME-6: theme-aware line/marker colours
        fig.update_traces(
            line=dict(color=_primary_color(), width=3),
            marker=dict(
                size=8,
                color=_primary_dark_color(),
                line=dict(color=_marker_border_color(), width=2),
            ),
        )
        # THEME-4/5: _chart_layout()
        fig.update_layout(**_chart_layout())
        fig.update_xaxes(**CHART_AXIS_CONFIG)
        fig.update_yaxes(**CHART_AXIS_CONFIG)
        st.plotly_chart(
            fig, config={"displayModeBar": False}, key="premium_trend_chart"
        )

    with st.expander("📊 Model Usage", expanded=False):
        counts = df["model_used"].value_counts()
        fig = px.pie(
            values=counts.values,
            names=counts.index,
            title="Predictions by Model",
            hole=0.4,
            # Blues_r starts at near-black navy for single-slice charts.
            # Use the app's explicit primary palette for readable fills.
            color_discrete_sequence=["#5a67d8", "#7c8aed", "#4c51bf", "#a3b0f5", "#3c4099"],
        )
        # THEME-6: theme-aware marker line
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color=_marker_border_color(), width=2)),
        )
        # THEME-4/5: _chart_layout() — exclude hovermode for pie charts
        pie_layout = {
            k: v for k, v in _chart_layout().items() if k != "hovermode"
        }
        fig.update_layout(
            **pie_layout,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5
            ),
        )
        st.plotly_chart(
            fig, config={"displayModeBar": False}, key="model_usage_chart"
        )

    with st.expander("⚠️ Risk Distribution", expanded=False):
        if not history:
            st.info("No predictions available")
            return

        risks = [get_risk_level(calculate_risk_score(p)) for p in history]
        risk_df = pd.DataFrame({"Risk": risks})
        counts = (
            risk_df["Risk"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
        )

        fig = px.bar(
            x=counts.index,
            y=counts.values,
            title="Risk Level Distribution",
            color=counts.index,
            color_discrete_map={
                "Low": "#48bb78",
                "Medium": "#ed8936",
                "High": "#f56565",
            },
            labels={"x": "Risk Level", "y": "Count"},
            text=counts.values,
        )
        # THEME-6: theme-aware marker line
        fig.update_traces(
            textposition="outside",
            marker_line_color=_marker_line_color(),
            marker_line_width=1.5,
        )
        # THEME-4/5: _chart_layout()
        fig.update_layout(**_chart_layout(showlegend=False))
        fig.update_xaxes(**CHART_AXIS_CONFIG)
        fig.update_yaxes(**CHART_AXIS_CONFIG)
        st.plotly_chart(
            fig, config={"displayModeBar": False}, key="risk_dist_chart"
        )

    with st.expander("📋 Recent Predictions", expanded=False):
        recent = df.tail(10)[
            ["age", "bmi", "smoker", "prediction", "model_used"]
        ].copy()
        recent["prediction"] = recent["prediction"].apply(lambda x: f"${x:,.2f}")
        recent.columns = ["Age", "BMI", "Smoker", "Premium", "Model"]
        render_html_table(recent, numeric_cols=["Age", "BMI"])


@safe_display
def display_export_options() -> None:
    """Display export options (BUG-5 stable anchor fix retained from v3.5.6)."""
    st.subheader("📤 Export")
    history = get_history()

    if not history:
        st.info("ℹ️ Make a prediction first")
        return

    anchor_key = st.session_state.get("export_anchor_key")
    anchor_csv = st.session_state.get("export_anchor_csv")
    anchor_json = st.session_state.get("export_anchor_json")

    last = history[-1]
    export_timestamp = last.timestamp.strftime("%Y%m%d_%H%M%S")

    if anchor_key and anchor_csv and anchor_json:
        csv = anchor_csv
        json_data = anchor_json
        export_timestamp = anchor_key
    else:
        df_last = pd.DataFrame([last.to_dict()])
        csv = df_last.to_csv(index=False)
        json_data = json.dumps(last.to_dict(), indent=2)

    st.markdown("**Last Prediction**")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📊 Download CSV",
            csv,
            f"prediction_{export_timestamp}.csv",
            "text/csv",
            key="export_last_csv_btn",
        )
    with col2:
        st.download_button(
            "📋 Download JSON",
            json_data,
            f"prediction_{export_timestamp}.json",
            "application/json",
            key="export_last_json_btn",
        )

    if len(history) > 1:
        st.markdown("---")
        export_limit = min(len(history), MAX_EXPORT_RECORDS)
        st.markdown(
            f"**History** ({len(history)} total, exporting last {export_limit})"
        )

        if export_limit > EXPORT_WARNING_THRESHOLD:
            st.warning(f"⚠️ Exporting {export_limit} records")

        all_data = [p.to_dict() for p in history[-export_limit:]]
        df_all = pd.DataFrame(all_data)

        history_ts = history[-export_limit].timestamp.strftime("%Y%m%d_%H%M%S")
        newest_ts = history[-1].timestamp.strftime("%Y%m%d_%H%M%S")
        _hist_cache_key = f"export_hist_csv_{history_ts}_{newest_ts}"
        _hist_json_key = f"export_hist_json_{history_ts}_{newest_ts}"

        if _hist_cache_key not in st.session_state:
            st.session_state[_hist_cache_key] = df_all.to_csv(index=False)
        if _hist_json_key not in st.session_state:
            st.session_state[_hist_json_key] = json.dumps(all_data, indent=2)

        csv_all = st.session_state[_hist_cache_key]
        json_all = st.session_state[_hist_json_key]

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📊 Download All CSV",
                csv_all,
                f"history_{history_ts}.csv",
                "text/csv",
                key="export_all_csv_btn",
            )
        with col2:
            st.download_button(
                "📋 Download All JSON",
                json_all,
                f"history_{history_ts}.json",
                "application/json",
                key="export_all_json_btn",
            )


# =============================================================================
# MAIN APPLICATION
# =============================================================================


def main() -> None:
    """Main application entry point."""
    try:
        _log_optional_deps_once()
        # BUG-7 FIX: init_session_state() must precede apply_enhanced_ui() so
        # dark_mode is guaranteed in session_state before CSS is generated.
        init_session_state()
        apply_enhanced_ui()

        if not API_BASE_URL:
            st.error("⚠️ **Configuration Error**: API_URL not set")
            st.markdown(
                """
            **Setup Instructions:**
            - **Local Development**: Add `API_URL=http://localhost:8000` to `.env` file
            - **Production**: Configure `API_URL` in your deployment environment
            """
            )
            st.stop()

        display_header()

        is_healthy = display_api_status()
        if not is_healthy:
            st.warning("⚠️ API is currently unavailable. Some features may not work.")

        create_input_form()
        display_admin_dashboard()

        if st.session_state.advanced_mode:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
                [
                    "🔮 Predict",
                    "⚖️ Compare",
                    "🔬 What-If",
                    "📁 Batch",
                    "📊 Stats",
                    "📤 Export",
                ]
            )
        else:
            tab1 = st.container()
            tab2 = tab3 = tab4 = tab5 = tab6 = None

        with tab1:
            if st.session_state.last_prediction is not None:
                display_prediction_result(
                    st.session_state.last_prediction, st.session_state.last_model_used
                )

                st.markdown("---")

                history = get_history()
                if history:
                    try:
                        last_pred = history[-1]
                        risk_score = calculate_risk_score(last_pred)
                        risk_level = get_risk_level(risk_score)

                        risk_colors = {
                            "Low": ("#48bb78", "✓"),
                            "Medium": ("#ed8936", "⚠"),
                            "High": ("#f56565", "⚠⚠"),
                        }

                        risk_color, risk_icon = risk_colors[risk_level]

                        # THEME-2 FIX: replaced hardcoded `background: white`
                        # and `color: #4a5568` with CSS variables so the card
                        # renders correctly in dark mode.
                        st.markdown(
                            f"""
                        <div style="background: var(--bg-primary);
                             border: 1px solid var(--border-light);
                             padding: 1.5rem; border-radius: 12px;
                             border-left: 5px solid {safe_html(risk_color)};
                             box-shadow: var(--shadow-sm);">
                            <h4 style="color: {safe_html(risk_color)};
                                       margin: 0 0 0.5rem 0;">
                                {safe_html(risk_icon)} {safe_html(risk_level)} Risk Level
                                ({safe_html(str(risk_score))}/100)
                            </h4>
                            <p style="color: var(--text-secondary); margin: 0;">
                                Based on smoking status, BMI, age, and family size
                            </p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        with st.expander("📋 Risk Factor Breakdown"):
                            factors = []
                            if last_pred.smoker == "yes":
                                factors.append("• Smoking: +40 points")
                            if last_pred.bmi > BMI_OVERWEIGHT:
                                factors.append(
                                    f"• Obesity (BMI {last_pred.bmi:.1f}): +20 points"
                                )
                            elif last_pred.bmi > BMI_NORMAL:
                                factors.append(
                                    f"• Overweight (BMI {last_pred.bmi:.1f}): +10 points"
                                )
                            if last_pred.age > 65:
                                factors.append(f"• Age {last_pred.age}: +25 points")
                            elif last_pred.age > 50:
                                factors.append(f"• Age {last_pred.age}: +15 points")
                            if last_pred.children > 3:
                                factors.append(
                                    f"• Large family "
                                    f"({last_pred.children} children): +10 points"
                                )

                            if factors:
                                st.markdown("\n".join(factors))
                            else:
                                st.success("No major risk factors identified")
                    except (IndexError, AttributeError) as e:
                        st.warning("⚠️ Unable to display risk assessment")
                        logger.error(f"Risk display error: {e}")
            else:
                display_welcome_message()

        if st.session_state.advanced_mode and tab2 is not None:
            with tab2:
                if not is_healthy:
                    st.warning("⚠️ API unavailable - cannot make new predictions")
                display_comparison()

        if st.session_state.advanced_mode and tab3 is not None:
            with tab3:
                if not is_healthy:
                    st.warning("⚠️ API unavailable - cannot run analysis")
                display_sensitivity_analysis()

        if st.session_state.advanced_mode and tab4 is not None:
            with tab4:
                if not is_healthy:
                    st.warning("⚠️ API unavailable - cannot process batch")
                display_batch_predictions()

        if st.session_state.advanced_mode and tab5 is not None:
            with tab5:
                display_model_stats()

        if st.session_state.advanced_mode and tab6 is not None:
            with tab6:
                display_export_options()

        display_footer()

        if st.session_state.show_admin and os.getenv("ADMIN_MODE", "").lower() == "true":
            st.markdown("---")
            with st.expander("🛠 Debug Information", expanded=False):
                history = get_history()

                debug_info = {
                    "Version": APP_VERSION,
                    "History Size": len(history),
                    "History Validated": st.session_state.get(
                        "history_validated", False
                    ),
                    "Last Request Time (predict)": st.session_state.get(
                        "last_request_time_predict", 0
                    ),
                    "Uploaded File ID": st.session_state.uploaded_file_id,
                    "Has Last Prediction": st.session_state.last_prediction is not None,
                    "Advanced Mode": st.session_state.advanced_mode,
                    "API Base URL": _mask_url_credentials(API_BASE_URL),
                    "Max Batch Rows": MAX_BATCH_ROWS,
                    "Max Results Memory": MAX_RESULTS_MEMORY,
                    "Health Check TTL": HEALTH_CHECK_TTL,
                    "PSUTIL Available": PSUTIL_AVAILABLE,
                    "Prediction Min ($)": PREDICTION_MIN,
                    "Prediction Max ($)": PREDICTION_MAX,
                    "Batch CSV Cache Cap (MB)": MAX_BATCH_CSV_BYTES // 1024 // 1024,
                    "Active Theme": _get_theme(),
                    "Theme Detection": "CSS var(--background-color) native mapping",
                }
                st.json(debug_info)

                if history:
                    st.markdown(f"**History ({len(history)} items):**")
                    for i, pred in enumerate(history[:5]):
                        st.write(
                            f"{i+1}. Age={pred.age}, BMI={pred.bmi:.1f}, "
                            f"Premium=${pred.prediction:,.2f}"
                        )

    except ValueError as e:
        st.error(f"⚠️ Configuration error: {str(e)}")
        logger.critical(f"Config error: {e}")
    except Exception as e:
        st.error("⚠️ Unexpected error occurred. Please refresh the page.")
        logger.critical(f"Critical error: {e}", exc_info=True)

        if st.session_state.get("show_admin", False) and os.getenv("ADMIN_MODE", "").lower() == "true":
            with st.expander("🔍 Error Details (Admin Only)"):
                st.exception(e)
    finally:
        pass


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

# st.set_page_config MUST be the very first Streamlit command in the script.
# Placing it here (module level, not inside __main__) guarantees this
# regardless of any future module-level st.* additions (Issue 19).
st.set_page_config(
    page_title="Insurance Cost Predictor Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

if __name__ == "__main__":
    main()