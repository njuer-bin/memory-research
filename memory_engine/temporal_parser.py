from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from calendar import monthrange


@dataclass
class TemporalInfo:
    text: str = ""
    start: int | None = None
    end: int | None = None
    granularity: str = "point"
    relation: str = "at"
    confidence: float = 1.0

    @property
    def timestamp(self) -> int | None:
        return self.start


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _base(ts: int) -> datetime:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)


def normalize_temporal(text: str, reference_ts: int) -> TemporalInfo:
    """把常见中文时间表达归一化为 [start, end] 毫秒区间。

    这是无模型 deterministic parser，目标是让 current/history/event ordering
    有稳定的结构化信号，而不是追求完整自然语言时间理解。
    """
    if not text:
        return TemporalInfo()

    base = _base(reference_ts)
    year = base.year

    # 绝对日期：2026-03-12 / 2026年3月12日
    m = re.search(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})日?", text)
    if m:
        y, mo, d = map(int, m.groups())
        dt = datetime(y, mo, d, tzinfo=timezone.utc)
        start = _ms(dt)
        return TemporalInfo(m.group(0), start, start + 86_399_999, "day", "at")

    # 月份：2026年3月
    m = re.search(r"(20\d{2})年(\d{1,2})月", text)
    if m:
        y, mo = map(int, m.groups())
        dt = datetime(y, mo, 1, tzinfo=timezone.utc)
        end = datetime(y, mo, monthrange(y, mo)[1], 23, 59, 59, 999000, tzinfo=timezone.utc)
        return TemporalInfo(m.group(0), _ms(dt), _ms(end), "month", "at")

    # 相对年份
    for phrase, delta in (("前年", -2), ("去年", -1), ("明年", 1)):
        if phrase in text:
            y = year + delta
            start = _ms(datetime(y, 1, 1, tzinfo=timezone.utc))
            end = _ms(datetime(y, 12, 31, 23, 59, 59, 999000, tzinfo=timezone.utc))
            relation = "before" if delta < 0 else "after"
            return TemporalInfo(phrase, start, end, "year", relation, 0.95)

    if "今年" in text:
        start = _ms(datetime(year, 1, 1, tzinfo=timezone.utc))
        end = _ms(datetime(year, 12, 31, 23, 59, 59, 999000, tzinfo=timezone.utc))
        return TemporalInfo("今年", start, end, "year", "at", 0.98)

    if "上个月" in text or "上月" in text:
        y, mo = year, base.month - 1
        if mo == 0:
            y, mo = y - 1, 12
        start = _ms(datetime(y, mo, 1, tzinfo=timezone.utc))
        end = _ms(datetime(y, mo, monthrange(y, mo)[1], 23, 59, 59, 999000, tzinfo=timezone.utc))
        return TemporalInfo("上个月", start, end, "month", "before", 0.95)

    if "这个月" in text or "本月" in text:
        start = _ms(datetime(year, base.month, 1, tzinfo=timezone.utc))
        end = _ms(datetime(year, base.month, monthrange(year, base.month)[1], 23, 59, 59, 999000, tzinfo=timezone.utc))
        return TemporalInfo("本月", start, end, "month", "at", 0.98)

    if "昨天" in text:
        day = base - timedelta(days=1)
        start = _ms(datetime(day.year, day.month, day.day, tzinfo=timezone.utc))
        return TemporalInfo("昨天", start, start + 86_399_999, "day", "before", 0.98)

    if "今天" in text:
        start = _ms(datetime(year, base.month, base.day, tzinfo=timezone.utc))
        return TemporalInfo("今天", start, start + 86_399_999, "day", "at", 0.98)

    if "明天" in text:
        day = base + timedelta(days=1)
        start = _ms(datetime(day.year, day.month, day.day, tzinfo=timezone.utc))
        return TemporalInfo("明天", start, start + 86_399_999, "day", "after", 0.98)

    if "以前" in text or "之前" in text:
        return TemporalInfo("以前" if "以前" in text else "之前", None, reference_ts, "open", "before", 0.85)

    if "之后" in text or "后来" in text:
        return TemporalInfo("之后" if "之后" in text else "后来", reference_ts, None, "open", "after", 0.85)

    if "最近" in text:
        return TemporalInfo("最近", reference_ts - 30 * 86_400_000, reference_ts, "range", "near", 0.8)

    return TemporalInfo()


def temporal_filter(info: TemporalInfo, timestamp: int) -> bool:
    if info.start is not None and timestamp < info.start:
        return False
    if info.end is not None and timestamp > info.end:
        return False
    return True
