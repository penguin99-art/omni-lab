"""Web tools: search and fetch."""

from __future__ import annotations

from edge_agent.tools import tool


@tool("搜索互联网，返回搜索结果摘要")
def web_search(query: str) -> str:
    """Search via DuckDuckGo. No API key required."""
    try:
        from duckduckgo_search import DDGS
        results = DDGS().text(query, max_results=5)
        if not results:
            return "未找到相关结果。"
        lines = []
        for r in results:
            lines.append(f"**{r.get('title', '')}**")
            lines.append(r.get("body", ""))
            lines.append(r.get("href", ""))
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"搜索失败: {e}"


@tool("抓取网页内容，将 HTML 转为纯文本")
def web_fetch(url: str) -> str:
    """Fetch a URL and return text content."""
    try:
        import httpx
        resp = httpx.get(url, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        if "html" in ct:
            try:
                from lxml import html as lhtml
                doc = lhtml.fromstring(resp.text)
                for tag in doc.iter("script", "style", "nav", "footer", "header"):
                    tag.getparent().remove(tag)
                text = doc.text_content()
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                return "\n".join(lines[:200])
            except Exception:
                return resp.text[:5000]
        return resp.text[:5000]
    except Exception as e:
        return f"抓取失败: {e}"
