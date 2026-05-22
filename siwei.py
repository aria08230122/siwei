#!/usr/bin/env python3
"""
siwei - 本地中间层脚本

职责:
  1. 本地规则引擎生成思维链 (CoT)，不额外调用任何 API
  2. 通过 HTTP/SSE 对接 OB 的 MCP 接口拉取记忆
  3. 对接第三方 Claude API 站点 (OpenAI 兼容格式)
  4. 人格模板从本地 YAML 文件加载

设计目标: 仅依赖 requests + pyyaml，可在 Termux 与电脑终端直接运行。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("缺少依赖 pyyaml，请运行: pip install pyyaml requests")

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("缺少依赖 requests，请运行: pip install pyyaml requests")


# --------------------------------------------------------------------------- #
# 配置加载 (支持 ${ENV_VAR} 替换)
# --------------------------------------------------------------------------- #
_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_env(value: Any) -> Any:
    """递归地把字符串里的 ${VAR} 替换成环境变量值。"""
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            name = m.group(1)
            default = ""
            if ":-" in name:  # 支持 ${VAR:-default}
                name, default = name.split(":-", 1)
            return os.environ.get(name, default)
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        sys.exit(f"找不到配置文件: {path}\n请复制 config.example.yaml 为 config.yaml 并填写。")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _expand_env(raw)


# --------------------------------------------------------------------------- #
# 人格模板
# --------------------------------------------------------------------------- #
@dataclass
class Persona:
    name: str
    system_prompt: str
    description: str = ""
    style: str = ""
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, persona_dir: str | Path, name: str) -> "Persona":
        persona_dir = Path(persona_dir)
        candidate = persona_dir / f"{name}.yaml"
        if not candidate.exists():
            candidate = persona_dir / f"{name}.yml"
        if not candidate.exists():
            available = sorted(
                p.stem for p in persona_dir.glob("*.y*ml")
            ) if persona_dir.exists() else []
            sys.exit(
                f"找不到人格模板 '{name}' (目录: {persona_dir})。"
                + (f" 可用: {', '.join(available)}" if available else "")
            )
        with candidate.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(
            name=data.get("name", name),
            system_prompt=data.get("system_prompt", "").strip(),
            description=data.get("description", ""),
            style=data.get("style", ""),
            raw=data,
        )

    def build_system_prompt(self) -> str:
        parts = [self.system_prompt]
        if self.style:
            parts.append(f"\n[表达风格]\n{self.style}")
        return "\n".join(p for p in parts if p).strip()


# --------------------------------------------------------------------------- #
# 本地规则引擎: 生成思维链 (不调任何 API)
# --------------------------------------------------------------------------- #
@dataclass
class Rule:
    name: str
    pattern: re.Pattern
    thoughts: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        flags = re.IGNORECASE
        return cls(
            name=d.get("name", "rule"),
            pattern=re.compile(d.get("pattern", ""), flags),
            thoughts=list(d.get("thoughts", [])),
        )


# 默认规则: 当 YAML 未提供 cot.rules 时使用。
_DEFAULT_RULES: list[dict] = [
    {
        "name": "question",
        "pattern": r"(\?|？|为什么|怎么|如何|是不是|能不能|什么|哪)",
        "thoughts": [
            "用户在提问，先判断这是事实型、操作型还是观点型问题。",
            "回顾记忆里是否有相关上下文，避免重复或前后矛盾。",
            "给出直接答案，再补充必要的依据或步骤。",
        ],
    },
    {
        "name": "task",
        "pattern": r"(帮我|写|生成|做一个|实现|创建|改一下|修复|优化)",
        "thoughts": [
            "这是一个任务请求，先拆解成可执行的小步骤。",
            "确认约束条件与已知信息，列出缺失的关键输入。",
            "按步骤推进，给出可直接使用的结果。",
        ],
    },
    {
        "name": "emotion",
        "pattern": r"(难过|开心|累|烦|焦虑|喜欢|讨厌|担心|害怕|孤独)",
        "thoughts": [
            "用户表达了情绪，先共情再回应内容。",
            "结合人格设定保持一致的语气，不要说教。",
        ],
    },
    {
        "name": "memory_recall",
        "pattern": r"(还记得|之前|上次|刚才|昨天|我们聊过|那个)",
        "thoughts": [
            "用户在引用过去的对话，优先依赖拉取到的记忆。",
            "若记忆中没有相关内容，坦诚说明而不是编造。",
        ],
    },
]


class RuleEngine:
    def __init__(self, rules: list[dict] | None = None, enabled: bool = True):
        source = rules if rules else _DEFAULT_RULES
        self.rules = [Rule.from_dict(r) for r in source]
        self.enabled = enabled

    def reason(self, user_input: str, memory: str = "") -> str:
        """根据输入与记忆，本地生成一段思维链文本。"""
        if not self.enabled:
            return ""
        steps: list[str] = []
        for rule in self.rules:
            if rule.pattern.search(user_input):
                steps.extend(rule.thoughts)

        # 去重并保序
        seen: set[str] = set()
        unique_steps = [s for s in steps if not (s in seen or seen.add(s))]

        if not unique_steps:
            unique_steps = [
                "理解用户这句话的核心意图。",
                "结合人格设定与已有记忆组织回应。",
                "给出清晰、自然、符合人格的回答。",
            ]

        header = "[思维链 / 本地推理]"
        if memory.strip():
            unique_steps.insert(0, "已检索到相关记忆，需在回应中加以利用。")
        body = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(unique_steps))
        return f"{header}\n{body}"


# --------------------------------------------------------------------------- #
# OB MCP 客户端 (HTTP / SSE, JSON-RPC 2.0 streamable-http)
# --------------------------------------------------------------------------- #
class MCPClient:
    """最小化的 MCP streamable-HTTP 客户端，用于拉取记忆。"""

    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("enabled", False))
        self.base_url = (cfg.get("base_url") or "").rstrip("/")
        self.token = cfg.get("token") or ""
        self.memory_tool = cfg.get("memory_tool", "breath")
        self.memory_args = cfg.get("memory_args", {}) or {}
        self.query_field = cfg.get("query_field", "query")
        self.timeout = cfg.get("timeout", 30)
        self.session_id: str | None = None
        self._initialized = False

        # 回合结束写回记忆 (grow=日记归档 / hold=单条记忆)
        wb = cfg.get("write_back", {}) or {}
        self.write_enabled = bool(wb.get("enabled", False))
        self.write_tool = wb.get("tool", "grow")
        self.write_min_chars = wb.get("min_chars", 10)
        self.write_tags = wb.get("tags", "")
        self.write_importance = wb.get("importance")

    def _headers(self) -> dict:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        return h

    def _post(self, payload: dict) -> dict | None:
        resp = requests.post(
            self.base_url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
            stream=True,
        )
        # 初始化时服务器可能下发会话 ID
        sid = resp.headers.get("Mcp-Session-Id")
        if sid:
            self.session_id = sid
        resp.raise_for_status()

        ctype = resp.headers.get("Content-Type", "")
        if "text/event-stream" in ctype:
            return self._parse_sse(resp, payload.get("id"))
        text = resp.text.strip()
        if not text:
            return None
        return json.loads(text)

    @staticmethod
    def _parse_sse(resp: requests.Response, want_id: Any) -> dict | None:
        """解析 SSE 流，返回与请求 id 匹配的 JSON-RPC 消息。"""
        last = None
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            data = raw_line[len("data:"):].strip()
            if not data or data == "[DONE]":
                continue
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue
            last = msg
            if want_id is not None and msg.get("id") == want_id:
                return msg
        return last

    def _notify(self, method: str, params: dict | None = None) -> None:
        """发送 JSON-RPC 通知 (无 id, 不期待响应)。"""
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        try:
            requests.post(
                self.base_url, headers=self._headers(),
                json=payload, timeout=self.timeout,
            )
        except requests.RequestException:
            pass

    def initialize(self) -> bool:
        if self._initialized:
            return True
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "siwei", "version": "1.0.0"},
            },
        }
        resp = self._post(payload)
        if resp and "error" in resp:
            raise RuntimeError(f"MCP initialize 失败: {resp['error']}")
        self._notify("notifications/initialized")
        self._initialized = True
        return True

    def fetch_memory(self, query: str) -> str:
        """调用记忆工具，返回拼接好的记忆文本。失败时返回空串。"""
        if not self.enabled:
            return ""
        if not self.base_url:
            return ""
        try:
            self.initialize()
            args = dict(self.memory_args)
            if self.query_field:
                args.setdefault(self.query_field, query)
            payload = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tools/call",
                "params": {"name": self.memory_tool, "arguments": args},
            }
            resp = self._post(payload)
            if not resp:
                return ""
            if "error" in resp:
                print(f"[MCP] 记忆工具返回错误: {resp['error']}", file=sys.stderr)
                return ""
            return self._extract_text(resp.get("result"))
        except requests.RequestException as e:
            print(f"[MCP] 拉取记忆失败 (网络): {e}", file=sys.stderr)
            return ""
        except Exception as e:  # noqa: BLE001 - 记忆是可选增强，失败不应中断主流程
            print(f"[MCP] 拉取记忆失败: {e}", file=sys.stderr)
            return ""

    def write_memory(self, content: str) -> bool:
        """把一段内容写回 OB (grow 日记 / hold 单条记忆)。失败时返回 False。"""
        if not (self.enabled and self.write_enabled and self.base_url):
            return False
        if len(content.strip()) < self.write_min_chars:
            return False
        try:
            self.initialize()
            args: dict[str, Any] = {"content": content}
            if self.write_tool == "hold":
                if self.write_tags:
                    args["tags"] = self.write_tags
                if self.write_importance is not None:
                    args["importance"] = self.write_importance
            payload = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tools/call",
                "params": {"name": self.write_tool, "arguments": args},
            }
            resp = self._post(payload)
            if resp and "error" in resp:
                print(f"[MCP] 写回记忆失败: {resp['error']}", file=sys.stderr)
                return False
            return True
        except requests.RequestException as e:
            print(f"[MCP] 写回记忆失败 (网络): {e}", file=sys.stderr)
            return False
        except Exception as e:  # noqa: BLE001 - 写回是可选增强，失败不应中断主流程
            print(f"[MCP] 写回记忆失败: {e}", file=sys.stderr)
            return False

    @classmethod
    def _extract_text(cls, result: Any) -> str:
        """从 MCP tools/call 结果里提取纯文本。

        兼容标准 MCP content 数组，也兼容 Ombre Brian 的 {"result": ...} 外壳。
        """
        if result is None:
            return ""
        if isinstance(result, str):
            return cls._unwrap(result)
        if isinstance(result, list):
            return "\n".join(cls._extract_text(x) for x in result).strip()
        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, list):
                chunks = []
                for item in content:
                    if isinstance(item, dict):
                        if "text" in item:
                            chunks.append(str(item["text"]))
                    else:
                        chunks.append(str(item))
                return cls._unwrap("\n".join(chunks).strip())
            # 非标准: result 本身带 result/memories/text 字段
            for key in ("result", "memories", "text"):
                if key in result:
                    return cls._unwrap(result[key])
        return json.dumps(result, ensure_ascii=False)

    @classmethod
    def _unwrap(cls, text: Any) -> str:
        """若文本是 {"result": ...} 之类的 JSON 串，剥掉外壳取内部文本。"""
        if not isinstance(text, str):
            return cls._extract_text(text)
        s = text.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                return s
            if isinstance(parsed, dict):
                for key in ("result", "memories", "text", "content"):
                    if key in parsed:
                        return cls._extract_text(parsed[key])
            elif isinstance(parsed, list):
                return "\n".join(cls._extract_text(x) for x in parsed)
            return s
        return s


# --------------------------------------------------------------------------- #
# Claude API 客户端 (第三方站, OpenAI 兼容格式)
# --------------------------------------------------------------------------- #
class ClaudeClient:
    def __init__(self, cfg: dict):
        self.base_url = (cfg.get("base_url") or "").rstrip("/")
        self.api_key = cfg.get("api_key") or ""
        self.model = cfg.get("model", "claude-3-5-sonnet")
        self.max_tokens = cfg.get("max_tokens", 2048)
        self.temperature = cfg.get("temperature", 0.7)
        self.timeout = cfg.get("timeout", 120)
        if not self.base_url:
            sys.exit("配置缺失: claude.base_url")
        if not self.api_key:
            sys.exit("配置缺失: claude.api_key (可用环境变量, 见 config.example.yaml)")

    def _endpoint(self) -> str:
        # base_url 通常已含 /v1；若没有则补全。
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    def chat(self, messages: list[dict], stream: bool = False) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }
        resp = requests.post(
            self._endpoint(), headers=headers, json=payload,
            timeout=self.timeout, stream=stream,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Claude API {resp.status_code}: {resp.text[:500]}")

        if stream:
            return self._consume_stream(resp)
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _consume_stream(resp: requests.Response) -> str:
        full = []
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content")
            if piece:
                sys.stdout.write(piece)
                sys.stdout.flush()
                full.append(piece)
        sys.stdout.write("\n")
        return "".join(full)


# --------------------------------------------------------------------------- #
# 中间层编排
# --------------------------------------------------------------------------- #
class Middleware:
    def __init__(self, config: dict):
        self.config = config
        persona_cfg = config.get("persona", {})
        self.persona = Persona.load(
            persona_cfg.get("dir", "personas"),
            persona_cfg.get("default", "default"),
        )
        cot_cfg = config.get("cot", {})
        self.rule_engine = RuleEngine(
            rules=cot_cfg.get("rules"),
            enabled=cot_cfg.get("enabled", True),
        )
        self.mcp = MCPClient(config.get("mcp", {}))
        self.claude = ClaudeClient(config.get("claude", {}))
        self.history: list[dict] = []
        self.max_history = config.get("max_history", 10)

    def build_messages(self, user_input: str) -> tuple[list[dict], dict]:
        """构造发往 Claude 的消息，并返回调试信息。"""
        memory = self.mcp.fetch_memory(user_input)
        cot = self.rule_engine.reason(user_input, memory)

        system_parts = [self.persona.build_system_prompt()]
        if memory:
            system_parts.append(f"\n[相关记忆]\n{memory}")
        if cot:
            system_parts.append(
                f"\n请在内部参考以下本地推理脉络来组织回答(不要原样复述):\n{cot}"
            )
        system_prompt = "\n".join(p for p in system_parts if p).strip()

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history[-self.max_history * 2:])
        messages.append({"role": "user", "content": user_input})

        debug = {"memory": memory, "cot": cot, "system": system_prompt}
        return messages, debug

    def respond(self, user_input: str, stream: bool = False,
                show_debug: bool = False) -> str:
        messages, debug = self.build_messages(user_input)
        if show_debug:
            if debug["memory"]:
                print(f"\n\033[2m[记忆]\n{debug['memory']}\033[0m", file=sys.stderr)
            if debug["cot"]:
                print(f"\033[2m[{debug['cot']}\033[0m\n", file=sys.stderr)
        reply = self.claude.chat(messages, stream=stream)
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": reply})
        if self.mcp.write_enabled:
            saved = self.mcp.write_memory(f"用户: {user_input}\n{self.persona.name}: {reply}")
            if show_debug and saved:
                print(f"\033[2m[已写回记忆 -> {self.mcp.write_tool}]\033[0m",
                      file=sys.stderr)
        return reply


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def repl(mw: Middleware, stream: bool, show_debug: bool) -> None:
    print(f"已加载人格: {mw.persona.name}"
          + (f" — {mw.persona.description}" if mw.persona.description else ""))
    print("输入消息开始对话 (输入 /exit 退出, /reset 清空历史, /persona 查看人格)\n")
    while True:
        try:
            user_input = input("\033[1m你 >\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            break
        if user_input == "/reset":
            mw.history.clear()
            print("历史已清空。")
            continue
        if user_input == "/persona":
            print(mw.persona.build_system_prompt())
            continue
        try:
            if stream:
                print("\033[36m助手 >\033[0m ", end="", flush=True)
                mw.respond(user_input, stream=True, show_debug=show_debug)
            else:
                reply = mw.respond(user_input, stream=False, show_debug=show_debug)
                print(f"\033[36m助手 >\033[0m {reply}\n")
        except Exception as e:  # noqa: BLE001
            print(f"\033[31m[错误] {e}\033[0m", file=sys.stderr)


def check_connection(mw: Middleware) -> int:
    """自检 OB MCP 与 Claude API 连通性，返回退出码。"""
    ok = "\033[32m✓\033[0m"
    bad = "\033[31m✗\033[0m"
    failures = 0

    print(f"人格: {ok} {mw.persona.name}")

    mcp = mw.mcp
    if not mcp.enabled:
        print(f"OB MCP: \033[2m已禁用 (mcp.enabled=false)\033[0m")
    elif not mcp.base_url:
        print(f"OB MCP: {bad} 未配置 base_url")
        failures += 1
    else:
        print(f"OB MCP: 连接 {mcp.base_url} ...")
        try:
            mcp.initialize()
            print(f"  initialize {ok}" + (f" (session={mcp.session_id})" if mcp.session_id else ""))
            mem = mcp.fetch_memory("自检测试")
            preview = mem.replace("\n", " ")[:80] if mem else "(空，但调用成功)"
            print(f"  {mcp.memory_tool} 拉记忆 {ok} -> {preview}")
            if mcp.write_enabled:
                print(f"  写回模式: {mcp.write_tool} (enabled)")
        except Exception as e:  # noqa: BLE001
            print(f"  {bad} {e}")
            failures += 1

    claude = mw.claude
    print(f"Claude API: 连接 {claude._endpoint()} ...")
    try:
        reply = claude.chat([{"role": "user", "content": "ping，请只回复 pong"}])
        print(f"  {ok} 模型 {claude.model} 响应: {reply.strip()[:60]}")
    except Exception as e:  # noqa: BLE001
        print(f"  {bad} {e}")
        failures += 1

    print()
    if failures:
        print(f"\033[31m自检发现 {failures} 项问题，请检查 config.yaml。\033[0m")
        return 1
    print("\033[32m全部连通，可以开聊。\033[0m")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="siwei 本地中间层")
    parser.add_argument("-c", "--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("-p", "--persona", help="覆盖默认人格名")
    parser.add_argument("-m", "--message", help="单次提问模式 (不进入交互)")
    parser.add_argument("--no-mcp", action="store_true", help="禁用记忆拉取")
    parser.add_argument("--no-write", action="store_true", help="禁用记忆写回")
    parser.add_argument("--no-cot", action="store_true", help="禁用本地思维链")
    parser.add_argument("--stream", action="store_true", help="流式输出")
    parser.add_argument("--debug", action="store_true", help="打印记忆与思维链")
    parser.add_argument("--check", action="store_true",
                        help="自检: 测试 OB MCP 与 Claude API 连通性后退出")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.persona:
        config.setdefault("persona", {})["default"] = args.persona
    if args.no_mcp:
        config.setdefault("mcp", {})["enabled"] = False
    if args.no_write:
        config.setdefault("mcp", {}).setdefault("write_back", {})["enabled"] = False
    if args.no_cot:
        config.setdefault("cot", {})["enabled"] = False

    mw = Middleware(config)

    if args.check:
        return check_connection(mw)

    if args.message:
        try:
            reply = mw.respond(args.message, stream=args.stream, show_debug=args.debug)
            if not args.stream:
                print(reply)
        except Exception as e:  # noqa: BLE001
            print(f"[错误] {e}", file=sys.stderr)
            return 1
        return 0

    repl(mw, stream=args.stream, show_debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
