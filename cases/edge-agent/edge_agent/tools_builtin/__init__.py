"""Built-in tool collection for the edge-agent framework."""

from .web import web_search, web_fetch
from .filesystem import read_file, write_file, edit_file, list_dir
from .shell import shell
from .computer import screenshot, click, type_text, scroll, hotkey
from .system import memory_save

ALL_TOOLS = [
    web_search, web_fetch,
    read_file, write_file, edit_file, list_dir,
    shell,
    screenshot, click, type_text, scroll, hotkey,
    memory_save,
]

__all__ = ["ALL_TOOLS"]
