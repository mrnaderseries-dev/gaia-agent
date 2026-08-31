from __future__ import annotations

from smolagents import (
    DuckDuckGoSearchTool,
    VisitWebpageTool,
)

class SafeDuckDuckGoSearch:
    """أداة بحث مغلفة لتتجاهل أي باراميترات زائدة مثل code"""
    def __init__(self):
        self._tool = DuckDuckGoSearchTool()
        self.name = self._tool.name
        self.description = self._tool.description
        self.inputs = self._tool.inputs
        self.output_type = self._tool.output_type

    def __call__(self, *args, **kwargs):
        # تصفية أي وسائط غير مطابقة مثل code وتمرير الـ query فقط
        clean_kwargs = {k: v for k, v in kwargs.items() if k in ['query', 'max_results']}
        return self._tool(*args, **clean_kwargs)


class WebTools:
    """
    Web tools used by the GAIA agent.

    Provides:
    - Web search (DuckDuckGo)
    - Webpage visiting and content extraction
    """

    def __init__(self) -> None:
        self.search = SafeDuckDuckGoSearch()  # استبدالها بالأداة الآمنة
        self.visit = VisitWebpageTool()

    def get_tools(self) -> list:
        """
        Return all web tools as a list.
        """
        return [
            self.search,
            self.visit,
        ]