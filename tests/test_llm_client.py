from __future__ import annotations
import sys, json, pytest
from unittest.mock import MagicMock, patch
from src.config import settings
from src.llm.client import LLMClient, LLMError

class TestLLMClient:
    def test_is_configured_empty(self):
        with patch.object(settings, 'openai_api_key', ''):
            assert not LLMClient.is_configured()
    def test_is_configured_valid(self):
        with patch.object(settings, 'openai_api_key', 'sk-real'):
            assert LLMClient.is_configured()
    def test_planner_is_configured_valid(self):
        with patch.object(settings, 'planner_api_key', 'sk-planner'):
            assert LLMClient.planner_is_configured()
    @patch('src.llm.client.httpx.post')
    def test_chat_json_success(self, mock_post):
        mr = MagicMock()
        mr.status_code = 200
        mr.json.return_value = {'choices': [{'message': {'content': json.dumps({'key': 'val'})}}]}
        mock_post.return_value = mr
        r = LLMClient().chat_json('sys', 'user')
        assert r == {'key': 'val'}
    @patch('src.llm.client.httpx.post')
    def test_chat_json_markdown(self, mock_post):
        mr = MagicMock()
        mr.status_code = 200
        mr.json.return_value = {'choices': [{'message': {'content': json.dumps({'a': 1})}}]}
        mock_post.return_value = mr
        r = LLMClient().chat_json('', '')
        assert r == {'a': 1}

    @patch('src.llm.client.httpx.post')
    def test_chat_returns_raw_content(self, mock_post):
        mr = MagicMock()
        mr.status_code = 200
        mr.json.return_value = {'choices': [{'message': {'content': '<html><body>ok</body></html>'}}]}
        mock_post.return_value = mr
        content = LLMClient().chat('sys', 'user')
        assert content == '<html><body>ok</body></html>'
