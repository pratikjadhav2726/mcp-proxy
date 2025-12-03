"""
Unit tests for ProjectionProcessor and GrepProcessor.
"""

import json
import pytest
from mcp_proxy.processors import ProjectionProcessor, GrepProcessor
from mcp.types import TextContent, Content


class TestProjectionProcessor:
    """Tests for ProjectionProcessor."""

    def test_include_mode_simple(self):
        """Test include mode with simple fields."""
        data = {
            "name": "John",
            "email": "john@example.com",
            "age": 30,
            "city": "New York"
        }
        projection = {
            "mode": "include",
            "fields": ["name", "email"]
        }
        result = ProjectionProcessor.apply_projection(data, projection)
        assert result == {"name": "John", "email": "john@example.com"}

    def test_exclude_mode_simple(self):
        """Test exclude mode with simple fields."""
        data = {
            "name": "John",
            "email": "john@example.com",
            "password": "secret123",
            "ssn": "123-45-6789"
        }
        projection = {
            "mode": "exclude",
            "fields": ["password", "ssn"]
        }
        result = ProjectionProcessor.apply_projection(data, projection)
        assert "password" not in result
        assert "ssn" not in result
        assert "name" in result
        assert "email" in result

    def test_include_mode_nested(self):
        """Test include mode with nested fields."""
        data = {
            "user": {
                "name": "John",
                "email": "john@example.com",
                "address": {
                    "street": "123 Main St",
                    "city": "New York"
                }
            },
            "metadata": {
                "created": "2024-01-01"
            }
        }
        projection = {
            "mode": "include",
            "fields": ["user.name", "user.email"]
        }
        result = ProjectionProcessor.apply_projection(data, projection)
        assert "user" in result
        assert result["user"] == {"name": "John", "email": "john@example.com"}

    def test_include_mode_with_arrays(self):
        """Test include mode with arrays."""
        data = {
            "users": [
                {"name": "John", "email": "john@example.com", "age": 30},
                {"name": "Jane", "email": "jane@example.com", "age": 25}
            ]
        }
        projection = {
            "mode": "include",
            "fields": ["users.name", "users.email"]
        }
        result = ProjectionProcessor.apply_projection(data, projection)
        assert "users" in result
        assert len(result["users"]) == 2
        assert result["users"][0] == {"name": "John", "email": "john@example.com"}
        assert result["users"][1] == {"name": "Jane", "email": "jane@example.com"}

    def test_exclude_mode_nested(self):
        """Test exclude mode with nested fields."""
        data = {
            "user": {
                "name": "John",
                "email": "john@example.com",
                "password": "secret123"
            }
        }
        projection = {
            "mode": "exclude",
            "fields": ["user.password"]
        }
        result = ProjectionProcessor.apply_projection(data, projection)
        assert "user" in result
        assert "password" not in result["user"]
        assert "name" in result["user"]
        assert "email" in result["user"]

    def test_project_content_json(self):
        """Test projecting JSON content."""
        json_data = {
            "name": "John",
            "email": "john@example.com",
            "age": 30,
            "city": "New York"
        }
        content = [TextContent(type="text", text=json.dumps(json_data))]
        projection = {
            "mode": "include",
            "fields": ["name", "email"]
        }
        result = ProjectionProcessor.project_content(content, projection)
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        parsed = json.loads(result[0].text)
        assert parsed == {"name": "John", "email": "john@example.com"}

    def test_project_content_plain_text(self):
        """Test that plain text is not modified."""
        content = [TextContent(type="text", text="This is plain text")]
        projection = {
            "mode": "include",
            "fields": ["name"]
        }
        result = ProjectionProcessor.project_content(content, projection)
        assert len(result) == 1
        assert result[0].text == "This is plain text"


class TestGrepProcessor:
    """Tests for GrepProcessor."""

    def test_grep_simple_text(self):
        """Test grep on simple text content."""
        content = [
            TextContent(type="text", text="Line 1: INFO message\nLine 2: ERROR occurred\nLine 3: WARN message")
        ]
        grep_spec = {
            "pattern": "ERROR",
            "caseInsensitive": False
        }
        result = GrepProcessor.apply_grep(content, grep_spec)
        assert len(result) == 1
        assert "ERROR" in result[0].text
        assert "INFO" not in result[0].text

    def test_grep_case_insensitive(self):
        """Test grep with case insensitive matching."""
        content = [
            TextContent(type="text", text="Line 1: error\nLine 2: ERROR\nLine 3: Error")
        ]
        grep_spec = {
            "pattern": "error",
            "caseInsensitive": True
        }
        result = GrepProcessor.apply_grep(content, grep_spec)
        assert len(result) == 1
        lines = result[0].text.split("\n")
        assert len(lines) == 3  # All should match

    def test_grep_max_matches(self):
        """Test grep with max matches limit."""
        content = [
            TextContent(type="text", text="\n".join([f"ERROR {i}" for i in range(10)]))
        ]
        grep_spec = {
            "pattern": "ERROR",
            "maxMatches": 5
        }
        result = GrepProcessor.apply_grep(content, grep_spec)
        assert len(result) == 1
        lines = result[0].text.split("\n")
        assert len([l for l in lines if l]) <= 5  # Should be limited to 5

    def test_grep_structured_content(self):
        """Test grep on structured JSON content."""
        json_data = {
            "users": [
                {"name": "John", "email": "john@gmail.com"},
                {"name": "Jane", "email": "jane@yahoo.com"},
                {"name": "Bob", "email": "bob@gmail.com"}
            ]
        }
        content = [TextContent(type="text", text=json.dumps(json_data))]
        grep_spec = {
            "pattern": "gmail",
            "target": "structuredContent",
            "caseInsensitive": True
        }
        result = GrepProcessor.apply_grep(content, grep_spec)
        assert len(result) == 1
        parsed = json.loads(result[0].text)
        assert "users" in parsed
        assert len(parsed["users"]) == 2  # Only gmail users

    def test_grep_invalid_pattern(self):
        """Test grep with invalid regex pattern."""
        content = [TextContent(type="text", text="Some text")]
        grep_spec = {
            "pattern": "[invalid regex",
            "caseInsensitive": False
        }
        result = GrepProcessor.apply_grep(content, grep_spec)
        assert len(result) == 1
        assert "Error" in result[0].text or "error" in result[0].text.lower()

    def test_grep_no_matches(self):
        """Test grep when no matches are found."""
        content = [TextContent(type="text", text="Line 1: INFO\nLine 2: DEBUG")]
        grep_spec = {
            "pattern": "ERROR",
            "caseInsensitive": False
        }
        result = GrepProcessor.apply_grep(content, grep_spec)
        assert len(result) == 1
        assert "No matches" in result[0].text

    def test_grep_regex_pattern(self):
        """Test grep with regex pattern."""
        content = [
            TextContent(type="text", text="ERROR: Failed\nWARN: Warning\nINFO: Success")
        ]
        grep_spec = {
            "pattern": "ERROR|WARN",
            "caseInsensitive": False
        }
        result = GrepProcessor.apply_grep(content, grep_spec)
        assert len(result) == 1
        assert "ERROR" in result[0].text
        assert "WARN" in result[0].text
        assert "INFO" not in result[0].text

    def test_grep_context_lines_before(self):
        """Test grep with context lines before matches."""
        content = [
            TextContent(type="text", text="Line 1: INFO\nLine 2: DEBUG\nLine 3: ERROR\nLine 4: WARN\nLine 5: INFO")
        ]
        grep_spec = {
            "pattern": "ERROR",
            "contextLines": {
                "before": 2
            }
        }
        result = GrepProcessor.apply_grep(content, grep_spec)
        assert len(result) == 1
        text = result[0].text
        assert "Line 1: INFO" in text
        assert "Line 2: DEBUG" in text
        assert "Line 3: ERROR" in text
        # Should not include lines after
        assert "Line 4: WARN" not in text

    def test_grep_context_lines_after(self):
        """Test grep with context lines after matches."""
        content = [
            TextContent(type="text", text="Line 1: INFO\nLine 2: ERROR\nLine 3: DEBUG\nLine 4: WARN\nLine 5: INFO")
        ]
        grep_spec = {
            "pattern": "ERROR",
            "contextLines": {
                "after": 2
            }
        }
        result = GrepProcessor.apply_grep(content, grep_spec)
        assert len(result) == 1
        text = result[0].text
        assert "Line 2: ERROR" in text
        assert "Line 3: DEBUG" in text
        assert "Line 4: WARN" in text
        # Should not include lines before
        assert "Line 1: INFO" not in text

    def test_grep_context_lines_both(self):
        """Test grep with context lines both before and after."""
        content = [
            TextContent(type="text", text="Line 1: INFO\nLine 2: DEBUG\nLine 3: ERROR\nLine 4: WARN\nLine 5: INFO")
        ]
        grep_spec = {
            "pattern": "ERROR",
            "contextLines": {
                "both": 1
            }
        }
        result = GrepProcessor.apply_grep(content, grep_spec)
        assert len(result) == 1
        text = result[0].text
        assert "Line 2: DEBUG" in text
        assert "Line 3: ERROR" in text
        assert "Line 4: WARN" in text
        # Should not include lines outside context
        assert "Line 1: INFO" not in text
        assert "Line 5: INFO" not in text

    def test_grep_context_lines_multiple_matches(self):
        """Test grep with context lines and multiple matches."""
        content = [
            TextContent(type="text", text="Line 1: INFO\nLine 2: ERROR\nLine 3: DEBUG\nLine 4: ERROR\nLine 5: WARN")
        ]
        grep_spec = {
            "pattern": "ERROR",
            "contextLines": {
                "both": 1
            }
        }
        result = GrepProcessor.apply_grep(content, grep_spec)
        assert len(result) == 1
        text = result[0].text
        # Should include context around both matches
        assert "Line 1: INFO" in text or "Line 2: ERROR" in text
        assert "Line 2: ERROR" in text
        assert "Line 3: DEBUG" in text
        assert "Line 4: ERROR" in text
        assert "Line 5: WARN" in text

    def test_grep_multiline_pattern(self):
        """Test grep with multiline pattern support."""
        content = [
            TextContent(type="text", text="def my_function():\n    return True\n\ndef other():\n    pass")
        ]
        # Pattern that matches def followed by return on next line, but stops at double newline
        grep_spec = {
            "pattern": "def my_function\\(\\).*?return True",
            "multiline": True
        }
        result = GrepProcessor.apply_grep(content, grep_spec)
        assert len(result) == 1
        text = result[0].text
        assert "def my_function()" in text
        assert "return True" in text
        # Should not match the other function (no return)
        assert "def other()" not in text

    def test_grep_multiline_with_dotall(self):
        """Test multiline pattern with dot matching newlines."""
        content = [
            TextContent(type="text", text="Start\nMiddle\nEnd\nOther")
        ]
        grep_spec = {
            "pattern": "Start.*End",
            "multiline": True
        }
        result = GrepProcessor.apply_grep(content, grep_spec)
        assert len(result) == 1
        text = result[0].text
        assert "Start" in text
        assert "End" in text
        assert "Middle" in text

    def test_grep_multiline_without_flag(self):
        """Test that multiline pattern without flag doesn't match across lines."""
        content = [
            TextContent(type="text", text="Start\nMiddle\nEnd")
        ]
        grep_spec = {
            "pattern": "Start.*End",
            "multiline": False
        }
        result = GrepProcessor.apply_grep(content, grep_spec)
        # Should not match because . doesn't match newline without multiline
        assert len(result) == 1
        assert "No matches" in result[0].text

    def test_grep_context_and_multiline(self):
        """Test combining context lines with multiline patterns."""
        # Note: Context lines work differently for multiline - they're not supported
        # because multiline matches span multiple lines already
        content = [
            TextContent(type="text", text="Line 1\nLine 2\nStart\nMiddle\nEnd\nLine 6\nLine 7")
        ]
        grep_spec = {
            "pattern": "Start.*End",
            "multiline": True
        }
        result = GrepProcessor.apply_grep(content, grep_spec)
        assert len(result) == 1
        text = result[0].text
        assert "Start" in text
        assert "End" in text
        assert "Middle" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

