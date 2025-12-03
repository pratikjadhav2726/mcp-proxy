"""
Unit tests for ProjectionProcessor and GrepProcessor.
"""

import json
import pytest
from proxy_server import ProjectionProcessor, GrepProcessor
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

