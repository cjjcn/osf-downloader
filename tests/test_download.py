"""Tests for core download functionality"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from osf_downloader.download import OSFDownloader, OSFRequestError


class TestProjectDownload:
    """Test downloading entire OSF projects"""

    def test_download_entire_project(self, console, output_dir, project_id):
        """
        Test downloading entire OSF project as ZIP
        (as described in README: osf-download <OSF_ID> ./data)
        """
        downloader = OSFDownloader(console=console, show_progress=True)

        save_path = output_dir / "project.zip"
        result = downloader.download(project_id, save_path)

        assert result.exists()
        assert result.suffix == ".zip"
        console.print(f"[green]✓[/green] Successfully downloaded project to: {result}")

    def test_download_project_auto_extension(self, console, output_dir, project_id):
        """
        Test downloading project where .zip extension is chosen automatically
        (as described in README: osf-download abcd1 ./datasets/osf)
        """
        downloader = OSFDownloader(console=console, show_progress=True)

        save_path = output_dir / "project_auto"
        result = downloader.download(project_id, save_path)

        assert result.exists()
        console.print(f"[green]✓[/green] Successfully downloaded project to: {result}")

    def test_download_project_to_existing_directory(
        self, console, output_dir, project_id
    ):
        downloader = OSFDownloader(console=console, show_progress=True)

        result = downloader.download(project_id, output_dir)

        assert result.exists()
        assert result.parent == output_dir
        assert result.name == "project.zip"


class TestFileDownload:
    """Test downloading individual files from OSF storage"""

    def test_download_single_file(self, console, output_dir, project_id, file_path):
        """
        Test downloading a single file by its path inside OSF storage
        (as described in README: osf-download <OSF_ID> ./data/myfile.csv path/inside/osf/myfile.csv)
        """
        downloader = OSFDownloader(console=console, show_progress=True)

        filename = file_path.split("/")[-1]
        save_path = output_dir / f"{filename}.zip"
        result = downloader.download(project_id, save_path, file_path)

        assert result.exists()
        console.print(f"[green]✓[/green] Successfully downloaded file to: {result}")

    def test_download_single_file_to_directory(
        self, console, output_dir, project_id, file_path
    ):
        """
        Test downloading a single file into a directory
        (as described in README: osf-download abcd1 ./datasets results/data.csv)
        """
        downloader = OSFDownloader(console=console, show_progress=True)

        filename = file_path.split("/")[-1]
        save_path = output_dir / filename
        result = downloader.download(project_id, save_path, file_path)

        assert result.exists()
        console.print(f"[green]✓[/green] Successfully downloaded file to: {result}")

    def test_download_single_file_into_existing_directory(
        self, console, output_dir, project_id, file_path
    ):
        downloader = OSFDownloader(console=console, show_progress=True)

        result = downloader.download(project_id, output_dir, file_path)

        assert result.exists()
        assert result.parent == output_dir
        assert result.name == file_path.split("/")[-1]


class TestPaginationSupport:
    """Test pagination support for large directories"""

    def test_walk_files_with_pagination(self, console):
        """
        Test that _walk_files() correctly handles paginated API responses
        with links.next pagination
        """
        # Create mock paginated API responses
        page1 = {
            "data": [
                {
                    "attributes": {"name": "file1.txt", "kind": "file"},
                    "links": {"download": "https://example.com/file1.txt"},
                },
                {
                    "attributes": {"name": "file2.txt", "kind": "file"},
                    "links": {"download": "https://example.com/file2.txt"},
                },
            ],
            "links": {"next": "https://api.example.com/page2"},
        }

        page2 = {
            "data": [
                {
                    "attributes": {"name": "file3.txt", "kind": "file"},
                    "links": {"download": "https://example.com/file3.txt"},
                },
            ],
            "links": {},
        }

        downloader = OSFDownloader(console=console)

        # Mock the _get_json_url method
        responses = iter([page1, page2])

        def mock_get_json(url):
            return next(responses)

        with patch.object(downloader, "_get_json_url", side_effect=mock_get_json):
            files = list(downloader._walk_files("https://api.example.com/page1"))

        # Should have retrieved all files from both pages
        assert len(files) == 3
        assert files[0] == ("https://example.com/file1.txt", "file1.txt")
        assert files[1] == ("https://example.com/file2.txt", "file2.txt")
        assert files[2] == ("https://example.com/file3.txt", "file3.txt")

    def test_resolve_file_path_with_pagination(self, console):
        """
        Test that _resolve_file_path() correctly handles paginated folder listings
        """
        # Create mock paginated API responses for a folder listing
        page1 = {
            "data": [
                {
                    "attributes": {"name": "file1.txt", "kind": "file"},
                    "links": {"download": "https://example.com/file1.txt"},
                },
                {
                    "attributes": {"name": "file2.txt", "kind": "file"},
                    "links": {"download": "https://example.com/file2.txt"},
                },
            ],
            "links": {"next": "https://api.example.com/page2"},
        }

        page2 = {
            "data": [
                {
                    "attributes": {"name": "target_file.csv", "kind": "file"},
                    "links": {"download": "https://example.com/target_file.csv"},
                },
            ],
            "links": {},
        }

        downloader = OSFDownloader(console=console)

        # Mock the _get_json_url method
        responses = iter([page1, page2])

        def mock_get_json(url):
            return next(responses)

        with patch.object(downloader, "_get_json_url", side_effect=mock_get_json):
            result = downloader._resolve_file_path(
                "https://api.example.com/page1", "target_file.csv"
            )

        # Should have found the file on the second page
        assert result == "https://example.com/target_file.csv"


class TestRetrySupport:
    """Test retry behavior for transient request failures"""

    def test_get_json_url_retries_on_502(self, console):
        downloader = OSFDownloader(console=console)

        retry_response = MagicMock()
        retry_response.status_code = 502

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.raise_for_status.return_value = None
        success_response.json.return_value = {"data": [], "links": {}}

        with patch.object(
            downloader.session,
            "get",
            side_effect=[retry_response, success_response],
        ) as mocked_get:
            with patch("osf_downloader.download.time.sleep"):
                data = downloader._get_json_url("https://api.example.com/page1")

        assert data == {"data": [], "links": {}}
        assert mocked_get.call_count == 2
        retry_response.close.assert_called_once()

    def test_get_json_url_raises_on_non_retryable_status(self, console):
        downloader = OSFDownloader(console=console)

        not_found_response = MagicMock()
        not_found_response.status_code = 404
        not_found_response.raise_for_status.side_effect = requests.HTTPError("404")

        with patch.object(downloader.session, "get", return_value=not_found_response):
            with pytest.raises(OSFRequestError):
                downloader._get_json_url("https://api.example.com/missing")
