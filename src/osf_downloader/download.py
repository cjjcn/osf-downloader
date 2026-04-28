# src/osf_downloader/download.py

from __future__ import annotations

import os
import random
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import requests
from rich.console import Console
from tqdm import tqdm


class OSFError(RuntimeError):
    pass


class OSFRequestError(OSFError):
    pass


class OSFNotFoundError(OSFError):
    pass


class OSFDownloader:
    API_ROOT = "https://api.osf.io/v2"
    REQUEST_TIMEOUT = int(os.getenv("OSF_REQUEST_TIMEOUT", "60"))
    MAX_RETRIES = int(os.getenv("OSF_MAX_RETRIES", "12"))
    RETRY_MAX_DELAY = float(os.getenv("OSF_RETRY_MAX_DELAY", "60"))
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
    REFRESHABLE_DOWNLOAD_STATUS_CODES = {400, 401, 403}

    _TQDM_COLOURS = [
        "green",
        "cyan",
        "magenta",
        "yellow",
        "blue",
        "red",
        "white",
    ]

    def __init__(
        self,
        *,
        console: Optional[Console] = None,
        show_progress: bool = True,
        max_workers: int = 8,
    ) -> None:
        self.console = console or Console()
        self.show_progress = show_progress
        self.max_workers = max_workers
        self.session = requests.Session()

    # =======================
    # Public API
    # =======================

    def download(
        self,
        project_id: str,
        save_path: Path,
        file_path: Optional[str] = None,
    ) -> Path:
        self._status(f"Connecting to OSF project {project_id}")

        node = self._get_json(f"/nodes/{project_id}")
        root_url = self._get_osfstorage_url(node)

        save_path = Path(save_path)
        save_path = self._resolve_save_path(save_path, file_path)

        if file_path:
            url = self._resolve_file_path(root_url, file_path)
            self._download_single(url, save_path)
        else:
            self._status("Listing all files in osfstorage")
            self._download_all_to_zip(
                root_url,
                self._walk_files_with_progress(root_url),
                save_path,
            )

        self._status(f"Saved to {save_path}")
        return save_path

    # =======================
    # OSF traversal
    # =======================

    def _get_osfstorage_url(self, node: dict) -> str:
        files_url = node["data"]["relationships"]["files"]["links"]["related"]["href"]
        data = self._get_json_url(files_url)

        for item in data["data"]:
            if item["attributes"]["provider"] == "osfstorage":
                return item["relationships"]["files"]["links"]["related"]["href"]

        raise OSFNotFoundError("osfstorage provider not found")

    def _walk_files(
        self,
        url: str,
        prefix: str = "",
        on_page: Optional[Callable[[], None]] = None,
    ) -> Iterable[tuple[str, str]]:
        while url:
            if on_page:
                on_page()
            data = self._get_json_url(url)

            for item in data["data"]:
                name = item["attributes"]["name"]
                kind = item["attributes"]["kind"]

                if kind == "file":
                    yield item["links"]["download"], f"{prefix}{name}"
                else:
                    next_url = item["relationships"]["files"]["links"]["related"]["href"]
                    yield from self._walk_files(
                        next_url,
                        f"{prefix}{name}/",
                        on_page=on_page,
                    )

            url = data.get("links", {}).get("next")

    def _walk_files_with_progress(self, root_url: str) -> Iterable[tuple[str, str]]:
        pages_scanned = 0

        def mark_page_scanned() -> None:
            nonlocal pages_scanned
            pages_scanned += 1

        with self._tqdm(
            total=None,
            desc="Listing files",
            unit="file",
            leave=True,
            position=0,
            colour=self._TQDM_COLOURS[1],
        ) as bar:
            for file_info in self._walk_files(root_url, on_page=mark_page_scanned):
                bar.update(1)
                bar.set_postfix_str(f"pages={pages_scanned}")
                yield file_info

            bar.set_postfix_str(f"pages={pages_scanned}")

    def _resolve_file_path(self, root_url: str, path: str) -> str:
        current = root_url
        for part in path.split("/"):
            found = False
            next_url = current
            while next_url:
                data = self._get_json_url(next_url)
                for item in data["data"]:
                    if item["attributes"]["name"] == part:
                        if item["attributes"]["kind"] == "folder":
                            current = item["relationships"]["files"]["links"]["related"][
                                "href"
                            ]
                            found = True
                            break
                        return item["links"]["download"]
                if found:
                    break
                next_url = data.get("links", {}).get("next")
            if not found:
                raise OSFNotFoundError(f"Path not found: {part}")
        raise OSFError(f"Path resolves to a folder: {path}")

    # =======================
    # Download logic
    # =======================

    def _download_single(self, url: str, target: Path) -> None:
        response = self._request_get(url, stream=True)

        os.makedirs(target.parent, exist_ok=True)

        total = int(response.headers.get("content-length", 0))
        chunks = response.iter_content(chunk_size=8192)

        with open(target, "wb") as f:
            if not self.show_progress:
                for chunk in chunks:
                    if chunk:
                        f.write(chunk)
                return

            with self._tqdm(
                total=total or None,
                desc=target.name,
                leave=True,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                colour=self._TQDM_COLOURS[0],
            ) as bar:
                for chunk in chunks:
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

    def _download_all_to_zip(
        self,
        root_url: str,
        files: Iterable[tuple[str, str]],
        target: Path,
    ) -> None:
        os.makedirs(target.parent, exist_ok=True)

        completed = self._existing_archive_members(target)
        seen_paths = set(completed)
        skipped_duplicates = 0

        if completed:
            self._status(f"Resuming archive with {len(completed)} existing files")

        with zipfile.ZipFile(
            target,
            "a" if target.exists() else "w",
            zipfile.ZIP_DEFLATED,
        ) as zf:
            for i, (url, arcname) in enumerate(files):
                if arcname in completed:
                    continue

                if arcname in seen_paths:
                    skipped_duplicates += 1
                    continue

                self._download_zip_entry(root_url, url, arcname, zf, i)
                seen_paths.add(arcname)

        if skipped_duplicates:
            self._status(f"Skipped {skipped_duplicates} duplicate file entries")

    def _download_zip_entry(
        self,
        root_url: str,
        url: str,
        arcname: str,
        zf: zipfile.ZipFile,
        colour_index: int,
    ) -> None:
        current_url = url

        for refresh_attempt in range(self.MAX_RETRIES):
            temp_path: Optional[Path] = None
            try:
                fd, temp_name = tempfile.mkstemp(prefix="osf-download-", suffix=".part")
                os.close(fd)
                temp_path = Path(temp_name)

                with self._request_get(current_url, stream=True) as response:
                    total = int(response.headers.get("content-length", 0))

                    with self._tqdm(
                        total=total or None,
                        desc=arcname,
                        position=1,
                        leave=False,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        colour=self._TQDM_COLOURS[colour_index % len(self._TQDM_COLOURS)],
                    ) as bar:
                        with open(temp_path, "wb") as temp_file:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    temp_file.write(chunk)
                                    bar.update(len(chunk))

                zf.write(temp_path, arcname)
                return
            except OSFRequestError as e:
                status = self._get_error_status_code(e)
                refreshable = status in self.REFRESHABLE_DOWNLOAD_STATUS_CODES
                if not refreshable or refresh_attempt == self.MAX_RETRIES - 1:
                    raise

                self._status(
                    f"Refreshing expired download URL for {arcname} (attempt {refresh_attempt + 1}/{self.MAX_RETRIES})"
                )
                current_url = self._resolve_file_path(root_url, arcname)
            finally:
                if temp_path and temp_path.exists():
                    temp_path.unlink()

    def _existing_archive_members(self, target: Path) -> set[str]:
        if not target.exists() or target.stat().st_size == 0:
            return set()

        try:
            with zipfile.ZipFile(target, "r") as existing_zip:
                return {info.filename for info in existing_zip.infolist()}
        except zipfile.BadZipFile:
            backup_path = target.with_suffix(f"{target.suffix}.corrupt.{int(time.time())}")
            target.rename(backup_path)
            self._status(
                f"Existing archive is not resumable; moved it to {backup_path.name} and starting over"
            )
            return set()

    def _tqdm(self, *args: Any, **kwargs: Any) -> tqdm:
        """Create a tqdm progress bar with optional colour.

        tqdm's `colour` kwarg isn't available in all versions; this wrapper
        keeps the CLI working on older tqdm versions by retrying without it.
        """

        kwargs.setdefault("disable", not self.show_progress)

        try:
            return tqdm(*args, **kwargs)
        except TypeError:
            if "colour" not in kwargs:
                raise
            kwargs.pop("colour", None)
            return tqdm(*args, **kwargs)

    # =======================
    # Utilities
    # =======================

    def _resolve_save_path(self, path: Path, file_path: Optional[str]) -> Path:
        if (path.exists() and path.is_dir()) or not path.name:
            filename = Path(file_path).name if file_path else "project.zip"
            return path / filename

        if path.suffix:
            return path
        return path.with_suffix(Path(file_path).suffix if file_path else ".zip")

    def _get_json(self, endpoint: str) -> dict:
        return self._get_json_url(f"{self.API_ROOT}{endpoint}")

    def _get_error_status_code(self, error: Exception) -> Optional[int]:
        cause = error.__cause__
        return getattr(getattr(cause, "response", None), "status_code", None)

    def _request_get(self, url: str, *, stream: bool = False) -> requests.Response:
        for attempt in range(self.MAX_RETRIES):
            response: Optional[requests.Response] = None
            try:
                response = self.session.get(
                    url,
                    stream=stream,
                    timeout=self.REQUEST_TIMEOUT,
                )

                if response.status_code in self.RETRYABLE_STATUS_CODES:
                    status = response.status_code
                    error = requests.HTTPError(f"{status} response for {url}")
                    error.response = response
                    response.close()
                    raise error

                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if response is not None:
                    response.close()
                status = getattr(getattr(e, "response", None), "status_code", None)
                retryable = status in self.RETRYABLE_STATUS_CODES or status is None

                if not retryable or attempt == self.MAX_RETRIES - 1:
                    raise OSFRequestError(str(e)) from e

                delay = min(self.RETRY_MAX_DELAY, (2**attempt) + random.uniform(0.0, 1.0))
                self._status(
                    f"Temporary request error ({status or type(e).__name__}); retrying in {delay:.1f}s (attempt {attempt + 1}/{self.MAX_RETRIES})"
                )
                time.sleep(delay)

        raise OSFRequestError(f"Failed request after {self.MAX_RETRIES} attempts: {url}")

    def _get_json_url(self, url: str) -> dict:
        try:
            response = self._request_get(url)
            return response.json()
        except ValueError as e:
            raise OSFRequestError(f"Invalid JSON response from {url}") from e

    def _status(self, message: str) -> None:
        if self.console:
            self.console.print(f"[blue]{message}[/blue]")
