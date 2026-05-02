# src/osf_downloader/download.py

from __future__ import annotations

import json
import os
import random
import tempfile
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock
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
    RATE_LIMIT_COOLDOWN = float(os.getenv("OSF_RATE_LIMIT_COOLDOWN", "60"))
    RATE_LIMIT_MAX_COOLDOWN = float(os.getenv("OSF_RATE_LIMIT_MAX_COOLDOWN", "180"))
    REQUEST_SPACING = float(os.getenv("OSF_REQUEST_SPACING", "0.12"))
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
        self._rate_limit_lock = Lock()
        self._rate_limited_until = 0.0
        self._request_spacing_lock = Lock()
        self._next_request_at = 0.0

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
        include_info: bool = False,
    ) -> Iterable[tuple[str, str]]:
        while url:
            if on_page:
                on_page()
            data = self._get_json_url(url)

            for item in data["data"]:
                name = item["attributes"]["name"]
                kind = item["attributes"]["kind"]

                if kind == "file":
                    if include_info:
                        yield (
                            item["links"]["download"],
                            f"{prefix}{name}",
                            item.get("links", {}).get("info"),
                        )
                    else:
                        yield item["links"]["download"], f"{prefix}{name}"
                else:
                    next_url = item["relationships"]["files"]["links"]["related"]["href"]
                    yield from self._walk_files(
                        next_url,
                        f"{prefix}{name}/",
                        on_page=on_page,
                        include_info=include_info,
                    )

            url = data.get("links", {}).get("next")

    def _walk_files_with_progress(self, root_url: str) -> Iterable[tuple[str, str, Optional[str]]]:
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
            for file_info in self._walk_files(
                root_url,
                on_page=mark_page_scanned,
                include_info=True,
            ):
                bar.update(1)
                bar.set_postfix_str(f"pages={pages_scanned}")
                yield file_info

            bar.set_postfix_str(f"pages={pages_scanned}")

    def _resolve_file_path(self, root_url: str, path: str) -> str:
        return self._resolve_file_path_with_progress(root_url, path)

    def _resolve_file_path_with_progress(
        self,
        root_url: str,
        path: str,
        on_page: Optional[Callable[[], None]] = None,
    ) -> str:
        current = root_url
        for part in path.split("/"):
            found = False
            next_url = current
            while next_url:
                if on_page:
                    on_page()
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
        files: Iterable[tuple[str, ...]],
        target: Path,
    ) -> None:
        os.makedirs(target.parent, exist_ok=True)

        completed = self._existing_archive_members(target)
        seen_paths = set(completed)
        skipped_duplicates = 0
        skipped_completed = 0
        skipped_missing = 0
        downloaded_files = 0
        inflight_files = 0
        resume_state = self._resume_state_path(target)

        if completed:
            self._status(f"Resuming archive with {len(completed)} existing files")

        if resume_state.exists():
            self._status(f"Using resume state: {resume_state.name}")
            pending_files = self._iter_resume_entries(resume_state)
        else:
            self._status("Building local resume state for faster restarts")
            self._write_resume_entries(resume_state, files)
            pending_files = self._iter_resume_entries(resume_state)

        total_entries = self._count_resume_entries(resume_state)

        with self._tqdm(
            total=total_entries or None,
            desc="Downloading files",
            unit="file",
            leave=True,
            position=0,
            colour=self._TQDM_COLOURS[0],
        ) as overall_bar:
            def update_overall_progress() -> None:
                overall_bar.set_postfix_str(
                    f"downloaded={downloaded_files} skipped={skipped_completed} missing={skipped_missing} inflight={inflight_files}"
                )

            update_overall_progress()

            with tempfile.TemporaryDirectory(prefix="osf-download-") as temp_dir:
                with zipfile.ZipFile(
                    target,
                    "a" if target.exists() else "w",
                    zipfile.ZIP_DEFLATED,
                ) as zf:
                    if self.max_workers <= 1:
                        for i, entry in enumerate(pending_files):
                            url, arcname, info_url = self._normalize_file_entry(entry)
                            temp_path: Optional[Path] = None
                            try:
                                if arcname in completed:
                                    skipped_completed += 1
                                    continue

                                if arcname in seen_paths:
                                    skipped_duplicates += 1
                                    continue

                                try:
                                    _, temp_path = self._fetch_zip_entry(
                                        root_url,
                                        url,
                                        arcname,
                                        info_url,
                                        i,
                                        temp_dir,
                                    )
                                except OSFNotFoundError:
                                    skipped_missing += 1
                                    self._status(f"Skipping missing file: {arcname}")
                                    continue

                                zf.write(temp_path, arcname)
                                seen_paths.add(arcname)
                                completed.add(arcname)
                                downloaded_files += 1
                            finally:
                                overall_bar.update(1)
                                update_overall_progress()
                                if temp_path and temp_path.exists():
                                    temp_path.unlink()
                    else:
                        pending_iter = enumerate(pending_files)
                        in_flight: dict[Future[tuple[str, Path]], str] = {}

                        def submit_ready_work() -> None:
                            nonlocal skipped_completed, skipped_duplicates, inflight_files
                            while len(in_flight) < self.max_workers:
                                try:
                                    index, entry = next(pending_iter)
                                except StopIteration:
                                    break

                                url, arcname, info_url = self._normalize_file_entry(entry)
                                if arcname in completed:
                                    skipped_completed += 1
                                    overall_bar.update(1)
                                    update_overall_progress()
                                    continue

                                if arcname in seen_paths:
                                    skipped_duplicates += 1
                                    overall_bar.update(1)
                                    update_overall_progress()
                                    continue

                                future = executor.submit(
                                    self._fetch_zip_entry,
                                    root_url,
                                    url,
                                    arcname,
                                    info_url,
                                    index,
                                    temp_dir,
                                )
                                in_flight[future] = arcname
                                inflight_files = len(in_flight)
                                update_overall_progress()

                        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                            submit_ready_work()

                            while in_flight:
                                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                                for future in done:
                                    arcname = in_flight.pop(future)
                                    inflight_files = len(in_flight)
                                    temp_path: Optional[Path] = None
                                    try:
                                        try:
                                            completed_arcname, temp_path = future.result()
                                        except OSFNotFoundError:
                                            skipped_missing += 1
                                            self._status(f"Skipping missing file: {arcname}")
                                            continue

                                        zf.write(temp_path, completed_arcname)
                                        seen_paths.add(completed_arcname)
                                        completed.add(completed_arcname)
                                        downloaded_files += 1
                                    finally:
                                        overall_bar.update(1)
                                        update_overall_progress()
                                        if temp_path and temp_path.exists():
                                            temp_path.unlink()

                                submit_ready_work()

        if skipped_duplicates:
            self._status(f"Skipped {skipped_duplicates} duplicate file entries")

        if skipped_missing:
            self._status(f"Skipped {skipped_missing} files no longer present on OSF")

        if resume_state.exists():
            resume_state.unlink()

    def _resume_state_path(self, target: Path) -> Path:
        return target.parent / f"{target.name}.resume.jsonl"

    def _write_resume_entries(
        self,
        state_path: Path,
        files: Iterable[tuple[str, ...]],
    ) -> None:
        temp_state = state_path.with_suffix(f"{state_path.suffix}.tmp")
        with open(temp_state, "w", encoding="utf-8") as f:
            for entry in files:
                url, arcname, info_url = self._normalize_file_entry(entry)
                record = {"url": url, "arcname": arcname}
                if info_url:
                    record["info_url"] = info_url
                f.write(json.dumps(record, ensure_ascii=True))
                f.write("\n")

        os.replace(temp_state, state_path)

    def _iter_resume_entries(self, state_path: Path) -> Iterable[tuple[str, str, Optional[str]]]:
        with open(state_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                yield item["url"], item["arcname"], item.get("info_url")

    def _normalize_file_entry(self, entry: tuple[str, ...]) -> tuple[str, str, Optional[str]]:
        if len(entry) == 2:
            url, arcname = entry
            return url, arcname, None
        if len(entry) == 3:
            url, arcname, info_url = entry
            return url, arcname, info_url
        raise OSFError(f"Unexpected file entry format: {entry!r}")

    def _count_resume_entries(self, state_path: Path) -> int:
        with open(state_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def _fetch_zip_entry(
        self,
        root_url: str,
        url: str,
        arcname: str,
        info_url: Optional[str],
        colour_index: int,
        temp_dir: str,
    ) -> tuple[str, Path]:
        current_url = url

        for refresh_attempt in range(self.MAX_RETRIES):
            temp_path: Optional[Path] = None
            download_complete = False
            try:
                fd, temp_name = tempfile.mkstemp(
                    prefix="osf-download-",
                    suffix=".part",
                    dir=temp_dir,
                )
                os.close(fd)
                temp_path = Path(temp_name)

                with self._request_get(current_url, stream=True) as response:
                    total = int(response.headers.get("content-length", 0))

                    with open(temp_path, "wb") as temp_file:
                        if not self.show_progress or self.max_workers > 1:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    temp_file.write(chunk)
                        else:
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
                                for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:
                                        temp_file.write(chunk)
                                        bar.update(len(chunk))

                download_complete = True
                return arcname, temp_path
            except OSFRequestError as e:
                status = self._get_error_status_code(e)
                refreshable = status in self.REFRESHABLE_DOWNLOAD_STATUS_CODES
                if not refreshable or refresh_attempt == self.MAX_RETRIES - 1:
                    raise

                self._status(
                    f"Refreshing expired download URL for {arcname} (attempt {refresh_attempt + 1}/{self.MAX_RETRIES})"
                )
                current_url = self._refresh_download_url(
                    root_url=root_url,
                    arcname=arcname,
                    current_url=current_url,
                    info_url=info_url,
                )
            except requests.RequestException as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                retryable = status in self.RETRYABLE_STATUS_CODES or status is None

                if not retryable or refresh_attempt == self.MAX_RETRIES - 1:
                    raise OSFRequestError(str(e)) from e

                retry_after = self._parse_retry_after(
                    getattr(getattr(e, "response", None), "headers", {}).get("Retry-After")
                )

                if status == 429:
                    if retry_after is not None:
                        delay = retry_after
                    else:
                        delay = max(self.RATE_LIMIT_COOLDOWN, 0.0)
                        delay = min(self.RATE_LIMIT_MAX_COOLDOWN, delay)
                    self._set_rate_limit_window(delay)
                    self._status(
                        f"Rate limited (429) while streaming {arcname}; pausing all requests for {delay:.1f}s (attempt {refresh_attempt + 1}/{self.MAX_RETRIES})"
                    )
                    continue

                delay = retry_after if retry_after is not None else min(
                    self.RETRY_MAX_DELAY,
                    (2**refresh_attempt) + random.uniform(0.0, 1.0),
                )
                self._status(
                    f"Temporary stream error ({status or type(e).__name__}) for {arcname}; retrying in {delay:.1f}s (attempt {refresh_attempt + 1}/{self.MAX_RETRIES})"
                )
                time.sleep(delay)
            finally:
                if temp_path and temp_path.exists() and not download_complete:
                    temp_path.unlink()

    def _refresh_download_url(
        self,
        *,
        root_url: str,
        arcname: str,
        current_url: str,
        info_url: Optional[str],
    ) -> str:
        # Fast path: the per-file API endpoint returns a fresh download URL.
        if info_url:
            try:
                file_payload = self._get_json_url(info_url)
                fresh_url = file_payload["data"]["links"]["download"]
                if fresh_url:
                    return fresh_url
            except (KeyError, OSFRequestError):
                pass

        # If current URL is the stable osf.io download endpoint, retrying it is cheap.
        if "osf.io/download/" in current_url:
            return current_url

        pages_scanned = 0

        def mark_page_scanned() -> None:
            nonlocal pages_scanned
            pages_scanned += 1
            if pages_scanned % 25 == 0:
                self._status(
                    f"Refreshing path scan progress for {arcname}: pages={pages_scanned}"
                )

        try:
            return self._resolve_file_path_with_progress(
                root_url,
                arcname,
                on_page=mark_page_scanned,
            )
        except OSFNotFoundError as refresh_error:
            raise OSFNotFoundError(
                f"Path no longer available when refreshing URL: {arcname}"
            ) from refresh_error

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
            self._wait_for_rate_limit_window()
            self._wait_for_request_slot()
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

                retry_after = self._parse_retry_after(getattr(getattr(e, "response", None), "headers", {}).get("Retry-After"))
                if status == 429:
                    if retry_after is not None:
                        delay = retry_after
                    else:
                        delay = max(self.RATE_LIMIT_COOLDOWN, 0.0)
                        delay = min(self.RATE_LIMIT_MAX_COOLDOWN, delay)
                    self._set_rate_limit_window(delay)
                    self._status(
                        f"Rate limited (429); pausing all requests for {delay:.1f}s (attempt {attempt + 1}/{self.MAX_RETRIES})"
                    )
                    continue

                delay = retry_after if retry_after is not None else min(
                    self.RETRY_MAX_DELAY,
                    (2**attempt) + random.uniform(0.0, 1.0),
                )
                self._status(
                    f"Temporary request error ({status or type(e).__name__}); retrying in {delay:.1f}s (attempt {attempt + 1}/{self.MAX_RETRIES})"
                )
                time.sleep(delay)

        raise OSFRequestError(f"Failed request after {self.MAX_RETRIES} attempts: {url}")

    def _wait_for_rate_limit_window(self) -> None:
        while True:
            with self._rate_limit_lock:
                remaining = self._rate_limited_until - time.monotonic()

            if remaining <= 0:
                return

            time.sleep(min(remaining, 1.0))

    def _set_rate_limit_window(self, delay_seconds: float) -> None:
        with self._rate_limit_lock:
            cooldown_until = time.monotonic() + max(delay_seconds, 0.0)
            if cooldown_until > self._rate_limited_until:
                self._rate_limited_until = cooldown_until

    def _wait_for_request_slot(self) -> None:
        while True:
            with self._request_spacing_lock:
                now = time.monotonic()
                wait_time = self._next_request_at - now
                if wait_time <= 0:
                    self._next_request_at = now + max(self.REQUEST_SPACING, 0.0)
                    return

            time.sleep(min(wait_time, 0.05))

    def _parse_retry_after(self, retry_after: Optional[str]) -> Optional[float]:
        if not retry_after:
            return None

        retry_after = retry_after.strip()
        try:
            value = float(retry_after)
            return value if value >= 0 else None
        except ValueError:
            return None

    def _get_json_url(self, url: str) -> dict:
        try:
            response = self._request_get(url)
            return response.json()
        except ValueError as e:
            raise OSFRequestError(f"Invalid JSON response from {url}") from e

    def _status(self, message: str) -> None:
        if self.console:
            self.console.print(f"[blue]{message}[/blue]")
