"""
Manual DICOM Classification Tool
---------------------------------

Purpose
-------
Fast manual sorting of DICOM images into three classes, with no model
prediction involved. You look at the image and press a key / click a
button to classify it. Nothing is copied or moved - this tool only
writes a CSV log mapping each image's file path to the tag you gave
it. The original DICOM files are never touched.

Background loading
-------------------
Scanning the dataset and hashing every file (needed for duplicate
detection) can take a while on a large archive. Instead of making you
wait for that to finish, this tool scans in a background thread and
streams newly found images into the UI as they're discovered - you
can start classifying almost immediately, and the pool of available
images just keeps growing while you work. If you reach the last
loaded image before scanning is done, it briefly waits for more
instead of ending the session.

Speed: hash cache
------------------
Hashing every file to detect duplicates/already-reviewed images is
the slow part on a large archive, since it means reading every file's
full contents. A local cache (HASH_CACHE_PATH) remembers each file's
hash keyed by its path, size, and modification time - on repeat runs
over a mostly-unchanged dataset, files that haven't changed reuse
their cached hash instead of being re-read from disk, which makes
finding the not-yet-labeled images much faster after the first run.
The very first scan of a new dataset still has to read everything
once; there's no way around that.

Classes
-------
- mid_saggital
- relevant_coronal
- not_relevant
- not_sure

Multi-user CSV
---------------
Since several people can classify against the same shared CSV, every
row also records who made that call (the "user" column). On startup
you're asked for your name/initials (defaults to your OS username),
and before scanning starts the tool reads the existing CSV and prints
a breakdown of how many images have been classified into each class
so far, and by whom.

Duplicate handling
-------------------
Files are de-duplicated by content (MD5 hash of the raw file bytes),
not by filename or path. If the exact same file exists under several
paths (e.g. copied into multiple folders), it is only shown once. The
hash is stored in the CSV, so duplicates are still recognized and
skipped on later runs even if a different one of their paths is the
one found first.

Correction behaviour
---------------------
If you go back with "Previous" and re-classify an image you already
labeled, the tool simply overwrites its tag (and user) in the CSV -
there is no file to move since nothing is copied.

Display
-------
The window and the image display area are sized relative to the
actual screen resolution at startup (detected via Tkinter), so the
tool comes up at a sensible, centered size on any monitor instead of
a fixed pixel size. It does not resize or reposition itself based on
the dimensions of whichever image is currently shown.

Keyboard
--------
Right Arrow : next image (skip, no label)
Left Arrow  : previous image

1 : mid saggital
2 : relevant coronal
3 : not relevant
n : not sure

Requirements
------------
pip install pydicom pillow pandas numpy
"""

import os
import io
import random
import hashlib
import threading
import queue
import getpass

import numpy as np
import pandas as pd
import pydicom

from PIL import Image

import tkinter as tk
from tkinter import messagebox, simpledialog

# =========================================================
# CONFIG - EDIT THESE
# =========================================================

# Folder containing the DICOM files to review (searched recursively)
DATA_PATH = "/path/to/dicom_root/"

# CSV log: one row per unique image (by content hash) with its tag.
REVIEW_CSV = "/path/to/review_results.csv"

# Local cache of file hashes, keyed by (path, size, mtime), so repeat
# scans of an unchanged dataset don't need to re-read every file just
# to recompute a hash it already knows. Safe to delete any time - the
# next scan will just be slower while it rebuilds.
HASH_CACHE_PATH = os.path.join(
    os.path.dirname(REVIEW_CSV) or ".", "hash_cache.csv"
)

CLASSES = ["mid_saggital", "relevant_coronal", "not_relevant"]
NOT_SURE = "not_sure"

# Set this to skip the startup name prompt (e.g. REVIEWER_NAME = "jsmith").
# Leave as None to be asked for a name/initials each time the tool starts.
REVIEWER_NAME = None

RANDOM_SEED = 42

# Newly found images are buffered and locally shuffled in chunks of
# this size before being added to the review queue, so you still get
# a randomized order without waiting for the entire dataset to be
# scanned first.
SHUFFLE = True
SHUFFLE_BUFFER_SIZE = 40

# Fraction of the detected screen size the window and the image
# display area should occupy. Kept below 1.0 so there's still room
# for window decorations / taskbars. The window is still fixed once
# computed - it does not resize/reposition based on image content.
WINDOW_WIDTH_FRACTION = 0.85
WINDOW_HEIGHT_FRACTION = 0.85
IMAGE_WIDTH_FRACTION = 0.65   # of the window width
IMAGE_HEIGHT_FRACTION = 0.72  # of the window height

STATS_PANEL_WIDTH = 280

# how often (in ms) the live stats panel re-reads the CSV from disk,
# so progress made by other reviewers on the same shared file shows
# up without restarting the tool
STATS_REFRESH_INTERVAL_MS = 4000

# how often (in DICOM files processed) the background thread prints
# a progress update to the console and flushes the hash cache to disk
PROGRESS_EVERY = 25

# how often (in ms) the UI checks for newly loaded images
POLL_INTERVAL_MS = 150

random.seed(RANDOM_SEED)


# =========================================================
# HELPERS
# =========================================================

def compute_file_hash(path, chunk_size=1 << 20):
    """MD5 of the raw file bytes, used to identify duplicate files
    regardless of their path/filename."""

    h = hashlib.md5()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def pil_to_tk_photoimage(pil_image):
    """Convert a PIL image to a tkinter PhotoImage via an in-memory
    PPM buffer, instead of PIL.ImageTk.PhotoImage.

    PIL.ImageTk relies on a compiled _imagingtk extension that
    PyInstaller frequently fails to bundle correctly (you'd see a
    'PyImagingPhoto' / 'No module named PIL._tkinter_finder' error
    in the packaged exe). Going through PPM bytes uses only
    tkinter's own built-in PhotoImage decoder, so it works
    identically whether run as a script or as a frozen executable.
    """

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PPM")
    return tk.PhotoImage(data=buffer.getvalue())


def is_dicom_file(full_path, filename):

    lower = filename.lower()

    if lower.endswith(".dcm"):
        return True

    # extension-less files: cheap sniff on the DICM magic bytes at
    # offset 128, common for clinical exports without a file extension
    if "." not in filename:

        try:
            with open(full_path, "rb") as f:
                f.seek(128)
                if f.read(4) == b"DICM":
                    return True
        except OSError:
            pass

    return False


def dicom_to_pil_image(dicom_path):
    """Load a DICOM file and convert its pixel data to a displayable
    8-bit RGB PIL image. Applies VOI LUT / windowing when available
    and handles MONOCHROME1 inversion."""

    ds = pydicom.dcmread(dicom_path)

    arr = ds.pixel_array.astype(np.float32)

    # collapse extra dimensions (e.g. multi-frame) to the first frame
    if arr.ndim > 2:
        arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr[..., 0]

    try:
        from pydicom.pixel_data_handlers.util import apply_voi_lut
        arr = apply_voi_lut(arr, ds).astype(np.float32)
    except Exception:
        pass

    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = arr.max() - arr

    arr_min, arr_max = arr.min(), arr.max()

    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
    else:
        arr = np.zeros_like(arr)

    arr = arr.astype(np.uint8)

    image = Image.fromarray(arr)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


# =========================================================
# HASH CACHE (path+size+mtime -> hash), persisted across runs
# This is what makes finding not-yet-labeled images fast on repeat
# runs: unchanged files reuse their cached hash instead of being
# re-read from disk.
# =========================================================

hash_cache = {}  # path -> {"size": int, "mtime": float, "hash": str}
hash_cache_lock = threading.Lock()
hash_cache_dirty = False

if os.path.exists(HASH_CACHE_PATH):
    try:
        df_cache = pd.read_csv(HASH_CACHE_PATH)
        for _, row in df_cache.iterrows():
            hash_cache[row["path"]] = {
                "size": row["size"],
                "mtime": row["mtime"],
                "hash": row["hash"],
            }
        print(f"Loaded hash cache: {len(hash_cache)} entries")
    except Exception as e:
        print(f"Could not read hash cache ({e}), starting fresh")


def flush_hash_cache():

    global hash_cache_dirty

    with hash_cache_lock:
        if not hash_cache_dirty:
            return
        rows = [
            {"path": p, "size": v["size"], "mtime": v["mtime"], "hash": v["hash"]}
            for p, v in hash_cache.items()
        ]
        hash_cache_dirty = False

    pd.DataFrame(rows).to_csv(HASH_CACHE_PATH, index=False)


def get_file_hash_cached(path):
    """Returns the MD5 hash of path, reusing the cache when the
    file's size and mtime haven't changed since it was last hashed."""

    global hash_cache_dirty

    stat = os.stat(path)
    size, mtime = stat.st_size, stat.st_mtime

    cached = hash_cache.get(path)

    if cached and cached["size"] == size and cached["mtime"] == mtime:
        return cached["hash"]

    file_hash = compute_file_hash(path)

    with hash_cache_lock:
        hash_cache[path] = {"size": size, "mtime": mtime, "hash": file_hash}
        hash_cache_dirty = True

    return file_hash


# =========================================================
# LOAD EXISTING REVIEW LOG
# =========================================================

os.makedirs(os.path.dirname(REVIEW_CSV) or ".", exist_ok=True)

CLASS_DISPLAY_NAMES = {
    "mid_saggital": "Mid Saggital",
    "relevant_coronal": "Relevant Coronal",
    "not_relevant": "Not Relevant",
    NOT_SURE: "Not Sure",
}

ALL_LABELS = CLASSES + [NOT_SURE]


def load_results_from_csv():
    """Reads REVIEW_CSV fresh from disk and returns a dict keyed by
    content hash. Called at startup and again periodically while the
    app is open, so that progress made by other people working off
    the same shared CSV shows up without restarting the tool."""

    results = {}

    if not os.path.exists(REVIEW_CSV):
        return results

    try:

        df = pd.read_csv(REVIEW_CSV)

        has_user_col = "user" in df.columns

        for _, row in df.iterrows():
            results[row["hash"]] = {
                "hash": row["hash"],
                "image_path": row["image_path"],
                "image": row["image"],
                "user_label": row["user_label"],
                "user": row["user"] if has_user_col else "unknown",
            }

    except Exception as e:
        print(f"Could not read existing CSV: {e}")

    return results


# keyed by content hash
existing_results = load_results_from_csv()

print(f"Previously reviewed images: {len(existing_results)}")


def write_review_csv():

    df = pd.DataFrame(list(existing_results.values()))
    df.to_csv(REVIEW_CSV, index=False)


def build_stats_lines(results):
    """Builds the lines shown in the live progress panel: total
    reviewed, a count per class (including classes with 0 so far),
    and a count per reviewer."""

    if not results:
        return ["No images classified yet."]

    df = pd.DataFrame(list(results.values()))

    lines = [f"Total classified: {len(df)}", "", "By class:"]

    counts = df["user_label"].value_counts()

    for label in ALL_LABELS:
        display = CLASS_DISPLAY_NAMES.get(label, label)
        count = int(counts.get(label, 0))
        lines.append(f"  {display}: {count}")

    if "user" in df.columns:
        lines.append("")
        lines.append("By reviewer:")
        for user, count in df["user"].value_counts().items():
            lines.append(f"  {user}: {count}")

    return lines


# =========================================================
# BACKGROUND SCAN WORKER
# =========================================================

def scan_worker(root_path, already_reviewed, result_queue):
    """Walks root_path, hashes every DICOM file it finds (reusing the
    on-disk hash cache for unchanged files), and pushes newly-found
    (not a duplicate, not already reviewed) images onto result_queue
    as it goes. Runs in a background thread so the UI doesn't have to
    wait for the full scan to finish."""

    print("Scanning for DICOM files (running in background)...")

    seen_hashes = set()

    processed = 0
    queued = 0
    duplicates_skipped = 0
    already_reviewed_skipped = 0

    try:

        for root, _, files in os.walk(root_path):

            for filename in files:

                full_path = os.path.join(root, filename)

                if not is_dicom_file(full_path, filename):
                    continue

                processed += 1

                try:
                    file_hash = get_file_hash_cached(full_path)
                except OSError as e:
                    print(f"  could not read {full_path}: {e}")
                    continue

                if processed % PROGRESS_EVERY == 0:
                    print(
                        f"  ...processed {processed} DICOM files "
                        f"({queued} queued for review, "
                        f"{duplicates_skipped} duplicates, "
                        f"{already_reviewed_skipped} already reviewed)",
                        end="\r",
                    )
                    flush_hash_cache()

                if file_hash in seen_hashes:
                    duplicates_skipped += 1
                    continue

                seen_hashes.add(file_hash)

                if file_hash in already_reviewed:
                    already_reviewed_skipped += 1
                    continue

                queued += 1
                result_queue.put(("image", full_path, file_hash))

    except Exception as e:
        print(f"\nScan error: {e}")

    flush_hash_cache()

    print()
    print(
        f"Scan complete. Processed {processed} DICOM files, "
        f"{queued} queued for review, "
        f"{duplicates_skipped} duplicates skipped, "
        f"{already_reviewed_skipped} already reviewed."
    )

    result_queue.put(("done", processed, queued, duplicates_skipped, already_reviewed_skipped))


# =========================================================
# APP
# =========================================================

class ReviewApp:

    def __init__(self, root, reviewer_name):

        self.root = root
        self.root.title("DICOM Classification Tool")
        self.reviewer_name = reviewer_name

        # ---------------------------------------------------
        # ADAPT TO DISPLAY SIZE
        # Window and image area are sized relative to the actual
        # detected screen resolution, and centered. The window is
        # still fixed once computed - it does not resize/reposition
        # itself based on whichever image is currently shown.
        # ---------------------------------------------------

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()

        window_w = int(screen_w * WINDOW_WIDTH_FRACTION)
        window_h = int(screen_h * WINDOW_HEIGHT_FRACTION)

        pos_x = (screen_w - window_w) // 2
        pos_y = (screen_h - window_h) // 2

        self.root.geometry(f"{window_w}x{window_h}+{pos_x}+{pos_y}")
        self.root.resizable(False, False)

        self.display_max_size = (
            int(window_w * IMAGE_WIDTH_FRACTION),
            int(window_h * IMAGE_HEIGHT_FRACTION),
        )

        self.history = []
        self.current_pos = 0

        self.image_paths = []
        self.image_hashes = {}
        self.pending_buffer = []

        self.loading_done = False
        self.waiting_for_image = True

        self.result_queue = queue.Queue()

        # ---------------------------------------------------
        # UI - split into a left content area and a right
        # "Classification Progress" panel that stays visible next
        # to the image at all times.
        # ---------------------------------------------------

        content_frame = tk.Frame(root)
        content_frame.pack(side="left", fill="both", expand=True)

        stats_frame = tk.Frame(
            root, width=STATS_PANEL_WIDTH, relief="groove", borderwidth=1
        )
        stats_frame.pack(side="right", fill="y", padx=(0, 10), pady=10)
        stats_frame.pack_propagate(False)

        tk.Label(
            stats_frame, text="Classification Progress",
            font=("Arial", 13, "bold")
        ).pack(pady=(12, 4), padx=10, anchor="w")

        self.stats_label = tk.Label(
            stats_frame, text="", font=("Arial", 11),
            justify="left", anchor="nw"
        )
        self.stats_label.pack(fill="both", expand=True, padx=12, pady=6)

        self.reviewer_label = tk.Label(
            content_frame, text=f"Reviewer: {self.reviewer_name}", font=("Arial", 10)
        )
        self.reviewer_label.pack(pady=(8, 0))

        self.image_frame = tk.Frame(
            content_frame,
            width=self.display_max_size[0],
            height=self.display_max_size[1],
        )
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(expand=True)

        self.info_label = tk.Label(content_frame, text="", font=("Arial", 14))
        self.info_label.pack()

        self.label_status = tk.Label(content_frame, text="", font=("Arial", 16, "bold"))
        self.label_status.pack(pady=10)

        button_frame = tk.Frame(content_frame)
        button_frame.pack(pady=10)

        tk.Button(
            button_frame, text="<< Prev", width=14,
            command=self.prev_image
        ).grid(row=0, column=0, padx=5)

        tk.Button(
            button_frame, text="Mid Saggital (1)", width=16,
            bg="#4C72B0", fg="white",
            command=lambda: self.save_label("mid_saggital")
        ).grid(row=0, column=1, padx=5)

        tk.Button(
            button_frame, text="Relevant Coronal (2)", width=18,
            bg="#DD8452", fg="white",
            command=lambda: self.save_label("relevant_coronal")
        ).grid(row=0, column=2, padx=5)

        tk.Button(
            button_frame, text="Not Relevant (3)", width=16,
            bg="#55A868", fg="white",
            command=lambda: self.save_label("not_relevant")
        ).grid(row=0, column=3, padx=5)

        tk.Button(
            button_frame, text="Not Sure (n)", width=14,
            bg="orange",
            command=lambda: self.save_label(NOT_SURE)
        ).grid(row=0, column=4, padx=5)

        tk.Button(
            button_frame, text="Next >>", width=14,
            command=self.next_image
        ).grid(row=0, column=5, padx=5)

        # ---------------------------------------------------
        # KEYBOARD SHORTCUTS
        # ---------------------------------------------------

        root.bind("<Right>", lambda e: self.next_image())
        root.bind("<Left>", lambda e: self.prev_image())

        root.bind("1", lambda e: self.save_label("mid_saggital"))
        root.bind("2", lambda e: self.save_label("relevant_coronal"))
        root.bind("3", lambda e: self.save_label("not_relevant"))
        root.bind("n", lambda e: self.save_label(NOT_SURE))

        # ---------------------------------------------------
        # LIVE STATS PANEL
        # ---------------------------------------------------

        self.update_stats_panel()
        self.root.after(STATS_REFRESH_INTERVAL_MS, self.periodic_stats_refresh)

        # ---------------------------------------------------
        # KICK OFF BACKGROUND SCAN + SHOW PLACEHOLDER
        # ---------------------------------------------------

        self._try_display()

        self.worker_thread = threading.Thread(
            target=scan_worker,
            args=(DATA_PATH, existing_results, self.result_queue),
            daemon=True,
        )
        self.worker_thread.start()

        self.root.after(POLL_INTERVAL_MS, self.poll_queue)

    # =====================================================
    # LIVE STATS PANEL
    # =====================================================

    def update_stats_panel(self):
        """Refreshes the on-screen progress panel from whatever is
        currently in the in-memory existing_results dict."""

        lines = build_stats_lines(existing_results)
        self.stats_label.config(text="\n".join(lines))

    def periodic_stats_refresh(self):
        """Re-reads the shared CSV from disk so progress made by
        other reviewers working off the same file shows up here
        without needing to restart the tool, then reschedules
        itself."""

        global existing_results

        fresh = load_results_from_csv()

        if fresh:
            existing_results = fresh

        self.update_stats_panel()

        # the currently displayed image's "current label" line may
        # have changed if someone else just (re)labeled it
        if not self.waiting_for_image and self.current_pos < len(self.image_paths):
            image_path = self.image_paths[self.current_pos]
            file_hash = self.image_hashes.get(image_path)
            current_label = existing_results.get(file_hash, {}).get("user_label")
            if current_label:
                self.label_status.config(text=f"Current label: {current_label}")
            else:
                self.label_status.config(text="Not yet labeled")

        self.root.after(STATS_REFRESH_INTERVAL_MS, self.periodic_stats_refresh)

    # =====================================================
    # BACKGROUND LOADING
    # =====================================================

    def poll_queue(self):

        try:
            while True:

                item = self.result_queue.get_nowait()

                if item[0] == "image":
                    _, path, file_hash = item
                    self.pending_buffer.append((path, file_hash))

                elif item[0] == "done":
                    self.loading_done = True

        except queue.Empty:
            pass

        self._flush_buffer()

        if self.waiting_for_image:
            self._try_display()

        queue_empty = self.result_queue.empty()

        if not self.loading_done or not queue_empty or self.pending_buffer:
            self.root.after(POLL_INTERVAL_MS, self.poll_queue)

    def _flush_buffer(self):

        if not self.pending_buffer:
            return

        if len(self.pending_buffer) < SHUFFLE_BUFFER_SIZE and not self.loading_done:
            # wait for a bigger chunk before shuffling it in, unless
            # scanning has already finished
            return

        if SHUFFLE:
            random.shuffle(self.pending_buffer)

        for path, file_hash in self.pending_buffer:
            self.image_paths.append(path)
            self.image_hashes[path] = file_hash

        self.pending_buffer = []

    # =====================================================
    # DISPLAY IMAGE
    # =====================================================

    def _try_display(self):
        """Show the image at current_pos if it's loaded yet. If not,
        either wait (still scanning) or end the session (scanning
        finished and there's truly nothing more)."""

        if self.current_pos < len(self.image_paths):
            self.waiting_for_image = False
            self.display_image()
            return

        if self.loading_done:
            self.waiting_for_image = False
            self.image_label.configure(image="")
            self.label_status.config(text="")
            self.info_label.config(text="No more images to review.")
            messagebox.showinfo("Done", "No more images to review.")
            self.root.destroy()
            return

        self.waiting_for_image = True
        self.image_label.configure(image="")
        self.label_status.config(text="")
        self.info_label.config(
            text=f"Loading images... ({len(self.image_paths)} ready so far)"
        )

    def display_image(self):

        image_path = self.image_paths[self.current_pos]

        try:
            image = dicom_to_pil_image(image_path)
        except Exception as e:
            messagebox.showwarning(
                "Read error",
                f"Could not read {os.path.basename(image_path)}: {e}"
            )
            self.next_image()
            return

        image.thumbnail(self.display_max_size)

        tk_image = pil_to_tk_photoimage(image)

        self.image_label.configure(image=tk_image)
        self.image_label.image = tk_image

        loaded_marker = "" if self.loading_done else " (still loading more...)"

        self.info_label.config(
            text=(
                f"{self.current_pos + 1}/{len(self.image_paths)}{loaded_marker}\n"
                f"{os.path.basename(image_path)}"
            )
        )

        file_hash = self.image_hashes[image_path]
        current_label = existing_results.get(file_hash, {}).get("user_label")

        if current_label:
            self.label_status.config(text=f"Current label: {current_label}")
        else:
            self.label_status.config(text="Not yet labeled")

    # =====================================================
    # SAVE / CORRECT LABEL
    # =====================================================

    def save_label(self, user_label):

        if self.current_pos >= len(self.image_paths):
            return

        image_path = self.image_paths[self.current_pos]
        filename = os.path.basename(image_path)
        file_hash = self.image_hashes[image_path]

        previous = existing_results.get(file_hash)

        existing_results[file_hash] = {
            "hash": file_hash,
            "image_path": image_path,
            "image": filename,
            "user_label": user_label,
            "user": self.reviewer_name,
        }

        write_review_csv()

        action = "Corrected" if previous is not None else "Saved"
        print(f"{action} -> {filename} => {user_label} (by {self.reviewer_name})")

        self.update_stats_panel()

        self.next_image()

    # =====================================================
    # NEXT / PREVIOUS
    # =====================================================

    def next_image(self):

        self.history.append(self.current_pos)
        self.current_pos += 1
        self._try_display()

    def prev_image(self):

        if len(self.history) == 0:
            return

        self.current_pos = self.history.pop()
        self._try_display()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    root = tk.Tk()
    root.withdraw()  # hide the main window until setup is done

    reviewer_name = REVIEWER_NAME

    if not reviewer_name:

        default_name = getpass.getuser()

        reviewer_name = simpledialog.askstring(
            "Reviewer Name",
            "Enter your name or initials:",
            initialvalue=default_name,
            parent=root,
        )

        if not reviewer_name:
            reviewer_name = default_name

    root.deiconify()

    app = ReviewApp(root, reviewer_name)
    root.mainloop()