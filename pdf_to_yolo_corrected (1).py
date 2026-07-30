"""
Colab script: tagged PDFs -> YOLO detection dataset (images/ + labels/).
CORRECTED VERSION. Paste each "# %% CELL" block into its own Colab cell.

WHAT CHANGED AND WHY (all verified empirically against MuPDF 1.29 /
PyMuPDF 1.28, not assumed):

1. ALIGNMENT MODEL WAS WRONG. bboxlog does not emit one entry per paint
   operator. For text it emits one entry per *flush* of the accumulated
   text object. Measured flush triggers: ET, BDC, BMC, EMC, q, Q, gs, Tr,
   and colour operators (g/rg/k/sc/scn/cs + stroke variants), plus any
   path paint. NOT flush triggers: Tf, Td/TD/Tm/T*, or consecutive Tj/TJ.
   So `BT (a) Tj (b) Tj ET` is ONE entry, not two -- the old walk desynced
   on the first ordinary two-line paragraph and silently mis-attributed
   every box after it on the page.

2. FORM XOBJECTS. A `Do` on a form consumed one entry while MuPDF emitted
   one per paint op inside the form, desyncing the remainder of the page.
   Tagged content inside the form (under its own /StructParents) was also
   invisible. Now recursed into properly.

3. ROTATION. bboxlog is in UNROTATED page space; page.rect and get_pixmap
   are both rotated. Every box on a /Rotate 90 page was wrong. Now
   multiplied through page.rotation_matrix.

4. BOX GRANULARITY. Keying on MCID shredded a paragraph into one box per
   line. Now unioned by the owning StructElem's objgen, so one LBody =
   one box.

5. SIZE FILTER. MIN_BOX_FRACTION=0.0003 is ~145 pt^2 on letter, which
   deletes every list bullet (~50 pt^2) and starves the Lbl class. Now a
   minimum in POINTS on each dimension.

6. QA GATE. Every desync mode found in testing produces a mismatch between
   paint ops consumed and real paint entries available. Pages that
   mismatch are quarantined instead of silently emitting bad labels.

7. Plus: inline images counted; non-paint bboxlog kinds (clip-*, group,
   layer) skipped rather than consumed; /Resources inheritance; sha1-based
   filenames (basenames collide under recursive glob); no image emitted
   with an empty label file; document-level train/val split.
"""

# %% CELL 1 - install deps (Colab)
pip install pikepdf pymupdf tqdm -q


# %% CELL 2 - config
import os

# Fixed LONG EDGE beats fixed DPI here: DocLayout-YOLO trains at imgsz=1024
# and letterboxes anyway, so anything larger is thrown away, and fixed DPI
# gives wildly different pixel sizes across A4 / letter / tabloid.
TARGET_LONG_EDGE = 1024
JPEG_QUALITY = 92
MIN_BOX_PT = 2.0          # minimum width AND height, in points
VAL_PCT = 10              # per-document split, deterministic from content hash
OUTPUT_DIR = "yolo_dataset"

# Matches the frozen LayoutLMv3 label schema (cell/item level, not
# container level). Table/L/LI/TR/THead/TBody are pure containers per the
# tagging ruleset -- they never own content, so they are not classes.
# Indices 0-13 are FROZEN to match the existing fine-tuned LayoutLMv3 label
# set. New classes are appended at 14+ so that parity is preserved; note
# that the YOLO schema is now a SUPERSET of the LayoutLMv3 one, so any
# cross-model comparison must be restricted to ids 0-13.
CLASS_MAP = {
    "H1": 0, "H2": 1, "H3": 2, "H4": 3, "H5": 4, "H6": 5,
    "P": 6, "Lbl": 7, "LBody": 8, "TH": 9, "TD": 10,
    "Caption": 11, "Figure": 12, "Formula": 13,
    "TOCI": 14, "Form": 15, "Note": 16,
}

# Explicit precedence, most specific first. The old "nearest specific
# ancestor wins, with P/Span treated as generic" heuristic breaks down once
# TOC exists: a TOC entry's chain is Link > Reference > TOCI > TOC, so
# "nearest" would pick Link. Ranking makes the intent declarative.
#
# Among tags of EQUAL rank the CLOSEST occurrence wins, which is what makes
# nested lists work (chain P > LBody > LI > L > LBody > LI > L resolves to
# the inner LBody).
#
# Tags absent from this list are transparent: resolution passes straight
# through them to the next ranked ancestor. That is deliberate for Link and
# Reference -- an inline hyperlink inside a paragraph resolves to the
# PARENT P and unions into it, instead of punching a hole in the paragraph
# box. Add "Link" here only if you actually want links as separate
# detections, and accept that paragraphs containing them will be split.
CLASS_PRIORITY = {
    tag: i for i, tag in enumerate([
        "Formula", "Figure", "Caption",
        "TH", "TD", "Lbl", "LBody",
        "TOCI", "Form", "Note",
        "H1", "H2", "H3", "H4", "H5", "H6",
        "P",
    ])
}

# Classes whose boxes may come from an OBJR-associated annotation rather
# than from marked content. Form fields have NO marked content at all --
# a widget's appearance is drawn from its /AP stream, so it never appears
# in the content stream or in bboxlog, and the MCID path can never see it.
#
# TOCI is deliberately NOT here: a TOC entry's visible text IS marked
# content, so it already gets a good box from the MCID path, and unioning
# in the link annotation's /Rect (which is often padded well beyond the
# glyphs) would only inflate it.
OBJR_CLASSES = {"Form"}

CLASS_NAMES = [n for n, _ in sorted(CLASS_MAP.items(), key=lambda kv: kv[1])]


# %% CELL 3 - extraction core
import pikepdf

PAINT_KINDS = {
    "fill-text", "stroke-text", "ignore-text",
    "fill-path", "stroke-path", "ignore-path",
    "fill-image", "fill-shade",
}
TEXT_SHOW_OPS = {"Tj", "TJ", "'", '"'}
PATH_PAINT_OPS = {"f", "F", "f*", "B", "B*", "b", "b*", "S", "s"}
FLUSH_OPS = {"ET", "q", "Q", "gs", "Tr",
             "g", "G", "rg", "RG", "k", "K",
             "sc", "SC", "scn", "SCN", "cs", "CS"}
MAX_FORM_DEPTH = 12


def _flatten_number_tree(node, out=None, depth=0):
    if out is None:
        out = {}
    if node is None or depth > 32:
        return out
    if "/Kids" in node:
        for kid in node.Kids:
            _flatten_number_tree(kid, out, depth + 1)
    elif "/Nums" in node:
        nums = node.Nums
        for i in range(0, len(nums) - 1, 2):
            try:
                out[int(nums[i])] = nums[i + 1]
            except (TypeError, ValueError):
                continue
    return out


def _parent_tree(pdf):
    root = pdf.Root.get("/StructTreeRoot")
    if root is None or "/ParentTree" not in root:
        return None, {}
    return root, _flatten_number_tree(root.ParentTree)


def _mcid_map_for_key(flat, key):
    """ParentTree entry -> {mcid: StructElem}. Entry is an array indexed by
    MCID; nulls are legal for MCIDs with no owner."""
    if key is None:
        return {}
    entry = flat.get(int(key))
    if entry is None:
        return {}
    if isinstance(entry, pikepdf.Dictionary):
        return {0: entry}
    out = {}
    for i, elem in enumerate(entry):
        if elem is not None and isinstance(elem, pikepdf.Dictionary):
            out[i] = elem
    return out


def _ancestor_chain(elem, max_depth=24):
    """Leaf-to-root list of (tag_name, objgen). objgen lets the caller union
    boxes at ELEMENT level instead of MCID level."""
    chain, node, depth = [], elem, 0
    while node is not None and "/S" in node and depth < max_depth:
        try:
            gen = tuple(node.objgen)
        except Exception:
            gen = (id(node), 0)
        chain.append((str(node.S).lstrip("/"), gen))
        node = node.get("/P")
        if node is not None and node.get("/Type") == pikepdf.Name.StructTreeRoot:
            break
        depth += 1
    return chain


class _Ctx:
    def __init__(self, bboxlog):
        self.bboxlog = bboxlog
        self.idx = 0
        self.consumed = 0
        self.overrun = False
        self.boxes = {}

    def take(self, owner):
        bl = self.bboxlog
        while self.idx < len(bl) and bl[self.idx][0] not in PAINT_KINDS:
            self.idx += 1          # skip clip-*, group, layer
        if self.idx >= len(bl):
            self.overrun = True
            return
        _kind, bb = bl[self.idx]
        self.idx += 1
        self.consumed += 1
        if owner is None:
            return
        x0, y0, x1, y1 = bb
        cur = self.boxes.get(owner)
        if cur is None:
            self.boxes[owner] = [x0, y0, x1, y1]
        else:
            cur[0] = min(cur[0], x0); cur[1] = min(cur[1], y0)
            cur[2] = max(cur[2], x1); cur[3] = max(cur[3], y1)


def _innermost(mc_stack):
    return next((m for m in reversed(mc_stack) if m is not None), None)


def _bdc_mcid(instr, resources):
    """/MCID from a BDC: inline dict form, or named form indirecting through
    /Resources/Properties."""
    if len(instr.operands) < 2:
        return None
    props = instr.operands[1]
    if isinstance(props, pikepdf.Dictionary):
        return int(props.MCID) if "/MCID" in props else None
    if isinstance(props, pikepdf.Name) and resources is not None:
        table = resources.get("/Properties")
        if table is not None:
            ref = table.get(str(props))
            if ref is not None and "/MCID" in ref:
                return int(ref.MCID)
    return None


def _run_stream(ctx, stream_obj, resources, scope, flat, mc_stack, depth):
    try:
        instructions = pikepdf.parse_content_stream(stream_obj)
    except (pikepdf.PdfError, TypeError, ValueError):
        ctx.overrun = True
        return

    pending_owner, has_pending = None, False

    def flush():
        nonlocal has_pending, pending_owner
        if has_pending:
            ctx.take(pending_owner)
            has_pending, pending_owner = False, None

    def owner_now():
        m = _innermost(mc_stack)
        return None if m is None else (scope, m)

    for instr in instructions:
        op = str(instr.operator)

        if op in TEXT_SHOW_OPS:
            if not has_pending:
                pending_owner, has_pending = owner_now(), True
            continue
        if op == "BDC":
            flush(); mc_stack.append(_bdc_mcid(instr, resources)); continue
        if op == "BMC":
            flush(); mc_stack.append(None); continue      # Artifact etc.
        if op == "EMC":
            flush()                                       # flush BEFORE popping
            if mc_stack:
                mc_stack.pop()
            continue
        if op in FLUSH_OPS:
            flush(); continue
        if op in PATH_PAINT_OPS or op == "sh":
            flush(); ctx.take(owner_now()); continue
        if op == "INLINE IMAGE":
            flush(); ctx.take(owner_now()); continue
        if op == "Do":
            flush()
            xobj = None
            if resources is not None and instr.operands:
                table = resources.get("/XObject")
                if table is not None:
                    xobj = table.get(str(instr.operands[0]))
            if xobj is None:
                continue
            subtype = xobj.get("/Subtype")
            if subtype == pikepdf.Name.Image:
                ctx.take(owner_now())
            elif subtype == pikepdf.Name.Form and depth < MAX_FORM_DEPTH:
                sp = xobj.get("/StructParents")
                sub_scope = ("form", int(sp)) if sp is not None else scope
                sub_res = xobj.get("/Resources") or resources
                _run_stream(ctx, xobj, sub_res, sub_scope, flat,
                            mc_stack, depth + 1)
            else:
                ctx.overrun = True     # unknown xobject -> can't stay aligned
            continue

    flush()


# Annotation flags (PDF 32000-1 table 165): bit 2 Hidden, bit 6 NoView.
ANNOT_HIDDEN, ANNOT_NOVIEW = 2, 32


def _resolve_appearance(annot):
    """Annotation's normal appearance stream, resolving the /AS state
    sub-dictionary used by checkboxes and radio buttons."""
    ap = annot.get("/AP")
    if ap is None:
        return None
    n = ap.get("/N")
    if isinstance(n, pikepdf.Stream):
        return n
    if isinstance(n, pikepdf.Dictionary):
        state = annot.get("/AS")
        if state is not None:
            cand = n.get(str(state))
            return cand if isinstance(cand, pikepdf.Stream) else None
        vals = [v for v in n.values() if isinstance(v, pikepdf.Stream)]
        return vals[0] if len(vals) == 1 else None
    return None


def _visible_annots(page_pk):
    """Annotations MuPDF will actually paint, in /Annots order."""
    annots = page_pk.obj.get("/Annots")
    if annots is None:
        return []
    out = []
    for a in annots:
        try:
            if a.get("/Subtype") == pikepdf.Name.Popup:
                continue
            flags = int(a.get("/F", 0))
            if flags & ANNOT_HIDDEN or flags & ANNOT_NOVIEW:
                continue
            out.append(a)
        except (TypeError, ValueError, AttributeError):
            continue
    return out


def _run_annot_appearances(ctx, page_pk, flat):
    """MuPDF paints annotation appearance streams AFTER all page content,
    in /Annots order (verified). Those paints land in bboxlog but are NOT
    in the page content stream, so without walking them here the alignment
    counters would mismatch on every page carrying a visible form field or
    bordered link -- i.e. it would quarantine precisely the pages a forms
    corpus is made of.

    Walking them also yields the DRAWN appearance bbox, which is tighter
    and more honest than /Rect (frequently padded well beyond the ink).
    """
    for a in _visible_annots(page_pk):
        ap = _resolve_appearance(a)
        if ap is None:
            continue                    # nothing painted, nothing to count
        try:
            xref = a.objgen[0]
        except Exception:
            continue
        # seed the MC stack so paints attribute to this annotation
        _run_stream(ctx, ap, ap.get("/Resources"), ("annot", xref),
                    flat, [0], 1)


def _annot_boxes(page_pk, page_fz, flat, ctx):
    """OBJR-associated structure elements -- Form widgets, Links, TOC link
    targets. These are reached through the ANNOTATION's /StructParent
    (singular, resolving to one StructElem) rather than the page's
    /StructParents (plural, resolving to an array indexed by MCID).

    Form widgets in particular have no marked content whatsoever: the
    appearance is drawn from the annotation's /AP stream, so nothing ever
    reaches the content stream or bboxlog. The MCID path is structurally
    incapable of seeing them.

    /Rect is in PDF user space (bottom-left origin). MuPDF page space has
    its origin at the CropBox top-left with y flipped, and is UNROTATED --
    verified against page.widgets(), which returns identical rects for
    /Rotate 0 and /Rotate 90. So these boxes take exactly the same
    rotation_matrix treatment as bboxlog boxes downstream.
    """
    annots = page_pk.obj.get("/Annots")
    if annots is None:
        return []
    cb = page_fz.cropbox
    x_off, y_top = cb.x0, cb.y1

    out = []
    for a in _visible_annots(page_pk):
        try:
            sp = a.get("/StructParent")
            rect = a.get("/Rect")
            if sp is None or rect is None or len(rect) != 4:
                continue
            elem = flat.get(int(sp))
            if not isinstance(elem, pikepdf.Dictionary) or "/S" not in elem:
                continue
            x0, y0, x1, y1 = (float(v) for v in rect)
        except (TypeError, ValueError, AttributeError):
            continue
        drawn = ctx.boxes.get((("annot", a.objgen[0]), 0))
        if drawn is not None:
            bbox = tuple(drawn)          # tighter than /Rect
        else:
            bbox = (min(x0, x1) - x_off, y_top - max(y0, y1),
                    max(x0, x1) - x_off, y_top - min(y0, y1))
        chain = _ancestor_chain(elem)
        if not chain:
            continue
        out.append({"mcid": None, "tag": chain[0][0], "ancestors": chain,
                    "owner": chain[0][1], "bbox": bbox, "source": "objr"})
    return out


def extract_page_boxes(pdf, page_index, doc_fitz):
    """
    Returns (results, n_consumed, n_expected).
    results: [{mcid, tag, ancestors, owner, bbox}], bbox in UNROTATED
    page-space points. Callers MUST apply page.rotation_matrix.
    Discard the page unless n_consumed == n_expected.
    """
    page_pk = pdf.pages[page_index]
    page_fz = doc_fitz[page_index]

    root, flat = _parent_tree(pdf)
    if root is None:
        return [], 0, 0

    bboxlog = page_fz.get_bboxlog()
    n_expected = sum(1 for kind, _ in bboxlog if kind in PAINT_KINDS)

    ctx = _Ctx(bboxlog)
    sp = page_pk.obj.get("/StructParents")
    page_scope = ("page", int(sp)) if sp is not None else ("page", -1)
    try:
        resources = page_pk.resources        # handles page-tree inheritance
    except Exception:
        resources = page_pk.obj.get("/Resources")

    _run_stream(ctx, page_pk, resources, page_scope, flat, [], 0)
    _run_annot_appearances(ctx, page_pk, flat)
    if ctx.overrun:
        return [], ctx.consumed, n_expected

    scopes, results = {}, []
    for (scope, mcid), bbox in ctx.boxes.items():
        if scope[0] == "annot":
            continue                     # handled via /StructParent below
        if scope not in scopes:
            scopes[scope] = _mcid_map_for_key(flat, scope[1])
        elem = scopes[scope].get(mcid)
        if elem is None:
            continue
        chain = _ancestor_chain(elem)
        if not chain:
            continue
        results.append({"mcid": mcid, "tag": chain[0][0], "ancestors": chain,
                        "owner": chain[0][1], "bbox": tuple(bbox),
                        "source": "mcid"})

    # OBJR-associated content. Annotations do not participate in the content
    # stream, so these never affect the alignment counters.
    for ab in _annot_boxes(page_pk, page_fz, flat, ctx):
        cls, _owner = _resolve_class(ab["ancestors"])
        if cls is not None and cls in OBJR_CLASSES:
            results.append(ab)
    return results, ctx.consumed, n_expected


# %% CELL 4 - one PDF -> images + YOLO labels
import hashlib
import fitz


def _resolve_class(ancestors):
    """ancestors: leaf-to-root [(tag, objgen)]. Returns
    (class_name, owner_objgen) for the highest-PRIORITY tag anywhere in the
    chain, breaking ties toward the closest occurrence.

    Worked examples:
      Span > Lbl > LI > L                  -> Lbl   (Span is transparent)
      P > LBody > LI > L                   -> LBody (LBody outranks P)
      P > TD > TR > Table                  -> TD
      Link > Reference > TOCI > TOC        -> TOCI  (Link is transparent)
      Link > P                             -> P     (link merges into para)
      Form                                 -> Form  (OBJR-only, no MCID)
      P > LBody > LI > L > LBody > LI > L  -> inner LBody (tie -> closest)
    """
    best_rank, best = None, (None, None)
    for tag, og in ancestors:              # leaf -> root
        rank = CLASS_PRIORITY.get(tag)
        if rank is None:
            continue                       # transparent: keep walking up
        if best_rank is None or rank < best_rank:
            best_rank, best = rank, (tag, og)
    return best


def sha1_file(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def process_pdf(pdf_path, out_root):
    """Render pages to JPEG + write YOLO labels. Returns a stats dict."""
    key = sha1_file(pdf_path)
    split = "val" if int(key[:8], 16) % 100 < VAL_PCT else "train"
    img_dir = os.path.join(out_root, "images", split)
    lbl_dir = os.path.join(out_root, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    stem = key[:12]          # basenames collide under recursive glob

    st = {"path": str(pdf_path), "sha1": key, "split": split, "pages": 0,
          "emitted": 0, "quarantined": 0, "empty": 0, "boxes": 0,
          "class_counts": {}, "status": "ok", "error": None}

    with pikepdf.open(pdf_path) as pdf, fitz.open(pdf_path) as doc:
        st["pages"] = len(doc)
        for idx in range(len(doc)):
            page = doc[idx]
            boxes, n_consumed, n_expected = extract_page_boxes(pdf, idx, doc)

            # --- QA GATE: pointer walked off, trust nothing on this page ---
            if n_consumed != n_expected:
                st["quarantined"] += 1
                continue
            if not boxes:
                st["empty"] += 1
                continue

            # union at ELEMENT level (one LBody = one box, not one per line)
            merged = {}
            for b in boxes:
                cls, owner = _resolve_class(b["ancestors"])
                if cls is None:
                    continue
                k = (cls, owner)
                x0, y0, x1, y1 = b["bbox"]
                cur = merged.get(k)
                if cur is None:
                    merged[k] = [x0, y0, x1, y1]
                else:
                    cur[0] = min(cur[0], x0); cur[1] = min(cur[1], y0)
                    cur[2] = max(cur[2], x1); cur[3] = max(cur[3], y1)
            if not merged:
                st["empty"] += 1
                continue

            # bboxlog is UNROTATED; page.rect and the pixmap are rotated
            rot = page.rotation_matrix
            page_rect = page.rect
            W, H = page_rect.width, page_rect.height

            lines = []
            for (cls, _), bb in merged.items():
                r = (fitz.Rect(bb) * rot).normalize() & page_rect
                if r.is_empty or r.width < MIN_BOX_PT or r.height < MIN_BOX_PT:
                    continue
                lines.append(f"{CLASS_MAP[cls]} "
                             f"{(r.x0 + r.x1) / 2 / W:.6f} "
                             f"{(r.y0 + r.y1) / 2 / H:.6f} "
                             f"{r.width / W:.6f} {r.height / H:.6f}")
                st["class_counts"][cls] = st["class_counts"].get(cls, 0) + 1

            if not lines:
                # never emit an image with an empty label file -- YOLO would
                # train it as a genuine background page
                st["empty"] += 1
                continue

            # render LAST, only for pages that survived
            zoom = TARGET_LONG_EDGE / max(W, H)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            name = f"{stem}_p{idx:04d}"
            with open(os.path.join(img_dir, name + ".jpg"), "wb") as f:
                f.write(pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY))
            pix = None
            with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
                f.write("\n".join(lines))

            st["emitted"] += 1
            st["boxes"] += len(lines)
    return st


# %% CELL 5 - batch driver (for thousands of PDFs use the parallel script)
import glob
import json
from tqdm import tqdm


def build_dataset(pdf_dir, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.jsonl")

    done, seen_hashes = set(), set()
    if os.path.exists(manifest_path):                       # resume support
        with open(manifest_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue                                # torn final line
                done.add(r["path"])
                if r.get("sha1") and r.get("status") in ("ok", "duplicate"):
                    seen_hashes.add(r["sha1"])

    paths = [p for p in sorted(glob.glob(os.path.join(pdf_dir, "**", "*.pdf"),
                                         recursive=True)) if p not in done]
    print(f"{len(paths)} pdfs to process ({len(done)} already done)")

    with open(manifest_path, "a") as mf:
        for path in tqdm(paths):
            try:
                # hash first: a corpus of thousands almost always contains
                # duplicates, and reprocessing them inflates the class counts
                # and silently reweights the dataset
                key = sha1_file(path)
                if key in seen_hashes:
                    st = {"path": path, "sha1": key, "status": "duplicate"}
                else:
                    seen_hashes.add(key)
                    st = process_pdf(path, output_dir)
            except Exception as e:
                st = {"path": path, "status": "failed",
                      "error": f"{type(e).__name__}: {e}"}
            mf.write(json.dumps(st) + "\n")
            mf.flush()

    return summarize(output_dir)


def summarize(output_dir=OUTPUT_DIR):
    """Does the alignment model hold on the real corpus, and is any class
    starved? Read this before spending money on a training run."""
    recs = []
    with open(os.path.join(output_dir, "manifest.jsonl")) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    ok = [r for r in recs if r.get("status") == "ok"]
    dup = sum(1 for r in recs if r.get("status") == "duplicate")
    pages = sum(r["pages"] for r in ok)
    quar = sum(r["quarantined"] for r in ok)
    counts = {}
    for r in ok:
        for k, v in r.get("class_counts", {}).items():
            counts[k] = counts.get(k, 0) + v

    print(f"pdfs {len(recs)}: {len(ok)} ok, {dup} duplicate, "
          f"{len(recs) - len(ok) - dup} failed")
    print(f"pages {pages}: emitted {sum(r['emitted'] for r in ok)}, "
          f"empty {sum(r['empty'] for r in ok)}, "
          f"quarantined {quar} ({100 * quar / max(pages, 1):.2f}%)")
    total = sum(counts.values()) or 1
    for n in CLASS_NAMES:
        c = counts.get(n, 0)
        flag = "   <-- STARVED" if c < 200 else ""
        print(f"  {n:<8} {c:>8,}  {100 * c / total:5.2f}%{flag}")

    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n"
                "train: images/train\nval: images/val\n"
                f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")
    return counts


# %% CELL 6 - QA OVERLAY: eyeball before you train
def qa_overlay(pdf_path, page_index, out_png="qa_overlay.png"):
    """Draw extracted boxes onto the rendered page. Run this on a random
    sample of ~100 real pages before committing to a training run -- the
    count gate proves the pointer stayed in step, not that the semantics
    are right for your producer."""
    palette = [(1, 0, 0), (0, .6, 0), (0, 0, 1), (.9, .5, 0),
               (.6, 0, .6), (0, .6, .6), (.5, .5, 0)]
    with pikepdf.open(pdf_path) as pdf, fitz.open(pdf_path) as doc:
        page = doc[page_index]
        boxes, nc, ne = extract_page_boxes(pdf, page_index, doc)
        gated = (nc == ne)
        if not gated:
            # still render it -- a quarantined page is the most useful thing
            # you can look at, since it shows WHY alignment failed
            page.insert_text((8, 14), f"QUARANTINED consumed={nc} expected={ne}",
                             fontsize=9, color=(1, 0, 0))
        rot = page.rotation_matrix
        merged = {}
        for b in boxes:
            cls, owner = _resolve_class(b["ancestors"])
            if cls is None:
                continue
            r = (fitz.Rect(b["bbox"]) * rot).normalize()
            k = (cls, owner)
            merged[k] = r if k not in merged else (merged[k] | r)
        for (cls, _), r in merged.items():
            col = palette[CLASS_MAP[cls] % len(palette)]
            page.draw_rect(r, color=col, width=1.0)
            page.insert_text((r.x0 + 1, max(r.y0 - 2, 8)), cls,
                             fontsize=6, color=col)
        page.get_pixmap(dpi=150).save(out_png)
    return {"png": out_png, "path": pdf_path, "page": page_index,
            "consumed": nc, "expected": ne, "passed": gated,
            "n_boxes": len(merged)}


# %% CELL 6b - sample real pages for eyeballing
import random


def qa_sample(output_dir=OUTPUT_DIR, n=24, out_dir="qa_overlays",
              seed=0, only_quarantined=False, show=True):
    """Sample n random PAGES (not PDFs) across everything the manifest says
    was processed, render overlays, and display them inline in Colab.

    Run this before committing to a training run. The count gate proves the
    bboxlog pointer stayed in step; it does NOT prove your producer nests
    tags the way the ruleset assumes -- e.g. whether TH cells get an extra
    nested P wrapper the way list items do. Only your eyes catch that.

    only_quarantined=True focuses on the failures, which is where you learn
    what your corpus actually contains.
    """
    os.makedirs(out_dir, exist_ok=True)
    recs = []
    with open(os.path.join(output_dir, "manifest.jsonl")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "ok" and r.get("pages"):
                    recs.append(r)
    if not recs:
        print("nothing in the manifest with status=ok -- run build_dataset first")
        return []

    # expand to (path, page) so sampling is uniform over PAGES, not documents
    pool = [(r["path"], i) for r in recs for i in range(r["pages"])]
    if only_quarantined:
        pool = [pp for pp in pool
                if any(r["path"] == pp[0] and r["quarantined"] for r in recs)]
    random.Random(seed).shuffle(pool)

    results = []
    for path, idx in pool:
        if len(results) >= n:
            break
        if not os.path.exists(path):
            continue
        png = os.path.join(out_dir, f"{os.path.basename(path)[:40]}_p{idx}.png")
        try:
            info = qa_overlay(path, idx, png)
        except Exception as e:
            print(f"  overlay failed {path} p{idx}: {type(e).__name__}: {e}")
            continue
        if only_quarantined and info["passed"]:
            continue
        results.append(info)

    npass = sum(1 for r in results if r["passed"])
    print(f"rendered {len(results)} overlays to {out_dir}/ "
          f"({npass} passed gate, {len(results) - npass} quarantined)")
    if show:
        try:
            from IPython.display import Image, display
            for r in results:
                flag = "PASS" if r["passed"] else "QUARANTINE"
                print(f"{flag}  {r['path']} page {r['page']}  "
                      f"boxes={r['n_boxes']}")
                display(Image(filename=r["png"]))
        except ImportError:
            pass
    return results


# %% CELL 7 - run it
# --- upload a zip of PDFs ---
from google.colab import files
import zipfile
up = files.upload()
with zipfile.ZipFile(list(up.keys())[0]) as z:
    z.extractall("input_pdfs")

build_dataset("input_pdfs")
#
# # eyeball a random sample of real pages (NOT a placeholder filename):
qa_sample(n=24)
qa_sample(n=12, only_quarantined=True)    # focus on the failures
#
# or one specific page:
import glob
qa_overlay(sorted(glob.glob("input_pdfs/**/*.pdf", recursive=True))[0], 0)
#
# --- package results ---
import shutil
shutil.make_archive("yolo_dataset", "zip", OUTPUT_DIR)
files.download("yolo_dataset.zip")
