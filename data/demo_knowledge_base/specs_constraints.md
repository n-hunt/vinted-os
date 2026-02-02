# System Specifications & Constraints

**Vinted Label Processing Pipeline - Technical Reference**

This document defines the critical technical constraints and specifications that govern the operation of the label processing system. Violating these constraints will result in print failures, data corruption, or rejected jobs.

---

## Physical Output Constraints

### Label Dimensions

**Constraint:** All generated PDF labels must be exactly **4.0 inches × 6.0 inches** (101.6mm × 152.4mm).

**Reasoning:**

The Zebra GK420d thermal printer uses pre-cut label stock with fixed dimensions of 4×6 inches. The label roll has perforations at these exact intervals. If the PDF page size exceeds these dimensions, the printer will:
- Print content across the perforation gap (splitting text/barcodes)
- Misalign subsequent labels in the queue
- Trigger a "media mismatch" error in CUPS

The `generator.py` module enforces this constraint by:
```python
page_width = 4.0  # inches
page_height = 6.0  # inches
```

Any deviation from this specification will cause physical printing defects.

---

### Resolution (DPI)

**Constraint:** All PDF-to-image conversions must use **203 DPI** (Dots Per Inch).

**Reasoning:**

The Zebra GK420d printhead has a native resolution of **203 DPI**. This is a fixed hardware characteristic determined by the density of the thermal elements.

**Why this matters:**
- **203 DPI**: Images render at 1:1 pixel mapping, resulting in fast processing and sharp output
- **300 DPI**: The printer's firmware must downsample the image, causing:
  - Slow rasterization (5-10 second delay per label)
  - Scaling artifacts (blurry barcodes, text aliasing)
  - Increased memory usage on the printer
- **150 DPI or lower**: Text becomes pixelated and barcodes may fail to scan

The `vision.py` module sets DPI during PDF conversion:
```python
dpi = 203  # Native printhead resolution
```

**Critical:** Never modify this value unless the printer hardware is replaced.

---

## Ingestion Rules & Filtering

### Transaction ID Pattern (Filename Extraction)

**Constraint:** Transaction IDs must match the pattern `(\d{10,})` (10 or more consecutive digits).

**Reasoning:**

Vinted generates unique order identifiers with at least 10 digits (e.g., `18273645901`, `12345678901234`). The `gmail.py` module uses this regex pattern to extract transaction IDs from PDF filenames.

**Accepted filenames:**
- SUCCESS: `Vinted-Label-18273645901.pdf` → ID: `18273645901`
- SUCCESS: `Order-return-form-12345678901.pdf` → ID: `12345678901`
- SUCCESS: `Vinted-Digital-Label-98765432101234.pdf` → ID: `98765432101234`

**Rejected filenames:**
- ERROR: `Invoice_scan.pdf` → No numeric ID
- ERROR: `Label-12345.pdf` → Only 5 digits (too short)
- ERROR: `Document-ABC123.pdf` → Contains letters

This strict filtering prevents the pipeline from processing:
- Customer service correspondence
- Scanned invoices
- Spam or phishing attachments
- Non-Vinted documents

**Implementation:**
```python
transaction_id_pattern = re.compile(r'(\d{10,})')
```

If an attachment lacks a valid transaction ID, it is logged as ignored and the message remains unprocessed in the inbox.

---

### Attachment Filename Patterns

**Constraint:** Only attachments matching specific patterns are processed:

**Return Forms:**
- Pattern: `^Order-return-form`
- Example: `Order-return-form-18273645901.pdf`

**Shipping Labels:**
- Pattern: `^Vinted-(?:Label|Digital-Label)`
- Examples: 
  - `Vinted-Label-18273645901.pdf`
  - `Vinted-Digital-Label-18273645901.pdf`

**Reasoning:**

These patterns are hardcoded by Vinted's email automation system. Any deviation indicates:
- Manual forwarding (customer renamed the file)
- Phishing attempt (fake Vinted email)
- Corrupted email structure

Files not matching these patterns are silently skipped to maintain data integrity.

---

## Image Processing Pipeline Constraints

### Binarization Threshold

**Constraint:** The grayscale-to-binary conversion threshold must be between **180-220**.

**Default:** `200`

**Reasoning:**

Thermal printers only understand two colors: black (heat applied) and white (no heat). The binarization process converts grayscale PDF scans to pure black/white using a pixel brightness threshold:

- **Pixel value > threshold** → White (background)
- **Pixel value ≤ threshold** → Black (printed)

**Effects of incorrect values:**
- **Threshold < 150**: Light gray text becomes white (invisible)
- **Threshold > 220**: Background noise becomes black (label is solid black)
- **Optimal range**: 180-200 (balances text clarity with background removal)

This value is configured in `config/settings.yaml`:
```yaml
image_processing:
  binarization_threshold: 200
```

---

## Data Persistence Requirements

### Transaction ID Uniqueness

**Constraint:** Each `vinted_order_id` must be unique across the database.

**Reasoning:**

The pipeline uses the transaction ID as the natural key for matching:
- Return forms (contain item list)
- Shipping labels (contain barcode/address)

If duplicate IDs exist:
- The matching algorithm may pair the wrong documents
- Database foreign key constraints may fail
- Print jobs may duplicate

**Enforcement:**
```python
vinted_order_id: str = Field(index=True, unique=True)
```

While not enforced as a database constraint (for fault tolerance), the application logic treats duplicates as errors.

---

## Network & Hardware Assumptions

### Printer Network Address

**Assumption:** The Zebra GK420d is reachable at `192.168.1.50` (or via USB at `/dev/usb/lp0`).

**Reasoning:**

CUPS requires a stable network address or device path. If the printer's IP changes (DHCP reassignment), print jobs will fail with "Connection timed out" errors.

**Recommendation:** Configure a static IP lease in the router's DHCP settings for the printer's MAC address.

---

### Gmail API Credentials

**Constraint:** Valid OAuth 2.0 credentials must exist at `credentials.json` and `token.json`.

**Reasoning:**

Without valid credentials:
- The pipeline cannot fetch emails
- All runs will immediately fail with authentication errors
- No transactions will be processed

Token expiration is handled automatically via refresh tokens, but initial setup requires manual OAuth flow.

---

## Summary of Critical Values

| Parameter | Value | Location | Modifiable? |
|-----------|-------|----------|-------------|
| Page Width | 4.0 inches | `settings.yaml` | ERROR: No (hardware constraint) |
| Page Height | 6.0 inches | `settings.yaml` | ERROR: No (hardware constraint) |
| DPI | 203 | `settings.yaml` | ERROR: No (hardware constraint) |
| Binarization Threshold | 200 | `settings.yaml` | SUCCESS: Yes (180-220 range) |
| Transaction ID Pattern | `(\d{10,})` | `settings.yaml` | WARNING: Caution (must match Vinted format) |
| Printer IP | 192.168.1.50 | CUPS config | SUCCESS: Yes (update in CUPS) |

**Legend:**
- ERROR: **No**: Changing this will cause hardware failures
- SUCCESS: **Yes**: Safe to modify within constraints
- WARNING: **Caution**: Modification requires understanding of data format


<!-- Test modification for incremental update -->
