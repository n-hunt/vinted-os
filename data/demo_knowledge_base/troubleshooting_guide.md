# Vinted Label Printing System - Troubleshooting Guide

This guide provides solutions to common issues encountered during label processing and printing operations.

---

## Zebra GK420d Hardware Status

### Issue: Status Light Flashing Red

**Diagnosis:**

A flashing red status light on the Zebra GK420d typically indicates one of two hardware states:
- **Media Out**: The label roll has run out or is not properly loaded
- **Head Open**: The printhead assembly is not fully closed or locked

**Solution:**

Follow these steps to resolve the hardware issue:

1. Open the printer lid by releasing the side latches
2. Check if the label roll is present and has remaining labels
3. Re-align the label roll guides so they are snug against the label edges but not over-tightened (should allow smooth movement)
4. Ensure the labels are threaded correctly under the printhead sensor
5. Close the lid firmly until you hear an audible click indicating it's locked
6. Press the green feed button once to recalibrate the media sensor
7. Verify the status light returns to solid green

If the issue persists after these steps, check for debris under the printhead or contact hardware support.

---

## Vision Processing Output Quality

### Issue: Generated Labels Are Blank or Text Is Faint

**Diagnosis:**

When processed labels appear blank or contain barely visible text, this is typically caused by an incorrectly configured `binarization_threshold` in the image processing pipeline. 

The binarization process converts grayscale images to pure black and white for optimal thermal printing. If the threshold is set **too low** (e.g., values below 150), the OpenCV algorithm treats light gray text pixels as "background noise" and converts them to white, effectively making text disappear.

The default threshold is **200**, which works well for most shipping labels. However, if labels from certain carriers use lighter ink or lower contrast, text may be filtered out.

**Solution:**

Adjust the binarization threshold in the configuration file:

1. Open `config/settings.yaml` in a text editor
2. Locate the `binarization_threshold` parameter under the `image_processing` section
3. Increase the value to **180** or **200** (default)
4. Save the file and restart the pipeline

**Example Configuration:**

```yaml
image_processing:
  binarization_threshold: 200  # Increase this value if text is faint
  dpi: 203
  crop_margin: 10
```

**Note:** Setting the threshold too high (above 220) may cause the entire label to become black. Start with 180 and increase in increments of 10 until text is clearly visible.

---

## CUPS Printer Connection Timeout

### Issue: "Connection timed out" errors when attempting to print

**Diagnosis:**

Print jobs fail with error messages like `unable to connect to printer "Zebra_GK420d" at 192.168.1.50 - Connection timed out`. This indicates network connectivity issues between the system and the printer.

**Solution:**

1. Verify the printer is powered on and the status light is solid green
2. Check that the printer's IP address matches the configured address in CUPS
3. Test network connectivity: `ping 192.168.1.50`
4. Restart the CUPS service: `sudo systemctl restart cups` (Linux) or `sudo launchctl stop org.cups.cupsd && sudo launchctl start org.cups.cupsd` (macOS)
5. Verify the printer queue is not paused: `lpstat -p Zebra_GK420d`
6. If using USB connection, check cable integrity and try a different USB port

If the printer IP has changed, update it in CUPS printer settings or reconfigure the printer using `lpadmin`.
