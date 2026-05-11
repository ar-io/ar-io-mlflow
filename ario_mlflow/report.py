"""Generate self-contained HTML verification reports for MLflow artifact viewer."""

import html
import os

from .arweave import WALLET_MODE_EPHEMERAL, WALLET_MODE_PERSISTENT


def generate_verification_html(
    proof: dict,
    anchor_result: dict | None,
    artifact_hash: str | None = None,
    artifact_verified: bool | None = None,
    verification: dict | None = None,
    cli_verify_cmd: str | None = None,
    verify_base_url: str | None = None,
    wallet_mode: str | None = None,
) -> str:
    """Generate an HTML report for the MLflow artifact viewer.

    Args:
        proof: Full proof envelope.
        anchor_result: Dict with tx_id, url, receipt (None if anchoring disabled).
        artifact_hash: SHA-256 hash of model artifacts.
        artifact_verified: Whether artifact integrity was checked and passed.
        verification: ar.io Verify result (from CLI verify command). Contains
            attestation_level, report_url, attested_by, attested_at.
        cli_verify_cmd: Full CLI command to print as the "verify this proof"
            hint (e.g. ``"ar-io-mlflow verify run <run_id>"`` or
            ``"ar-io-mlflow verify model fraud-detector/3"``). If omitted, falls
            back to the training-run form using ``run_id`` from the proof.
        verify_base_url: Base URL for the ar.io Verify dashboard (tx_id is
            appended). Falls back to ``ARIO_MLFLOW_ARIO_VERIFY_URL``. If
            neither is set, the CLI command is shown without an external
            verify-link (the CLI is always actionable).
        wallet_mode: One of ``"user-configured"``, ``"persistent"``, or
            ``"ephemeral"`` — rendered as a small transparency note so
            readers can tell whether the proof was signed with a
            production-quality wallet or a demo-default one.
    """
    record = proof.get("record", {})
    event_type = record.get("event_type", "unknown")
    run_id = record.get("run_id", record.get("source_run_id", ""))
    timestamp = record.get("timestamp", "")

    tx_id = anchor_result["tx_id"] if anchor_result else proof.get("arweave_tx_id")
    arweave_url = (anchor_result.get("url", "") if anchor_result
                   else proof.get("arweave_url", ""))

    # Determine status
    if verification and verification.get("attestation_level"):
        status = "verified"
        status_color = "#22c55e"
        status_label = f"Verified (Level {verification['attestation_level']})"
    elif tx_id:
        status = "anchored"
        status_color = "#22c55e"
        status_label = "Anchored"
    else:
        status = "signed"
        status_color = "#eab308"
        status_label = "Signed (local)"

    record_hash = proof.get("record_hash", "")
    previous_hash = proof.get("previous_hash", "")
    signature = proof.get("signature", "")
    public_key = proof.get("public_key", "")

    # Wallet-mode notice — only shown when the plugin ran on an auto-generated
    # wallet, so readers can tell at a glance that this is a demo-default
    # signer (not a caller-configured production wallet).
    wallet_notice = ""
    if wallet_mode == WALLET_MODE_PERSISTENT:
        wallet_notice = (
            '<div style="background:#fef9c3;border:1px solid #fde047;border-radius:6px;'
            'padding:10px 14px;margin-bottom:16px;font-size:13px;color:#713f12;">'
            "Signed with the plugin's auto-generated wallet "
            "(<code>~/.ario-mlflow/wallet.json</code>). Set "
            "<code>ARIO_MLFLOW_ARWEAVE_WALLET</code> to sign with your own wallet.</div>"
        )
    elif wallet_mode == WALLET_MODE_EPHEMERAL:
        wallet_notice = (
            '<div style="background:#fee2e2;border:1px solid #fecaca;border-radius:6px;'
            'padding:10px 14px;margin-bottom:16px;font-size:13px;color:#7f1d1d;">'
            "Signed with an <strong>in-memory, ephemeral</strong> wallet. "
            "The signing address will rotate on restart. Configure "
            "<code>ARIO_MLFLOW_ARWEAVE_WALLET</code> for stable provenance.</div>"
        )

    # Artifact integrity rows
    integrity_row = ""
    if artifact_hash:
        if artifact_verified is True:
            integrity_row = _row("Artifact Integrity", _badge("Verified", "#22c55e"))
        elif artifact_verified is False:
            integrity_row = _row("Artifact Integrity", _badge("MISMATCH", "#ef4444"))
        # If artifact_verified is None, don't show an integrity status row —
        # this is the training anchor where we're recording the hash, not checking it
        integrity_row += _row("Artifact Hash", _mono(artifact_hash))

    # Arweave anchoring rows
    arweave_row = ""
    if tx_id:
        link = f'<a href="{html.escape(arweave_url)}" target="_blank" rel="noopener">{html.escape(tx_id)}</a>'
        arweave_row = _row("Arweave TX", link)
        arweave_row += _row("Gateway URL", f'<a href="{html.escape(arweave_url)}" target="_blank" rel="noopener">{html.escape(arweave_url)}</a>')

    # Turbo receipt rows
    receipt_row = ""
    turbo_receipt = proof.get("turbo_receipt") or (anchor_result.get("receipt") if anchor_result else None)
    if turbo_receipt:
        if turbo_receipt.get("timestamp"):
            receipt_row += _row("Turbo Timestamp", _mono(str(turbo_receipt["timestamp"])) + " ms")
        if turbo_receipt.get("owner"):
            receipt_row += _row("Turbo Owner", _mono(str(turbo_receipt["owner"])))
        if turbo_receipt.get("signature"):
            receipt_row += _row("Turbo Signature", _mono(str(turbo_receipt["signature"])))

    # ar.io Verify rows
    verify_row = ""
    if verification:
        if verification.get("attestation_level"):
            verify_row += _row("Attestation Level", _badge(f"Level {verification['attestation_level']}", "#22c55e"))
        if verification.get("attested_by"):
            verify_row += _row("Attested By", html.escape(str(verification["attested_by"])))
        if verification.get("attested_at"):
            verify_row += _row("Attested At", html.escape(str(verification["attested_at"])[:19]) + "Z")
        if verification.get("report_url"):
            url = html.escape(str(verification["report_url"]))
            verify_row += _row("Report", f'<a href="{url}" target="_blank" rel="noopener">View on ar.io Verify</a>')

    # ar.io Verify link (CLI command always shown; external link only if
    # a verify URL is configured — no silent fallback to a developer
    # endpoint).
    verify_link = ""
    if tx_id and not verification:
        base_raw = verify_base_url or os.environ.get("ARIO_MLFLOW_ARIO_VERIFY_URL")
        cmd = cli_verify_cmd or (f"ar-io-mlflow verify run {run_id}" if run_id else "ar-io-mlflow verify run <run_id>")
        if base_raw:
            verify_url = f"{base_raw.rstrip('/')}/{html.escape(tx_id)}"
            link_html = (
                f', or <a href="{verify_url}" target="_blank" rel="noopener">'
                f'check manually on ar.io Verify</a>'
            )
        else:
            link_html = ""
        verify_link = f"""
  <div style="background:#fff;border:1px solid #e5e7eb;border-radius:6px;padding:14px;margin-bottom:16px;">
    <div style="font-size:13px;font-weight:600;margin-bottom:6px;">ar.io Verification</div>
    <div style="font-size:13px;color:#6b7280;">
      Run <code style="font-family:'SF Mono',monospace;font-size:12px;">{html.escape(cmd)}</code> to verify this proof
      and update this report{link_html}.
    </div>
  </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ar.io Verification Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #fafafa; color: #1a1a1a; padding: 24px; }}
  .container {{ max-width: 720px; margin: 0 auto; }}
  h1 {{ font-size: 18px; font-weight: 600; margin-bottom: 4px; }}
  .subtitle {{ color: #6b7280; font-size: 13px; margin-bottom: 20px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; margin-bottom: 16px; }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #f3f4f6; font-size: 13px; vertical-align: top; }}
  td:first-child {{ width: 160px; color: #6b7280; font-weight: 500; }}
  .mono {{ font-family: "SF Mono", "Fira Code", monospace; font-size: 12px; word-break: break-all; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; color: #fff; }}
  a {{ color: #2563eb; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .section-label {{ font-size: 11px; font-weight: 600; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; padding: 8px 14px 4px; }}
  .verify-section {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 16px; margin-top: 8px; }}
  .verify-section h2 {{ font-size: 14px; font-weight: 600; margin-bottom: 8px; }}
  .verify-section pre {{ font-size: 12px; background: #f9fafb; padding: 12px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; word-break: break-all; }}
  .verify-section code {{ font-family: "SF Mono", "Fira Code", monospace; }}
</style>
</head>
<body>
<div class="container">
  <h1>ar.io Verification Report</h1>
  <div class="subtitle">{html.escape(event_type)} &mdash; {html.escape(timestamp)}</div>

  {wallet_notice}

  <table>
    {_row("Status", _badge(status_label, status_color))}
    {_row("Event Type", html.escape(event_type))}
    {_row("Run ID", _mono(run_id))}
    {_row("Timestamp", html.escape(timestamp))}
    {integrity_row}
  </table>

  {'<table>' + arweave_row + receipt_row + '</table>' if arweave_row or receipt_row else ''}

  {'<table>' + verify_row + '</table>' if verify_row else ''}

  {verify_link}

  <table>
    {_row("Record Hash", _mono(record_hash))}
    {_row("Previous Hash", _mono(previous_hash))}
    {_row("Public Key", _mono(public_key))}
    {_row("Signature", _mono(signature))}
  </table>

  <div class="verify-section">
    <h2>Independent Verification</h2>
    <p style="font-size:13px;color:#6b7280;margin-bottom:10px;">
      To verify this proof independently:
    </p>
    <pre><code># 1. Fetch the proof from Arweave (gateway serves raw data at /raw/&lt;tx_id&gt;)
curl https://{html.escape(_gateway_host(arweave_url))}/raw/{html.escape(tx_id or 'TX_ID')}

# 2. Verify: re-hash the record field with SHA-256
#    and compare to record_hash

# 3. Verify the Ed25519 signature over:
#    canonical_json({{"record_hash", "previous_hash", "timestamp"}})
#    using the public key above

# 4. Check previous_hash matches the prior record's
#    record_hash for chain integrity</code></pre>
  </div>
</div>
</body>
</html>"""


def _gateway_host(arweave_url: str | None) -> str:
    """Extract the gateway hostname from an Arweave URL, or return a default."""
    if not arweave_url:
        return "turbo-gateway.com"
    from urllib.parse import urlparse

    host = urlparse(arweave_url).hostname
    return host or "turbo-gateway.com"


def _row(label: str, value: str) -> str:
    return f"<tr><td>{html.escape(label)}</td><td>{value}</td></tr>\n"


def _mono(text: str) -> str:
    return f'<span class="mono">{html.escape(text)}</span>'


def _badge(text: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{html.escape(text)}</span>'
