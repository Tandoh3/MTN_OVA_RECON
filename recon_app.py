"""
MTN OVA vs T24 OVA reconciliation app.

Run:   streamlit run recon_app.py
Needs: pip install streamlit polars xlsxwriter fastexcel
"""
import io
import re
from datetime import datetime

import polars as pl
import streamlit as st
import xlsxwriter

# =====================================================================
# T24
# =====================================================================
T24_ENCODING = "cp1252"
T24_RENAME = {"FT": "FT_REFERENCE", "VALU_DATE": "VALUE_DATE",
              "REFERENCE": "DATE_AND_TIME"}   # REFERENCE holds YYMMDDhhmm
T24_COLUMNS = [   # order used when the file has no header row
    "TRANSACTION_TYPE", "FT_REFERENCE", "DEBIT_ACCOUNT", "CREDIT_ACCOUNT",
    "CURRENCY", "AMOUNT", "VALUE_DATE", "PROCESS_DATE", "PROCESS_DATE_2",
    "DATE_AND_TIME", "TRANSACTION_DETAILS", "PAYMENT_DETAILS", "INPUTTER",
    "BRANCH_CODE",
]
T24_FINAL = [
    "TRANSACTION_TYPE", "FT_REFERENCE", "DEBIT_ACCOUNT", "CREDIT_ACCOUNT",
    "CURRENCY", "AMOUNT", "VALUE_DATE", "PROCESS_DATE", "DATE_AND_TIME",
    "TRANSACTION_REF", "INPUTTER", "BRANCH_CODE",
]


def _clean_t24_name(c):
    c = re.sub(r"\s+", "_", c.strip().strip('"').strip().upper())
    return T24_RENAME.get(c, c)


def _looks_like_header(line):
    u = line.upper()
    return "TRANSACTION" in u and "DEBIT" in u and "CREDIT" in u


def _ref_number(col):
    """First 'ý'-separated part that is all digits, else null."""
    return pl.col(col).str.extract(r"(?:^|ý)\s*(\d+)\s*(?:ý|$)", 1)


def prepare_t24(raw: bytes, name: str):
    """Returns (t24_ova, messages)."""
    msgs = []
    text = raw.decode(T24_ENCODING).lstrip("﻿")
    first, rest = (text.split("\n", 1) + [""])[:2]
    first_clean = first.strip().strip('"')
    sep = "|" if "|" in first_clean else ","

    if _looks_like_header(first_clean):
        names, seen = [], {}
        for n in (_clean_t24_name(c) for c in first_clean.split(sep)):
            seen[n] = seen.get(n, 0) + 1
            names.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
        body, source = rest, "header row"
    else:
        names, body, source = T24_COLUMNS, text, "no header, standard column order"

    try:
        df = pl.read_csv(io.StringIO(body), separator=sep, has_header=False,
                         new_columns=names, infer_schema=False)
    except pl.exceptions.ComputeError as e:
        raise ValueError(f"{name}: could not read with separator {sep!r} and "
                         f"{len(names)} columns ({source}).") from e
    if df.width != len(names):
        raise ValueError(f"{name}: expected {len(names)} columns, found {df.width} ({source}).")
    msgs.append(f"{name}: separator {sep!r}, {source}, {df.height:,} rows")

    df = df.drop([c for c in ["PROCESS_DATE_2"] if c in df.columns]).with_columns(
        TRANSACTION_REF=pl.coalesce(_ref_number("TRANSACTION_DETAILS"),
                                    _ref_number("PAYMENT_DETAILS")).cast(pl.Int64, strict=False),
        DATE_AND_TIME=pl.col("DATE_AND_TIME").str.strip_chars()
                        .str.strptime(pl.Datetime, "%y%m%d%H%M", strict=False),
        _RAW_DT=pl.col("DATE_AND_TIME"),
    )
    bad = df.filter(pl.col("DATE_AND_TIME").is_null()
                    & pl.col("_RAW_DT").str.strip_chars().ne(""))
    if bad.height:
        msgs.append(f"⚠️ {bad.height} DATE_AND_TIME values not in YYMMDDhhmm, e.g. "
                    f"{bad['_RAW_DT'].head(5).to_list()}")
    return df.select(T24_FINAL), msgs


# =====================================================================
# MTN
# =====================================================================
MTN_COLUMNS = [
    "Id", "External id", "Date", "Status", "Type",
    "From", "From account", "From name", "From handler name",
    "To", "Fee", "To account", "To name", "To handler name", "To message",
    "Initiated by", "Currency", "Amount", "Balance",
]
MTN_OPTIONAL = {"Fee"}   # filled with 0 when a file has no Fee column


def _read_mtn(raw: bytes, name: str):
    if name.lower().endswith((".xlsx", ".xls")):
        df = pl.read_excel(io.BytesIO(raw), infer_schema_length=0)
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1252")
        header = text.split("\n", 1)[0]
        sep = max([",", "|", ";", "\t"], key=header.count)
        df = pl.read_csv(io.StringIO(text), separator=sep, infer_schema=False)
    return df.rename({c: c.strip().strip('"').strip() for c in df.columns})


def prepare_mtn_file(raw: bytes, name: str):
    df = _read_mtn(raw, name)
    missing = [c for c in MTN_COLUMNS if c not in df.columns and c not in MTN_OPTIONAL]
    if missing:
        raise ValueError(f"missing columns {missing}")
    df = df.with_columns([pl.lit(None, pl.String).alias(c)
                          for c in MTN_OPTIONAL if c not in df.columns])

    # Drop completely empty rows first, then clean
    df = df.select(MTN_COLUMNS).filter(
        ~pl.all_horizontal(pl.all().cast(pl.String).str.strip_chars().fill_null("") == "")
    )
    return df.with_columns(
        Id=pl.col("Id").cast(pl.String).str.strip_chars().str.replace(r"\.0+$", "")
             .cast(pl.Int64, strict=False),
        Currency=pl.when(pl.col("Currency").cast(pl.String).str.strip_chars()
                           .fill_null("") == "")
                   .then(pl.lit("GHS"))
                   .otherwise(pl.col("Currency").cast(pl.String).str.strip_chars()),
        Fee=pl.col("Fee").cast(pl.String).str.replace_all(r"[,\s]", "")
              .cast(pl.Float64, strict=False).fill_null(0.0),
        SOURCE_FILE=pl.lit(name),
    )


def prepare_mtn(files):
    """files: list of (name, bytes). Returns (mtn_ova, messages)."""
    frames, msgs = [], []
    for name, raw in files:
        try:
            df = prepare_mtn_file(raw, name)
            frames.append(df)
            msgs.append(f"✅ {name}: {df.height:,} rows")
        except Exception as e:
            msgs.append(f"❌ {name}: {e}")
    if not frames:
        raise ValueError("None of the MTN files could be read.")
    mtn = pl.concat(frames, how="vertical_relaxed")
    dupes = mtn.filter(pl.col("Id").is_not_null() & pl.col("Id").is_duplicated())
    if dupes.height:
        msgs.append(f"⚠️ {dupes['Id'].n_unique():,} Ids appear in more than one row "
                    "(overlapping exports?)")
    return mtn, msgs


# =====================================================================
# Reconciliation
# =====================================================================
STATUS_INFO = {
    "MISSING_IN_MTN":     "In T24 but no MTN transaction with that Id",
    "MISSING_IN_T24":     "In MTN but not posted in T24",
    "AMOUNT_MISMATCH":    "Same reference, amounts differ",
    "CURRENCY_MISMATCH":  "Same reference and amount, currencies differ",
    "MTN_NOT_SUCCESSFUL": "Posted in T24 but MTN status is not successful",
    "T24_NO_REFERENCE":   "T24 row with no number in TRANSACTION_REF",
    "MTN_NO_ID":          "MTN row with no usable Id",
    "MATCHED":            "Reference, amount, currency and status all agree",
}


def _amount(col, ignore_sign):
    a = (pl.col(col).cast(pl.String).str.replace_all(r"[,\s]", "")
         .cast(pl.Float64, strict=False))
    return a.abs() if ignore_sign else a


def reconcile(t24_ova, mtn_ova, success_statuses, ignore_sign=True, include_fee=False,
              t24_key="TRANSACTION_REF", mtn_key="Id"):
    t24 = (t24_ova.rename({c: f"T24_{c}" for c in t24_ova.columns})
           .with_columns(REF=pl.col(f"T24_{t24_key}").cast(pl.Int64, strict=False),
                         T24_AMOUNT=_amount("T24_AMOUNT", ignore_sign), _IN_T24=pl.lit(True)))
    mtn = (mtn_ova.rename({c: f"MTN_{c}" for c in mtn_ova.columns})
           .with_columns(REF=pl.col(f"MTN_{mtn_key}").cast(pl.Int64, strict=False),
                         MTN_Amount=_amount("MTN_Amount", ignore_sign),
                         MTN_Fee=_amount("MTN_Fee", ignore_sign).fill_null(0.0),
                         _IN_MTN=pl.lit(True))
           .with_columns(MTN_COMPARE_AMOUNT=(pl.col("MTN_Amount") + pl.col("MTN_Fee"))
                         if include_fee else pl.col("MTN_Amount")))

    occ = pl.int_range(pl.len()).over("REF").alias("_OCC")   # pair duplicates 1-to-1
    recon = (t24.filter(pl.col("REF").is_not_null()).with_columns(occ)
             .join(mtn.filter(pl.col("REF").is_not_null()).with_columns(occ),
                   on=["REF", "_OCC"], how="full", coalesce=True))
    recon = pl.concat([recon,
                       t24.filter(pl.col("REF").is_null()),
                       mtn.filter(pl.col("REF").is_null())], how="diagonal_relaxed")

    up = lambda c: pl.col(c).cast(pl.String).str.strip_chars().str.to_uppercase()
    ok_status = [s.strip().upper() for s in success_statuses]
    recon = recon.with_columns(
        RECON_STATUS=(
            pl.when(pl.col("REF").is_null() & pl.col("_IN_T24")).then(pl.lit("T24_NO_REFERENCE"))
            .when(pl.col("REF").is_null()).then(pl.lit("MTN_NO_ID"))
            .when(pl.col("_IN_MTN").is_null()).then(pl.lit("MISSING_IN_MTN"))
            .when(pl.col("_IN_T24").is_null()).then(pl.lit("MISSING_IN_T24"))
            .when((pl.col("T24_AMOUNT") != pl.col("MTN_COMPARE_AMOUNT")).fill_null(True))
                .then(pl.lit("AMOUNT_MISMATCH"))
            .when((up("T24_CURRENCY") != up("MTN_Currency")).fill_null(True))
                .then(pl.lit("CURRENCY_MISMATCH"))
            .when(~up("MTN_Status").is_in(ok_status).fill_null(False))
                .then(pl.lit("MTN_NOT_SUCCESSFUL"))
            .otherwise(pl.lit("MATCHED"))
        ),
        AMOUNT_DIFF=(pl.col("T24_AMOUNT") - pl.col("MTN_COMPARE_AMOUNT")).round(2),
        DUPLICATE_REF=pl.col("REF").is_not_null() & (pl.col("_OCC").max().over("REF") > 0),
    )

    lead = ["RECON_STATUS", "REF", "T24_AMOUNT", "MTN_Amount", "MTN_Fee",
            "MTN_COMPARE_AMOUNT", "AMOUNT_DIFF", "DUPLICATE_REF"]
    order = {s: i for i, s in enumerate(STATUS_INFO)}
    return (recon
            .with_columns(_ORD=pl.col("RECON_STATUS").replace_strict(order))
            .sort(["_ORD", "REF"], nulls_last=True)
            .select(lead
                    + [c for c in recon.columns if c.startswith("T24_") and c not in lead]
                    + [c for c in recon.columns if c.startswith("MTN_") and c not in lead]))


def summarise(recon):
    counts = recon.group_by("RECON_STATUS").agg(
        ROWS=pl.len(),
        T24_AMOUNT=pl.col("T24_AMOUNT").sum().round(2),
        MTN_AMOUNT=pl.col("MTN_Amount").sum().round(2),
        MTN_FEE=pl.col("MTN_Fee").sum().round(2),
    )
    return (pl.DataFrame({"RECON_STATUS": list(STATUS_INFO),
                          "DESCRIPTION": list(STATUS_INFO.values())})
            .join(counts, on="RECON_STATUS", how="left")
            .with_columns(pl.col("ROWS").fill_null(0),
                          pl.col("T24_AMOUNT", "MTN_AMOUNT", "MTN_FEE").fill_null(0.0)))


# =====================================================================
# Completeness check
# =====================================================================
MTN_DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M", "%Y-%m-%d", "%d/%m/%Y",
]


def _mtn_date(df, col):
    if df.schema[col].is_temporal():
        return pl.col(col).cast(pl.Datetime)
    s = pl.col(col).cast(pl.String).str.strip_chars().str.replace(r"\.\d+$", "")
    return pl.coalesce([s.str.strptime(pl.Datetime, f, strict=False) for f in MTN_DATE_FORMATS])


def _t24_date(dt_col, pd_col):
    return pl.coalesce(
        pl.col(dt_col).cast(pl.Datetime),
        pl.col(pd_col).cast(pl.String).str.strip_chars()
          .str.strptime(pl.Datetime, "%Y%m%d", strict=False),
    )


def completeness(t24_ova, mtn_ova, ignore_sign=True, include_fee=False):
    """Returns (coverage, daily, messages, (common_start, common_end))."""
    t24_dt = t24_ova.select(DT=_t24_date("DATE_AND_TIME", "PROCESS_DATE"),
                            AMT=_amount("AMOUNT", ignore_sign))
    mtn_amt = _amount("Amount", ignore_sign)
    if include_fee:
        mtn_amt = mtn_amt + _amount("Fee", ignore_sign).fill_null(0.0)
    mtn_dt = mtn_ova.select(DT=_mtn_date(mtn_ova, "Date"), AMT=mtn_amt)

    coverage = pl.concat([
        d.select(SOURCE=pl.lit(s), ROWS=pl.len(), NO_DATE=pl.col("DT").null_count(),
                 FIRST=pl.col("DT").min(), LAST=pl.col("DT").max())
        for s, d in (("T24", t24_dt), ("MTN", mtn_dt))
    ])

    msgs = []
    for row in coverage.iter_rows(named=True):
        if row["NO_DATE"]:
            msgs.append(f"⚠️ {row['NO_DATE']:,} {row['SOURCE']} rows have no readable date")
    if coverage["FIRST"].null_count():
        msgs.append("❌ One side has no readable dates, so coverage can't be compared.")
        return coverage, pl.DataFrame(), msgs, (None, None)

    t24_first, mtn_first = coverage["FIRST"]
    t24_last, mtn_last = coverage["LAST"]
    common = (max(t24_first, mtn_first), min(t24_last, mtn_last))
    if common[0] > common[1]:
        msgs.append("❌ The T24 and MTN data do not overlap in time at all")
    if t24_first.date() != mtn_first.date():
        msgs.append(f"⚠️ Start dates differ: {'T24' if t24_first > mtn_first else 'MTN'} starts later")
    if t24_last.date() != mtn_last.date():
        msgs.append(f"⚠️ End dates differ: {'T24' if t24_last < mtn_last else 'MTN'} ends earlier")

    def per_day(d, side):
        return (d.filter(pl.col("DT").is_not_null())
                 .group_by(DAY=pl.col("DT").dt.date())
                 .agg(pl.len().alias(f"{side}_ROWS"),
                      pl.col("AMT").sum().round(2).alias(f"{side}_AMOUNT")))

    daily = (
        pl.DataFrame({"DAY": pl.date_range(coverage["FIRST"].min().date(),
                                           coverage["LAST"].max().date(), "1d", eager=True)})
        .join(per_day(t24_dt, "T24"), on="DAY", how="left")
        .join(per_day(mtn_dt, "MTN"), on="DAY", how="left")
        .with_columns(pl.col("T24_ROWS", "MTN_ROWS").fill_null(0),
                      pl.col("T24_AMOUNT", "MTN_AMOUNT").fill_null(0.0))
        .with_columns(
            ROW_DIFF=pl.col("T24_ROWS").cast(pl.Int64) - pl.col("MTN_ROWS").cast(pl.Int64),
            AMOUNT_DIFF=(pl.col("T24_AMOUNT") - pl.col("MTN_AMOUNT")).round(2),
        )
        .with_columns(
            CHECK=(
                pl.when((pl.col("T24_ROWS") == 0) & (pl.col("MTN_ROWS") == 0)).then(pl.lit("NO_DATA_EITHER_SIDE"))
                .when(pl.col("MTN_ROWS") == 0).then(pl.lit("NO_MTN_DATA"))
                .when(pl.col("T24_ROWS") == 0).then(pl.lit("NO_T24_DATA"))
                .when((pl.col("ROW_DIFF") != 0) | (pl.col("AMOUNT_DIFF") != 0)).then(pl.lit("DIFFERENCE"))
                .otherwise(pl.lit("OK"))
            ),
        )
        .sort("DAY")
    )
    gaps = daily.filter(pl.col("CHECK").str.starts_with("NO_"))
    if gaps.height:
        msgs.append(f"⚠️ {gaps.height} day(s) missing data on one or both sides")
    else:
        msgs.append("✅ Every day in the range has data on both sides")
    return coverage, daily, msgs, common


def flag_outside_period(unrec, common):
    start, end = common
    if start is None:
        return unrec.with_columns(OUTSIDE_COMMON_PERIOD=pl.lit(None, pl.Boolean))
    return (unrec
            .with_columns(_DT=pl.coalesce(_t24_date("T24_DATE_AND_TIME", "T24_PROCESS_DATE"),
                                          _mtn_date(unrec, "MTN_Date")))
            .with_columns(OUTSIDE_COMMON_PERIOD=~pl.col("_DT").is_between(start, end))
            .drop("_DT"))


# =====================================================================
# Excel report (in memory, for the download button)
# =====================================================================
def excel_report(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with xlsxwriter.Workbook(buf, {"in_memory": True}) as wb:
        for name, df in sheets.items():
            if df.width:
                df.write_excel(wb, worksheet=name[:31], autofit=True,
                               dtype_formats={pl.Datetime: "yyyy-mm-dd hh:mm", pl.Date: "yyyy-mm-dd"},
                               table_style="Table Style Light 9")
    return buf.getvalue()


# =====================================================================
# App
# =====================================================================
@st.cache_data(show_spinner="Reading T24 file...")
def load_t24(raw, name):
    return prepare_t24(raw, name)


@st.cache_data(show_spinner="Reading MTN files...")
def load_mtn(files):
    return prepare_mtn(list(files))


def main():
    st.set_page_config(page_title="MTN OVA ↔ T24 Recon", page_icon="🔁", layout="wide")
    st.title("MTN OVA ↔ T24 OVA Reconciliation")

    with st.sidebar:
        st.header("1. Upload data")
        t24_file = st.file_uploader("T24 OVA file", type=["csv", "txt"])
        mtn_files = st.file_uploader("MTN OVA files (select all)", type=["csv", "txt", "xlsx", "xls"],
                                     accept_multiple_files=True)
        st.header("2. Settings")
        statuses = st.text_input("MTN statuses that count as successful", "SUCCESSFUL",
                                 help="Comma-separated")
        ignore_sign = st.checkbox("Ignore amount sign (MTN debits are negative)", value=True)
        include_fee = st.checkbox("Add MTN Fee to Amount before comparing", value=False,
                                  help="Tick if T24 posts the amount including the MTN fee")

    if not t24_file or not mtn_files:
        st.info("Upload the T24 file and the MTN files in the sidebar to start.")
        return

    try:
        t24_ova, t24_msgs = load_t24(t24_file.getvalue(), t24_file.name)
        mtn_ova, mtn_msgs = load_mtn(tuple((f.name, f.getvalue()) for f in mtn_files))
    except Exception as e:
        st.error(f"Could not load the data: {e}")
        return

    success = [s for s in statuses.split(",") if s.strip()]
    recon = reconcile(t24_ova, mtn_ova, success, ignore_sign, include_fee)
    summary = summarise(recon)
    coverage, daily, cov_msgs, common = completeness(t24_ova, mtn_ova, ignore_sign, include_fee)
    matched = recon.filter(pl.col("RECON_STATUS") == "MATCHED")
    unrec = flag_outside_period(recon.filter(pl.col("RECON_STATUS") != "MATCHED"), common)

    # ---- headline numbers
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("T24 rows", f"{t24_ova.height:,}")
    c2.metric("MTN rows", f"{mtn_ova.height:,}")
    c3.metric("Matched", f"{matched.height:,}")
    c4.metric("Unreconciled", f"{unrec.height:,}")
    c5.metric("Match rate", f"{matched.height / max(recon.height, 1):.1%}")
    c6.metric("MTN fees", f"{mtn_ova['Fee'].abs().sum():,.2f}")

    with st.expander("Load log"):
        for m in t24_msgs + mtn_msgs:
            st.write(m)

    tab_sum, tab_unrec, tab_match, tab_comp, tab_raw = st.tabs(
        ["Summary", f"Unreconciled ({unrec.height:,})", f"Matched ({matched.height:,})",
         "Data completeness", "Source data"])

    with tab_sum:
        st.dataframe(summary, hide_index=True, width="stretch")

    with tab_unrec:
        f1, f2, f3 = st.columns([2, 1, 1])
        pick = f1.multiselect("Status", [s for s in STATUS_INFO if s != "MATCHED"])
        ref = f2.text_input("Search REF")
        outside = f3.selectbox("Common period", ["All", "Inside only", "Outside only"])
        view = unrec
        if pick:
            view = view.filter(pl.col("RECON_STATUS").is_in(pick))
        if ref.strip():
            view = view.filter(pl.col("REF").cast(pl.String).str.contains(ref.strip(), literal=True))
        if outside == "Inside only":
            view = view.filter(~pl.col("OUTSIDE_COMMON_PERIOD").fill_null(False))
        elif outside == "Outside only":
            view = view.filter(pl.col("OUTSIDE_COMMON_PERIOD").fill_null(False))
        st.caption(f"{view.height:,} rows")
        st.dataframe(view, hide_index=True, width="stretch")

    with tab_match:
        st.dataframe(matched, hide_index=True, width="stretch")

    with tab_comp:
        for m in cov_msgs:
            st.write(m)
        if common[0] is not None:
            st.write(f"**Common period:** {common[0]} → {common[1]}")
        st.dataframe(coverage, hide_index=True, width="stretch")
        if daily.height:
            only_issues = st.checkbox("Show only days with issues", value=True)
            d = daily.filter(pl.col("CHECK") != "OK") if only_issues else daily
            st.dataframe(d, hide_index=True, width="stretch")

    with tab_raw:
        st.subheader("T24")
        st.dataframe(t24_ova, hide_index=True, width="stretch")
        st.subheader("MTN")
        st.dataframe(mtn_ova, hide_index=True, width="stretch")

    # ---- downloads
    st.divider()
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Full recon report (Excel)",
        excel_report({
            "Summary": summary, "Unreconciled": unrec, "Matched": matched,
            "Coverage": coverage, "Daily": daily,
        }),
        file_name=f"MTN_T24_OVA_RECON_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    d2.download_button(
        "⬇️ Unreconciled only (CSV)",
        unrec.write_csv(),
        file_name=f"MTN_T24_OVA_UNRECONCILED_{stamp}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()