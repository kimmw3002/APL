import argparse
import os

from NSFopen.read import read

# Nanosurf "Data Info" group headers append " group" for Scan/Feedback only.
GROUP_SUFFIX = {"-- Scan --": "-- Scan group --",
                "-- Feedback --": "-- Feedback group --"}


def fix_encoding(s):
    """NSFopen decodes the header as latin-1, so UTF-8 bytes (µ, °) come back
    mangled (Âµ, Â°). Re-encode to recover the original characters."""
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def info_text(nid_path):
    afm = read(nid_path)
    info = afm.param[("HeaderDump", "DataSet-Info")]
    lines = []
    for key, val in info.items():
        key, val = fix_encoding(key), fix_encoding(str(val))
        if val == "--------":                       # group divider
            lines.append(GROUP_SUFFIX.get(key, key))
        else:
            lines.append(f"{key}\t{val}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Extract Nanosurf .nid metadata to a Data Info-style text file.")
    ap.add_argument("nid", help="Path to the input .nid file.")
    args = ap.parse_args()

    base = os.path.splitext(os.path.basename(args.nid))[0]
    out = os.path.join(os.path.dirname(os.path.abspath(args.nid)), base + " Info.txt")

    with open(out, "w", encoding="utf-8") as f:
        f.write(info_text(args.nid))
    print("saved:", out)


if __name__ == "__main__":
    main()
