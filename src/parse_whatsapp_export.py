"""
Parses a WhatsApp chat export (.txt, exported "without media") and pulls out
candidate messages that look customer-support-flavored, based on keyword
matching against the 36 Dataset A intents.

This runs entirely locally -- it reads the raw export, but only WRITES OUT
the filtered candidate subset (plus a rough guessed intent per candidate).
Review/correct the guessed intents yourself before using this as Dataset B;
this script does NOT produce ground truth, only a shortlist to speed up
manual labeling.

Usage:
    python parse_whatsapp_export.py --input data/whatsapp_export_raw.txt

Output:
    data/whatsapp_candidates.csv  -- columns: raw_text, guessed_intent,
    matched_keywords. Open this, delete rows that aren't actually usable,
    fix/confirm the intent for the ones you keep, then merge into Dataset B.
"""

from pathlib import Path
import argparse
import csv
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# WhatsApp export line format (varies slightly by platform/locale), e.g.:
#   12/09/24, 6:41 PM - Tanmay: order cancel krna h bhai
#   [12/09/24, 6:41:02 PM] Tanmay: order cancel krna h bhai
LINE_PATTERN = re.compile(
    r"""^
    (?:\[)?
    \d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:[APap][Mm])?
    (?:\])?
    \s*[-\u2013]?\s*
    (?P<sender>[^:]+):\s*
    (?P<message>.*)
    $""",
    re.VERBOSE,
)

# Keyword -> intent, kept intentionally loose (substring match) since real
# messages won't be as clean as synthetic ones. A message can match more
# than one intent's keywords; all matches are reported so you can pick the
# right one during manual review.
INTENT_KEYWORDS = {
    "order_cancel": ["order cancel", "cancel order", "order cncl"],
    "order_status": ["order status", "order kaha", "mera order"],
    "order_modify": ["order modify", "order change", "order edit"],
    "order_cancellation_failed": ["cancel nahi hua", "cancel failed"],
    "wrong_item": ["wrong item", "wrong product", "galat item", "galat product"],
    "delivery_address_change": ["address change", "shipping address"],
    "delivery_delay": ["delivery late", "delivery delay", "late ho raha"],
    "delivery_tracking": ["track", "courier kaha", "parcel status", "shipment"],
    "delivery_damaged": ["damaged", "tuta hua", "broken", "phata hua"],
    "delivery_missing": ["delivered but", "delivered nahi mila", "parcel missing"],
    "return_item": ["return krna", "return item", "wapas krna"],
    "exchange_item": ["exchange krna", "exchange item", "size change"],
    "refund_status": ["refund status", "refund kaha", "paisa wapas"],
    "refund_after_cancellation": ["refund after cancel", "cancel ke baad refund"],
    "refund_missing": ["refund nahi mila", "refund missing"],
    "payment_failed": ["payment fail", "transaction fail", "payment nahi hua"],
    "payment_pending": ["payment pending", "transaction pending"],
    "wrong_charge": ["wrong charge", "galat amount", "extra charge"],
    "duplicate_charge": ["double charge", "twice", "do baar paisa"],
    "account_login": ["login nahi", "login issue", "account access"],
    "password_reset": ["password reset", "password bhul", "forgot password"],
    "profile_change": ["profile update", "profile change", "naam change"],
    "human_agent": ["human agent", "real agent", "insaan se baat", "human se"],
    "subscription_cancel": ["subscription cancel", "membership cancel"],
    "subscription_upgrade": ["subscription upgrade", "plan upgrade"],
    "subscription_charged": ["subscription charge", "renewal charge"],
    "bill_due_date": ["bill due", "due date", "payment date"],
    "bill_payment": ["bill pay", "pay bill", "bill clear"],
    "flight_booking": ["flight book", "flight ticket"],
    "flight_status": ["flight status", "flight delay", "flight time"],
    "hotel_booking": ["hotel book", "hotel stay"],
    "cab_booking": ["cab book", "gaadi chahiye", "cab lagani"],
    "greeting": ["hii", "hello", "namaste", "hey"],
    "thanks": ["thank you", "thanks", "shukriya", "dhanyavad"],
    "complaint": ["disappointed", "kharab service", "bad experience", "worst service"],
    "faq_how_to": ["kaise kare", "kaise kre", "how to", "kaise use"],
}


def parse_whatsapp_export(path):
    messages = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        buffer = None
        for line in f:
            line = line.rstrip("\n")
            match = LINE_PATTERN.match(line)
            if match:
                if buffer is not None:
                    messages.append(buffer)
                buffer = match.group("message").strip()
            else:
                # Continuation of a multi-line message.
                if buffer is not None:
                    buffer += " " + line.strip()
        if buffer is not None:
            messages.append(buffer)
    return messages


def find_candidates(messages):
    candidates = []
    seen = set()

    for msg in messages:
        if not msg or msg in seen:
            continue

        lower = msg.lower()

        # Skip WhatsApp system/media placeholder lines.
        if lower in ("<media omitted>", "this message was deleted", "null"):
            continue

        matched = []
        for intent, keywords in INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in lower:
                    matched.append((intent, kw))
                    break  # one match per intent is enough to flag it

        if matched:
            seen.add(msg)
            guessed_intent = matched[0][0]
            matched_keywords = "; ".join(f"{i}:{k}" for i, k in matched)
            candidates.append({
                "raw_text": msg,
                "guessed_intent": guessed_intent,
                "matched_keywords": matched_keywords,
            })

    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help="Path to WhatsApp export .txt")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: data/whatsapp_candidates.csv)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (PROJECT_ROOT / input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Export file not found: {input_path}")

    output_path = Path(args.output) if args.output else PROJECT_ROOT / "data" / "whatsapp_candidates.csv"

    print(f"Parsing: {input_path}")
    messages = parse_whatsapp_export(input_path)
    print(f"Total messages parsed: {len(messages)}")

    candidates = find_candidates(messages)
    print(f"Candidate customer-support-flavored messages found: {len(candidates)}")

    if not candidates:
        print(
            "\nNo candidates matched the keyword list. This chat may not "
            "contain support-flavored language, or the keyword list needs "
            "expanding -- feel free to edit INTENT_KEYWORDS and re-run."
        )
        return

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["raw_text", "guessed_intent", "matched_keywords"])
        writer.writeheader()
        writer.writerows(candidates)

    print(f"\nSaved candidates to: {output_path}")
    print(
        "\nNext step: open this CSV, delete rows that aren't usable, and "
        "fix/confirm 'guessed_intent' for the ones you keep (the keyword "
        "match is a rough first pass, not ground truth). Nothing beyond "
        "this filtered file needs to leave your machine."
    )

    # Quick coverage summary so you know which intents got zero hits.
    covered_intents = {c["guessed_intent"] for c in candidates}
    missing = sorted(set(INTENT_KEYWORDS.keys()) - covered_intents)
    print(f"\nIntents with at least one candidate: {len(covered_intents)} / {len(INTENT_KEYWORDS)}")
    if missing:
        print("Intents with zero candidates in this chat (will need hand-authored fallback):")
        for intent in missing:
            print(f"  - {intent}")


if __name__ == "__main__":
    main()