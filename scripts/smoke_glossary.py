"""Pure-logic smoke test for the glossary (no model, runs in seconds): parsing, the ASR
context string, and which (source, translation) pairs a sentence gets in each direction.
    uv run python scripts/smoke_glossary.py
"""

from livecaption.glossary import Entry, asr_context, parse, translation_pairs

g = parse("""
# people and products
Flander
Andy
MPPS
联调 = integration testing
工单 = ticket   # trailing comment
ticket = 票据
API

""")

# 1. parsing: comments / blanks dropped, translations split on "=", a repeated term ignored
assert g == [
    Entry("Flander"), Entry("Andy"), Entry("MPPS"),
    Entry("联调", "integration testing"), Entry("工单", "ticket"),
    Entry("ticket", "票据"), Entry("API"),
], g
assert parse("Ticket\nticket = x") == [Entry("Ticket")]

# 2. ASR context: every term and translation once, space-separated
assert asr_context(parse("联调 = integration testing\nFlander\nFlander = Flander")) == \
    "联调 integration testing Flander"

# 3. into Chinese: English side -> Chinese side; Latin names pinned as themselves; "ticket" is
#    covered by two entries (工单 = ticket, ticket = 票据): the first one wins
pairs = translation_pairs(g, "Andy finished the integration testing for the ticket API.", True)
assert pairs == [("Andy", "Andy"), ("integration testing", "联调"), ("ticket", "工单"),
                 ("API", "API")], pairs
# a Chinese term already in a Chinese-target sentence needs no pair
assert translation_pairs(g, "我们联调一下", True) == []

# 4. into English: Chinese side -> English side; untranslated Latin terms not pinned
pairs = translation_pairs(g, "这个接口我们下午和 Flander 再联调一下，顺便看工单。", False)
assert pairs == [("联调", "integration testing"), ("工单", "ticket")], pairs

# 5. Latin terms match whole words, case-insensitively ("API" not inside "rapid")
assert translation_pairs(g, "a rapid fix", True) == []
assert translation_pairs(g, "the api is down", True) == [("API", "API")]

# 6. mixed-language source into Chinese: only the English side needs pinning
pairs = translation_pairs(g, "明天和 Flander 做 integration testing", True)
assert pairs == [("Flander", "Flander"), ("integration testing", "联调")], pairs

print("glossary smoke test PASSED")
