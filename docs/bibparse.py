"""Very simple bibtex parser for use in MSMBuilder doc generation

Matthew Harrigan
(c)2016, MIT License
"""

from pyparsing import CaselessKeyword as kwd
from pyparsing import QuotedString, Word, alphanums, Suppress, OneOrMore, nums, \
    Group, Optional, ZeroOrMore, alphas, alphas8bit, delimitedList

# Change these if you need more flexibility:
entry_type = kwd("article") | kwd("unpublished")
cite_key = Word(alphanums + ":/._")

LCURLY = Suppress('{')
RCURLY = Suppress('}')
COMMA = Suppress(',')
AT = Suppress('@')
EQUALS = Suppress('=')

field_val = Word(nums) | QuotedString('{', endQuoteChar='}', multiline=True,
                                      convertWhitespaceEscapes=False)
title_field = Group(kwd('title') + EQUALS + field_val)
journal_field = Group(kwd('journal') + EQUALS + field_val)
year_field = Group(kwd('year') + EQUALS + field_val)
volume_field = Group(kwd('volume') + EQUALS + field_val)
pages_field = Group(kwd('pages') + EQUALS + field_val)
abstract_field = Group(kwd('abstract') + EQUALS + field_val)
doi_field = Group(kwd('doi') + EQUALS + field_val)
other_field = Group(Word(alphanums) + EQUALS + field_val)

author = OneOrMore(~kwd('and') + Word(alphas + alphas8bit + '.,-'))
author.setParseAction(lambda xx: ' '.join(str(x) for x in xx))
author_list = LCURLY + delimitedList(author, 'and') + RCURLY
author_field = Group(kwd('author') + EQUALS + Group(author_list))

entry_item = (title_field | author_field | journal_field | year_field
              | volume_field | pages_field | abstract_field | doi_field
              | Suppress(other_field))


class BibEntry(object):
    def __init__(self, type, cite_key, fields):
        self.type = type
        self.cite_key = cite_key
        self.fields = fields
        self.__dict__.update(**fields)


def to_BibEntry(toks):
    return BibEntry(toks[0], toks[1], dict(toks[2:]))


entry = (AT + entry_type + LCURLY + cite_key + COMMA
         + ZeroOrMore(entry_item + COMMA) + Optional(entry_item) + RCURLY)
entry.setParseAction(to_BibEntry)
entries = OneOrMore(entry)
