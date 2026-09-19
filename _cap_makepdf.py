from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

writer = PdfWriter()
page = writer.add_blank_page(612, 792)
text = (
    "The packing list shows quantities 120, 45, 7, and 300 units. "
    "Question: what is their sum?"
)
content = f"BT /F1 14 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
stream = DecodedStreamObject()
stream.set_data(content)
stream_ref = writer._add_object(stream)
page[NameObject("/Contents")] = stream_ref
font = DictionaryObject()
font[NameObject("/Type")] = NameObject("/Font")
font[NameObject("/Subtype")] = NameObject("/Type1")
font[NameObject("/BaseFont")] = NameObject("/Helvetica")
font_ref = writer._add_object(font)
resources = DictionaryObject()
fonts = DictionaryObject()
fonts[NameObject("/F1")] = font_ref
resources[NameObject("/Font")] = fonts
page[NameObject("/Resources")] = resources
with open("_cap_note.pdf", "wb") as handle:
    writer.write(handle)
print("pdf-written")
