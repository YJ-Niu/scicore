"""Pure-Python replacement for the deleted Cython ``_mio5_utils`` extension.

Provides the MATLAB v5 variable reader used by ``scicore.io.matlab._mio5``.

The on-disk layout matches ``VarWriter5`` in ``_mio5.py``:

- full tag: 8 bytes (u4 mdtype + u4 byte_count); the data follows
  immediately (this port has no data_pos field), padded to a 64-bit
  boundary;
- small tag: 8 bytes (u4 (byte_count << 16 | mdtype) + 4 data bytes);
- disambiguation: a first u4 word with ``word >> 16 == 0`` is a full tag
  (the word is the mdtype); ``1 <= word >> 16 <= 4`` is a small tag — the
  writer therefore emits zero-byte elements as full tags;
- array flags: 16 raw bytes (u4 data_type, u4 byte_count, u4 flags_class,
  u4 nzmax), followed by tagged dims and name elements.

Supported on read: all numeric matrix classes (including complex and
logical) and char. Cell/struct/sparse/object variables raise
``MatReadError``, which ``get_variables`` degrades to a warning.
"""

from __future__ import annotations

import struct

import rsnumpy as np

from ._miobase import MatReadError
from ._mio5_params import (
    miINT8, miINT16, miINT32, miINT64, miSINGLE, miDOUBLE,
    miUINT8, miUINT16, miUINT32, miUINT64, miUTF16, miUTF32,
    mat_struct,
    mxCHAR_CLASS, mxDOUBLE_CLASS, mxINT8_CLASS, mxINT16_CLASS,
    mxINT32_CLASS, mxINT64_CLASS, mxSINGLE_CLASS,
    mxUINT8_CLASS, mxUINT16_CLASS, mxUINT32_CLASS, mxUINT64_CLASS,
)

__all__ = ['VarReader5']

# mdtype -> struct format character for numeric elements
_MDTYPE_TO_STRUCT = {
    miINT8: 'b', miUINT8: 'B', miINT16: 'h', miUINT16: 'H',
    miINT32: 'i', miUINT32: 'I', miINT64: 'q', miUINT64: 'Q',
    miSINGLE: 'f', miDOUBLE: 'd',
}

_SUPPORTED_MCLASSES = frozenset({
    mxDOUBLE_CLASS, mxSINGLE_CLASS, mxINT8_CLASS, mxUINT8_CLASS,
    mxINT16_CLASS, mxUINT16_CLASS, mxINT32_CLASS, mxUINT32_CLASS,
    mxINT64_CLASS, mxUINT64_CLASS, mxCHAR_CLASS,
})


class VarReader5:
    """MATLAB v5 variable reader (see the module docstring for the layout)."""

    def __init__(self, file_reader):
        self.file_reader = file_reader
        self.mat_stream = file_reader.mat_stream
        self.byte_order = file_reader.byte_order

    def set_stream(self, stream):
        """Read from `stream` (used for compressed matrices)."""
        self.mat_stream = stream

    def _read(self, nbytes):
        raw = self.mat_stream.read(nbytes)
        if len(raw) != nbytes:
            raise MatReadError("unexpected end of stream")
        return raw

    def _unpack(self, fmt):
        fmt = self.byte_order + fmt
        return struct.unpack(fmt, self._read(struct.calcsize(fmt)))

    def read_full_tag(self):
        """Read an 8-byte full tag, return (mdtype, byte_count)."""
        return self._unpack('II')

    def read_data(self):
        """Read one tagged element, return (mdtype, data_bytes)."""
        word = self._unpack('I')[0]
        byte_count = word >> 16
        if byte_count == 0:
            # full tag: the word is the mdtype, the byte count follows
            mdtype = word
            byte_count = self._unpack('I')[0]
            data = self._read(byte_count)
            pad = (8 - byte_count % 8) % 8
            if pad:
                self._read(pad)
        else:
            mdtype = word & 0xFFFF
            data = self._read(4)[:byte_count]
        return mdtype, data

    def _decode_data(self, data, mdtype):
        """Decode a raw element's bytes into an array.

        rsnumpy's frombuffer is broken (returns wrong size/data), so decode
        the bytes with struct into a native Python list and build the array
        from that list.
        """
        fmt = _MDTYPE_TO_STRUCT.get(mdtype)
        if fmt is None:
            raise MatReadError(f"unsupported data type {mdtype}")
        n = len(data) // struct.calcsize(fmt)
        if n == 0:
            return np.array([])
        values = list(struct.unpack(self.byte_order + f'{n}{fmt}', data))
        return np.array(values)

    def read_header(self, check_stream_limit=False):
        """Read flags/dims/name after a miMATRIX tag; return a header."""
        _data_type, _dims_bytes, flags_class, nzmax = self._unpack('IIII')
        mclass = flags_class & 0xFF
        is_logical = bool((flags_class >> 9) & 1)
        is_global = bool((flags_class >> 10) & 1)
        is_complex = bool((flags_class >> 11) & 1)
        _dims_mdtype, dims_data = self.read_data()
        ndim = len(dims_data) // 4
        if ndim:
            shape = tuple(struct.unpack(f'{self.byte_order}{ndim}I',
                                        dims_data))
        else:
            shape = ()
        name_mdtype, name_data = self.read_data()
        if name_mdtype == miUINT16:
            codec = 'utf-16-le' if self.byte_order == '<' else 'utf-16-be'
        else:
            codec = 'latin-1'
        hdr = mat_struct()
        hdr.name = name_data.decode(codec).encode('latin-1', 'replace')
        hdr.is_global = is_global
        hdr.is_complex = is_complex
        hdr.is_logical = is_logical
        hdr.mclass = mclass
        hdr.ndim = len(shape)
        hdr.shape = shape
        hdr.nzmax = nzmax
        return hdr

    def array_from_header(self, header, process=True):
        """Read the data element(s) described by `header`."""
        mclass = header.mclass
        shape = header.shape
        if mclass == mxCHAR_CLASS:
            return self._read_char(shape)
        if mclass not in _SUPPORTED_MCLASSES:
            raise MatReadError(
                f"unsupported matrix class {mclass} (cell/struct/sparse/"
                "object are not supported by this reader yet)")
        mdtype, data = self.read_data()
        arr = self._decode_data(data, mdtype)
        if header.is_complex:
            mdtype_i, data_i = self.read_data()
            arr = arr + 1j * self._decode_data(data_i, mdtype_i)
        # logicals are stored as mxUINT8_CLASS with the is_logical flag bit
        if header.is_logical:
            arr = arr.astype(bool)
        if shape and arr.size:
            if len(shape) == 2:
                # data is column-major (F order) and rsnumpy's reshape has
                # no order argument: reshape the transposed shape, then .T
                arr = arr.reshape(shape[::-1]).T
            else:
                arr = arr.reshape(shape)
        return arr

    def _read_char(self, shape):
        mdtype, data = self.read_data()
        if len(data) == 0:
            return ''
        if mdtype == miUINT16:
            codec = 'utf-16-le' if self.byte_order == '<' else 'utf-16-be'
        elif mdtype == miUTF16:
            codec = 'utf-16'
        elif mdtype == miUTF32:
            codec = 'utf-32'
        else:  # miINT8 / miUTF8
            codec = 'latin-1'
        text = data.decode(codec)
        if len(shape) <= 1:
            return text
        return np.array(list(text))
