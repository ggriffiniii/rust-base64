use byteorder::{BigEndian, ByteOrder};

pub mod arch;

pub trait IntoBulkEncoding: ::private::Sealed + Copy {
    type BulkEncoding: BulkEncoding;
    fn into_bulk_encoding(self) -> Self::BulkEncoding;
}

pub trait BulkEncoding: ::private::Sealed {
    const MIN_INPUT_BYTES: usize;

    fn bulk_encode(self, input: &[u8], output: &mut [u8]) -> (usize, usize);
}

#[derive(Debug, Default, Copy, Clone)]
pub struct ScalarBulkEncoding<C>(C);
impl<C> ScalarBulkEncoding<C> where C: ::encode::Encoding {
    const INPUT_CHUNK_BYTES_READ: usize = 26;
    const INPUT_CHUNK_BYTES_ENCODED: usize = 24;
    const OUTPUT_CHUNK_BYTES_WRITTEN: usize = 32;

    #[inline]
    fn encode_chunk(self, input: u64, output: &mut [u8]) {
        const LOW_SIX_BITS: u64 = 0x3F;
        for byte_idx in 0..8 {
            let shift_amount = 64 - (byte_idx as u64+1)*6;
            let shifted_input = input >> shift_amount;
            output[byte_idx] = self.0.encode_u6((shifted_input & LOW_SIX_BITS) as u8);
        }
    }
}

impl<C> BulkEncoding for ScalarBulkEncoding<C>
where
    C: ::encode::Encoding,
{
    const MIN_INPUT_BYTES: usize = 26;

    #[inline]
    fn bulk_encode(self, input: &[u8], output: &mut [u8]) -> (usize, usize) {
        let mut input_index: usize = 0;


        let last_fast_index: isize = input.len() as isize - Self::INPUT_CHUNK_BYTES_READ as isize;
        let mut output_index = 0;

        while input_index as isize <= last_fast_index {
            // Major performance wins from letting the optimizer do the bounds check once, mostly
            // on the output side
            let input_chunk = &input[input_index..(input_index + Self::INPUT_CHUNK_BYTES_READ)];
            let output_chunk =
                &mut output[output_index..(output_index + Self::OUTPUT_CHUNK_BYTES_WRITTEN)];

            // Hand-unrolling for 32 vs 16 or 8 bytes produces yields performance about equivalent
            // to unsafe pointer code on a Xeon E5-1650v3. 64 byte unrolling was slightly better for
            // large inputs but significantly worse for 50-byte input, unsurprisingly. I suspect
            // that it's a not uncommon use case to encode smallish chunks of data (e.g. a 64-byte
            // SHA-512 digest), so it would be nice if that fit in the unrolled loop at least once.
            // Plus, single-digit percentage performance differences might well be quite different
            // on different hardware.
            for chunk in 0..4 {
                let input = BigEndian::read_u64(&input_chunk[chunk*6..]);
                self.encode_chunk(input, &mut output_chunk[chunk*8..chunk*8+8]);
            }
            input_index += Self::INPUT_CHUNK_BYTES_ENCODED;
            output_index += Self::OUTPUT_CHUNK_BYTES_WRITTEN;
        }
        (input_index, output_index)
    }
}

impl<C> ::private::Sealed for ScalarBulkEncoding<C> {}

impl IntoBulkEncoding for &::CustomConfig {
    type BulkEncoding = ScalarBulkEncoding<Self>;

    #[inline]
    fn into_bulk_encoding(self) -> ScalarBulkEncoding<Self> {
        ScalarBulkEncoding(self)
    }
}
