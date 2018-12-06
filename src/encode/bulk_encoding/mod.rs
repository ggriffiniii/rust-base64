use byteorder::{BigEndian, ByteOrder};
use Encoding;

pub mod arch;

// TODO: remove the dead_code bypass once a custom alphabet encoding is provided.
#[allow(dead_code)]
struct ScalarBulkEncoding<C>(C);
impl<C> ScalarBulkEncoding<C> {
    const INPUT_CHUNK_BYTES_READ: usize = 26;
    const INPUT_CHUNK_BYTES_ENCODED: usize = 24;
    const OUTPUT_CHUNK_BYTES_WRITTEN: usize = 32;
}

impl<C> ::BulkEncoding for ScalarBulkEncoding<C> where C: Encoding {
    const MIN_INPUT_BYTES: usize = 26;

    #[inline]
    fn bulk_encode(self, input: &[u8], output: &mut [u8]) -> (usize, usize) {
        let mut input_index: usize = 0;

        const LOW_SIX_BITS: u64 = 0x3F;

        let last_fast_index: isize = input.len() as isize - Self::INPUT_CHUNK_BYTES_READ as isize;
        let mut output_index = 0;

        while input_index as isize <= last_fast_index {
            // Major performance wins from letting the optimizer do the bounds check once, mostly
            // on the output side
            let input_chunk = &input[input_index..(input_index + Self::INPUT_CHUNK_BYTES_READ)];
            let output_chunk = &mut output[output_index..(output_index + Self::OUTPUT_CHUNK_BYTES_WRITTEN)];

            // Hand-unrolling for 32 vs 16 or 8 bytes produces yields performance about equivalent
            // to unsafe pointer code on a Xeon E5-1650v3. 64 byte unrolling was slightly better for
            // large inputs but significantly worse for 50-byte input, unsurprisingly. I suspect
            // that it's a not uncommon use case to encode smallish chunks of data (e.g. a 64-byte
            // SHA-512 digest), so it would be nice if that fit in the unrolled loop at least once.
            // Plus, single-digit percentage performance differences might well be quite different
            // on different hardware.

            let input_u64 = BigEndian::read_u64(&input_chunk[0..]);

            output_chunk[0] = self.0.encode_u6(((input_u64 >> 58) & LOW_SIX_BITS) as u8);
            output_chunk[1] = self.0.encode_u6(((input_u64 >> 52) & LOW_SIX_BITS) as u8);
            output_chunk[2] = self.0.encode_u6(((input_u64 >> 46) & LOW_SIX_BITS) as u8);
            output_chunk[3] = self.0.encode_u6(((input_u64 >> 40) & LOW_SIX_BITS) as u8);
            output_chunk[4] = self.0.encode_u6(((input_u64 >> 34) & LOW_SIX_BITS) as u8);
            output_chunk[5] = self.0.encode_u6(((input_u64 >> 28) & LOW_SIX_BITS) as u8);
            output_chunk[6] = self.0.encode_u6(((input_u64 >> 22) & LOW_SIX_BITS) as u8);
            output_chunk[7] = self.0.encode_u6(((input_u64 >> 16) & LOW_SIX_BITS) as u8);

            let input_u64 = BigEndian::read_u64(&input_chunk[6..]);

            output_chunk[8] = self.0.encode_u6(((input_u64 >> 58) & LOW_SIX_BITS) as u8);
            output_chunk[9] = self.0.encode_u6(((input_u64 >> 52) & LOW_SIX_BITS) as u8);
            output_chunk[10] = self.0.encode_u6(((input_u64 >> 46) & LOW_SIX_BITS) as u8);
            output_chunk[11] = self.0.encode_u6(((input_u64 >> 40) & LOW_SIX_BITS) as u8);
            output_chunk[12] = self.0.encode_u6(((input_u64 >> 34) & LOW_SIX_BITS) as u8);
            output_chunk[13] = self.0.encode_u6(((input_u64 >> 28) & LOW_SIX_BITS) as u8);
            output_chunk[14] = self.0.encode_u6(((input_u64 >> 22) & LOW_SIX_BITS) as u8);
            output_chunk[15] = self.0.encode_u6(((input_u64 >> 16) & LOW_SIX_BITS) as u8);

            let input_u64 = BigEndian::read_u64(&input_chunk[12..]);

            output_chunk[16] = self.0.encode_u6(((input_u64 >> 58) & LOW_SIX_BITS) as u8);
            output_chunk[17] = self.0.encode_u6(((input_u64 >> 52) & LOW_SIX_BITS) as u8);
            output_chunk[18] = self.0.encode_u6(((input_u64 >> 46) & LOW_SIX_BITS) as u8);
            output_chunk[19] = self.0.encode_u6(((input_u64 >> 40) & LOW_SIX_BITS) as u8);
            output_chunk[20] = self.0.encode_u6(((input_u64 >> 34) & LOW_SIX_BITS) as u8);
            output_chunk[21] = self.0.encode_u6(((input_u64 >> 28) & LOW_SIX_BITS) as u8);
            output_chunk[22] = self.0.encode_u6(((input_u64 >> 22) & LOW_SIX_BITS) as u8);
            output_chunk[23] = self.0.encode_u6(((input_u64 >> 16) & LOW_SIX_BITS) as u8);

            let input_u64 = BigEndian::read_u64(&input_chunk[18..]);

            output_chunk[24] = self.0.encode_u6(((input_u64 >> 58) & LOW_SIX_BITS) as u8);
            output_chunk[25] = self.0.encode_u6(((input_u64 >> 52) & LOW_SIX_BITS) as u8);
            output_chunk[26] = self.0.encode_u6(((input_u64 >> 46) & LOW_SIX_BITS) as u8);
            output_chunk[27] = self.0.encode_u6(((input_u64 >> 40) & LOW_SIX_BITS) as u8);
            output_chunk[28] = self.0.encode_u6(((input_u64 >> 34) & LOW_SIX_BITS) as u8);
            output_chunk[29] = self.0.encode_u6(((input_u64 >> 28) & LOW_SIX_BITS) as u8);
            output_chunk[30] = self.0.encode_u6(((input_u64 >> 22) & LOW_SIX_BITS) as u8);
            output_chunk[31] = self.0.encode_u6(((input_u64 >> 16) & LOW_SIX_BITS) as u8);

            input_index += Self::INPUT_CHUNK_BYTES_ENCODED;
            output_index += Self::OUTPUT_CHUNK_BYTES_WRITTEN;
        }
        (input_index, output_index)
    }
}

impl<C> ::private::Sealed for ScalarBulkEncoding<C> {}