use encode::bulk_encoding::ScalarBulkEncoding;

impl IntoBulkEncoding for character_set::Standard {
    type BulkEncoding = ScalarBulkEncoding<Self>;

    #[inline]
    fn into_bulk_encoding(self) -> ScalarBulkEncoding<Self> {
        ScalarBulkEncoding(self)
    }
}

impl IntoBulkEncoding for character_set::UrlSafe {
    type BulkEncoding = ScalarBulkEncoding<Self>;

    #[inline]
    fn into_bulk_encoding(self) -> ScalarBulkEncoding<Self> {
        ScalarBulkEncoding(self)
    }
}

impl IntoBulkEncoding for character_set::Crypt {
    type BulkEncoding = ScalarBulkEncoding<Self>;

    #[inline]
    fn into_bulk_encoding(self) -> ScalarBulkEncoding<Self> {
        ScalarBulkEncoding(self)
    }
}