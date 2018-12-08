use encode::bulk_encoding;

impl bulk_encoding::IntoBulkEncoding for ::StandardAlphabet {
    type BulkEncoding = bulk_encoding::ScalarBulkEncoding<Self>;

    #[inline]
    fn into_bulk_encoding(self) -> Self::BulkEncoding {
        bulk_encoding::ScalarBulkEncoding(self)
    }
}

impl bulk_encoding::IntoBulkEncoding for ::UrlSafeAlphabet {
    type BulkEncoding = bulk_encoding::ScalarBulkEncoding<Self>;

    #[inline]
    fn into_bulk_encoding(self) -> Self::BulkEncoding {
        bulk_encoding::ScalarBulkEncoding(self)
    }
}

impl bulk_encoding::IntoBulkEncoding for ::CryptAlphabet {
    type BulkEncoding = bulk_encoding::ScalarBulkEncoding<Self>;

    #[inline]
    fn into_bulk_encoding(self) -> Self::BulkEncoding {
        bulk_encoding::ScalarBulkEncoding(self)
    }
}