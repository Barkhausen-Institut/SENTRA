use occlum_sgx::SGXQuote;

pub fn generate_attestation_report(fake_attestation:bool) ->  Vec<u8> {

    if fake_attestation
    {
        println!("Generate fake attestation report...");
        // Define empty quote
        let quote = [0u8; 64];
        return quote.to_vec();
    }
    else
    {
        println!("Generate attestation report...");
        // 1. Define your 64-byte report data
        let report_data = [0u8; 64];
        // 2. Generate the SGX Quote

        let quote = SGXQuote::from_report_data(&report_data).unwrap();
        // 3. (Optional) Inspect the quote metadata

        println!("MrEnclave: {:?}", quote.mrenclave());
        println!("MrSigner:  {:?}", quote.mrsigner());
        // 4. Convert to bytes for transmission to a remote verifier

        let quote = quote.as_slice();
        return quote.to_vec();
    }
}

